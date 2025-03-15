import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from mem_train.dataset import MEM_build_Dataset  
from mem_vae_utils import(
    load_instructblip_model_lavis,
    prepare_blip2_embeddings,
    generate_mem_prompt
)
from blip2_conditioned_VAE import create_MemMapVAE
from constants import OBJECT_CATEGORIES
import os
from datetime import datetime
import json

def custom_collate_fn(batch):
    batch_dict = {}
    batch_dict["local_map"] = torch.stack([sample["local_map"] for sample in batch], dim=0)
    batch_dict["rgb_views"] = torch.stack([sample["rgb_views"] for sample in batch], dim=0)
    batch_dict["onehot_info"] = torch.stack([sample["onehot_info"] for sample in batch], dim=0)

    # Only include "blip2_embeds" if every sample has it.
    if all("blip2_embeds" in sample and sample["blip2_embeds"] is not None for sample in batch):
        batch_dict["blip2_embeds"] = torch.stack([sample["blip2_embeds"] for sample in batch], dim=0)
    
    return batch_dict

# loss calculation
def compute_recon_loss(logits, target, class_weights=None, ignore_index=0):
    if class_weights is not None:
        class_weights = class_weights.to(logits.device)
    return F.cross_entropy(logits, target, weight=class_weights, ignore_index=ignore_index)

def compute_kl_loss(mu, logvar):
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl_loss.mean()

def compute_aux_loss(aux_out, aux_target):
    aux_target = aux_target.float()
    return F.binary_cross_entropy_with_logits(aux_out, aux_target)

def compute_vae_loss(
    target,       # Ground truth map, shape (B, H, W)
    logits,       # Decoder output logits, shape (B, num_classes, H, W)
    mu, logvar,   # Latent parameters, each of shape (B, latent_dim)
    class_weights=None, ignore_index=0,  # Parameters for reconstruction loss
    beta=1.0,     # Weight for the KL divergence loss
    aux_out=None, aux_target=None, aux_weight=1.0  # Auxiliary loss parameters
):
    recon_loss = compute_recon_loss(logits, target, class_weights, ignore_index)
    kl_loss = compute_kl_loss(mu, logvar)
    
    total_loss = recon_loss + beta * kl_loss
    aux_loss = None
    if aux_out is not None and aux_target is not None:
        aux_loss = compute_aux_loss(aux_out, aux_target)
        total_loss = total_loss + aux_weight * aux_loss
        
    return total_loss, recon_loss, kl_loss, aux_loss

train_config = {
    "model_name": "mem_map_vae",
    "dataset_name": "gibson",
    "enable_oh_aux_task": True, # if have oh prediction tasks
    "num_epochs": 40,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "beta": 1.0,
    "aux_weight": 1.0,
    "class_weights": None # torch.tensor([0.0, 0.1, 0.1] + [1.0] * (18 - 3))
}

model_config = {
    "latent_dim": 128,
    "cond_dim": 768,  # mem condition input
    "num_classes": 18,  # map class to predict
    "emb_dim": 32,
    "transformer_dim": 128,
    "n_transformer_layers": 2,
    "n_heads": 4,
    "encoder_use_cond_input": True,   # Use late conditioning in encoder.
    "decoder_use_cond_input": True,     # Use late conditioning in decoder.
    "mem_generator_token_dim": 768,
    "mem_generator_aggregated_dim": 768,
    "mem_generator_use_attention": True,
    "oh_aux_task": train_config["enable_oh_aux_task"]
}

if __name__ == "__main__":
    # load training data
    IMAGE_MAP_DIR = "data/semantic_maps/gibson/image_map_pairs"
    dataset = MEM_build_Dataset(
        root_dir=IMAGE_MAP_DIR,
        view_wise_oh=False, # False, one hot existence for the whole local map
        shuffle_views=True # randomly shuffle views' order
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=train_config["batch_size"], 
        shuffle=True, 
        num_workers=4,
        collate_fn=custom_collate_fn  # Use the collate function defined earlier.
    )
    
    #####################################################################################################
    # load training model
    mem_generator, map_vae = create_MemMapVAE(model_config)

    # Initialize wandb.
    task_name = f"{train_config['model_name']}_{train_config['dataset_name']}"
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(project=task_name, config=train_config, name=f"{task_name}_{start_time}")
    
    # optimizer = optim.Adam([{"params": map_vae.encoder.parameters(), "lr": 1e-4},{"params": map_vae.decoder.parameters(), "lr": 1e-4},{"params": mem_generator.parameters(), "lr": 1e-5}], weight_decay=1e-5)
    optimizer = optim.Adam(
        list(map_vae.parameters()) + list(mem_generator.parameters()),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"]
    )

    map_vae.train()
    mem_generator.train()
    
    global_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_vae.to(device)
    mem_generator.to(device)
    
    # Ensure checkpoints directory exists.
    MODEL_SAVE_DIR = "mem_train/checkpoints"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    meta_file = os.path.join(MODEL_SAVE_DIR, f"{task_name}_meta.json")
    
    # Check if continuing training.
    start_epoch = 0
    best_loss = float("inf")
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            meta_data = json.load(f)
        start_epoch = meta_data.get("last_epoch", 0)
        best_loss = meta_data.get("best_loss", float("inf"))
        latest_checkpoint = meta_data.get("latest_checkpoint", None)
        if latest_checkpoint and os.path.exists(latest_checkpoint):
            checkpoint = torch.load(latest_checkpoint)
            map_vae.load_state_dict(checkpoint["map_vae_state_dict"])
            mem_generator.load_state_dict(checkpoint["mem_generator_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            global_step = checkpoint.get("global_step", 0)
        print(f"Resumed training from epoch {start_epoch+1}, best loss {best_loss:.4f}")
    
        
    for epoch in range(train_config["num_epochs"]):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{train_config['num_epochs']}")
        for batch in pbar:
            local_map_batch = batch["local_map"].to(device)  # (B, 65, 65)
            rgb_views_batch = batch["rgb_views"].to(device)    # (B, 4, 3, 1024, 1024)
            oh_batch = batch["onehot_info"].to(device)         # (B, 17) or (B, 4, 17)
            if "blip2_embeds" in batch:
                blip2_embeds_batch = batch["blip2_embeds"].to(device)  # (B, 4, 32, 768)
            else:
                try:
                    blip2_embeds_batch = prepare_blip2_embeddings(
                        blip2_model, 
                        vis_processors, 
                        txt_processors, 
                        rgb_views_batch, 
                        mem_prompt,
                        device
                    )
                except: # first time loading
                    blip2_model_name="blip2_t5_instruct"
                    blip2_model, vis_processors, txt_processors = load_instructblip_model_lavis(blip2_model_name)
                    blip2_model.to(device)
                    mem_prompt = generate_mem_prompt(OBJECT_CATEGORIES[train_config['dataset_name']])

            # Generate the mem_condition using the MemGenerator.
            mem_condition_batch, _ = mem_generator(blip2_embeds_batch)  # shape: (B, cond_dim)
            
            # Forward pass through the VAE.
            if model_config["oh_aux_task"] :
                logits, mu, logvar, aux_out = map_vae(local_map_batch, mem_condition_batch)
            else:
                logits, mu, logvar = map_vae(local_map_batch, mem_condition_batch)
                aux_out = None

            # Compute losses.
            total_loss, recon_loss, kl_loss, aux_loss = compute_vae_loss(
                target=local_map_batch,          # (B, 65, 65)
                logits=logits,                   # (B, num_classes, 65, 65)
                mu=mu, logvar=logvar,             # (B, latent_dim)
                class_weights=train_config.get("class_weights", None),
                ignore_index=0,                  # Assuming 0 is out-of-bound.
                beta=train_config.get("beta", 1.0),
                aux_out=aux_out,
                aux_target=oh_batch.float(),     # (B, 17)
                aux_weight=train_config.get("aux_weight", 1.0)
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            global_step += 1
            epoch_loss += total_loss.item()
            
            # Log metrics to wandb.
            wandb.log({
                "epoch": epoch + 1,
                "step": global_step,
                "loss/total": total_loss.item(),
                "loss/recon": recon_loss.item(),
                "loss/kl": kl_loss.item(),
                "loss/aux": aux_loss.item() if aux_loss is not None else 0.0
            })
            
            pbar.set_postfix({
                "Total": f"{total_loss.item():.4f}",
                "Recon": f"{recon_loss.item():.4f}",
                "KL": f"{kl_loss.item():.4f}",
                "Aux": f"{aux_loss.item():.4f}" if aux_loss is not None else "0.0000"
            })
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        latest_checkpoint_path = f"{MODEL_SAVE_DIR}/vae_checkpoint_latest.pth"
        checkpoint = {
            "map_vae_state_dict": map_vae.state_dict(),
            "mem_generator_state_dict": mem_generator.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "train_config": train_config,
            "model_config": model_config
        }
        torch.save(checkpoint, latest_checkpoint_path)
        wandb.save(latest_checkpoint_path)
        
        # Update meta data.
        meta_data = {
            "last_epoch": epoch + 1,
            "global_step": global_step,
            "latest_checkpoint": latest_checkpoint_path,
            "best_loss": best_loss
        }
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_checkpoint_path = f"{MODEL_SAVE_DIR}/vae_checkpoint_best.pth"
            torch.save(checkpoint, best_checkpoint_path)
            meta_data["best_loss"] = best_loss
            meta_data["best_checkpoint"] = best_checkpoint_path
            meta_data["best_epoch"] = epoch
            wandb.save(best_checkpoint_path)
        
        with open(meta_file, "w") as f:
            json.dump(meta_data, f)

    wandb.finish()

