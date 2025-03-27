import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from mem_train.dataset import MEM_build_Dataset, custom_collate_fn
from mem_vae_utils import(
    WarmupDecayLR,
    CLIPEncoder,
    DepthEncoder,
    to_pil
)
from Embed_conditioned_VQVAE import create_MemMapVQVAE
from constants import OBJECT_CATEGORIES

# loss calculation
def compute_recon_loss(logits, target, class_weights=None, ignore_index=-100):
    if class_weights is not None:
        class_weights = class_weights.to(logits.device)
    return F.cross_entropy(logits, target, weight=class_weights, ignore_index=ignore_index)

def compute_aux_loss(aux_out, aux_target, weight=None):
    aux_target = aux_target.float()
    if weight is not None:
        weight = weight.to(aux_out.device)
    return F.binary_cross_entropy_with_logits(aux_out, aux_target, weight=weight)

def compute_vqvae_loss(
    target,       # Ground truth map, shape (B, H, W)
    logits,       # Decoder output logits, shape (B, num_classes, H, W)
    vq_loss,      # Vector quantization loss
    class_weights=None, 
    ignore_index=0,  # Parameters for reconstruction loss
    beta=1.0,     # Weight for the VQ loss
    aux_out=None, 
    aux_target=None, 
    aux_weight=1.0  # Auxiliary loss parameters
):
    recon_loss = compute_recon_loss(logits, target, class_weights, ignore_index)
    
    if vq_loss.dim() > 0:
        vq_loss = vq_loss.mean()
    
    total_loss = recon_loss + beta * vq_loss
    aux_loss = None
    if aux_out is not None and aux_target is not None:
        aux_loss = compute_aux_loss(aux_out, aux_target, class_weights[2:])
        total_loss = total_loss + aux_weight * aux_loss
        
    return total_loss, recon_loss, vq_loss, aux_loss

if __name__ == "__main__":
    
    train_config = {
        "model_name": "mem_map_vqvae_clip",
        "dataset_name": "mp3d",
        "enable_oh_aux_task": True, # if have oh prediction tasks
        "num_epochs": 60,
        "batch_size": 128,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "warm_up_steps": 500,
        "ignored_class_id": -100, # 0-17 represents the class does not take into account in loss cal
        "beta": 1,
        "aux_weight": 1,
        "class_weights": None
    }

    model_config = {
        "embedding_dim": 256,
        "num_embeddings": 512,        # Size of the codebook (dictionary)
        "commitment_cost": 0.25,      # Weight for commitment loss
        "cond_dim": 1024,              # mem condition input
        "rgb_depth_dim": 2048,
        "fused_dim": 1024,
        "num_views": 4,
        "emb_dim": 32,
        "transformer_dim": 256,
        "n_transformer_layers": 3,
        "n_heads": 8,
        "encoder_use_cond_input": True,   # Use late conditioning in encoder
        "decoder_use_cond_input": True,   # Use late conditioning in decoder
        "dropout_p": 0.2,
        "num_classes": 18,            # map class to predict
        "oh_aux_task": train_config["enable_oh_aux_task"]
    }
    # load training data
    NUM_WORKERS = 8
    IMAGE_MAP_DIR = "data/semantic_maps"
    SPLIT_RATIO = 0.9
    
    train_ds = MEM_build_Dataset(
        root_dir=IMAGE_MAP_DIR,
        dataset=train_config['dataset_name'],
        view_wise_oh=False, 
        shuffle_views=True,
        rotate_map=True,
        smooth_map=True,
        split="train",
        split_ratio=SPLIT_RATIO
    )
    
    test_ds = MEM_build_Dataset(
        root_dir=IMAGE_MAP_DIR,
        dataset=train_config['dataset_name'],
        view_wise_oh=False, 
        shuffle_views=False,
        rotate_map=True,
        smooth_map=True,
        split="test",
        split_ratio=SPLIT_RATIO
    )

    train_dataloader = DataLoader(
        train_ds,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    test_dataloader = DataLoader(
        test_ds,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        persistent_workers=True
    )
    
    #####################################################################################################
    NUM_CLASS = len(OBJECT_CATEGORIES[train_config['dataset_name']]) # no wall
    train_config['class_weights'] = torch.tensor([1.0, 1.0] + [1.0] * (NUM_CLASS - 2))
    model_config['num_classes'] = NUM_CLASS
    # load training model
    conditioner, vqvae = create_MemMapVQVAE(model_config)

    # Initialize wandb.
    project_name = f"{train_config['model_name']}_{train_config['dataset_name']}"
    task_name = f"no_wall_equal_loss"
    wandb.init(
        project=project_name, 
        config=train_config, 
        name=task_name)
    
    # optimizer = optim.Adam([{"params": vqvae.encoder.parameters(), "lr": 1e-4},{"params": vqvae.decoder.parameters(), "lr": 1e-4},{"params": conditioner.parameters(), "lr": 1e-5}], weight_decay=1e-5)
    optimizer = optim.AdamW(
        list(vqvae.parameters()) + list(conditioner.parameters()),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"]
    )
    
    total_steps = train_config["num_epochs"] * len(train_dataloader)
    warmup_steps = train_config["warm_up_steps"]
    scheduler = WarmupDecayLR(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        warmup_min_lr=0,
        warmup_max_lr=train_config["learning_rate"]
    )
    
    # Set device and move models.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae.to(device)
    conditioner.to(device)
    
    # Wrap in DataParallel if multiple GPUs are available.
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        vqvae = torch.nn.DataParallel(vqvae)
        conditioner = torch.nn.DataParallel(conditioner)
        
    # Ensure checkpoints directory exists.
    MODEL_SAVE_DIR = f"mem_train/checkpoints_{project_name}"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    meta_file = os.path.join(MODEL_SAVE_DIR, f"{task_name}_meta.json")
    
    # Check if continuing training.
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            meta_data = json.load(f)
        start_epoch = meta_data.get("last_epoch", 0)
        best_loss = meta_data.get("best_loss", float("inf"))
        latest_checkpoint = meta_data.get("latest_checkpoint", None)
        if latest_checkpoint and os.path.exists(latest_checkpoint):
            checkpoint = torch.load(latest_checkpoint)
            vqvae.load_state_dict(checkpoint["vqvae_state_dict"])
            conditioner.load_state_dict(checkpoint["conditioner_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            global_step = checkpoint.get("global_step", 0)
        print(f"Resumed training from epoch {start_epoch+1}, best loss {best_loss:.4f}")
    else:
        meta_data = {}
    
        
    for epoch in range(start_epoch, train_config["num_epochs"]):
        # Set models to train mode
        vqvae.train()
        conditioner.train()  # Keep memory generator in eval mode
        epoch_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{train_config['num_epochs']}")
        for batch in pbar:
            local_map_batch = batch["local_map"].to(device)  # (B, 65, 65)
            rgb_views_batch = batch["rgb_views"] 
            depth_views_batch = batch["depth_views"]
            oh_batch = batch["onehot_info"].to(device) # (B, NUM_CLASS -2) only for objects, no wall, no floor
            if "rgb_embeds" in batch:
                rgb_embeds_batch = batch["rgb_embeds"].to(device)  # (B, 4, 2048)
            else:
                try:
                    flat_imgs = [to_pil(view) for views in rgb_views_batch for view in views]
                    flat_feats = rgb_clip_encoder.extract_fts(flat_imgs)
                    rgb_embeds_batch = flat_feats.reshape(rgb_views_batch.shape[0], 4, -1)
                except: # first time loading
                    rgb_clip_encoder = CLIPEncoder(device)
            
            if "depth_embeds" in batch:
                depth_embeds_batch = batch["depth_embeds"].to(device)  # (B, 4, 2048)
            else:
                try:
                    B, V, H, W = depth_views_batch.shape
                    # 1️⃣ Flatten view dimension into batch, then permute to (batch*views, H, W, 1)
                    flat_depth = depth_views_batch.reshape(-1, H, W).unsqueeze(-1)
                    flat_feats = depth_encoder.extract_fts(flat_depth)

                    # 3️⃣ Reshape back into (B, num_views, feat_dim)
                    feat_dim = flat_feats.shape[-1]
                    depth_feats = torch.from_numpy(flat_feats).reshape(B, V, feat_dim)
                    depth_embeds_batch = torch.from_numpy(flat_feats).reshape(B, V, -1)
                except: # first time loading
                    depth_encoder = DepthEncoder(device)

            # Generate the mem_condition using the MemGenerator.
            mem_condition_batch, _ = conditioner(rgb_embeds_batch, depth_embeds_batch)
            
            # Forward pass through the VQVAE.
            if model_config["oh_aux_task"]:
                logits, vq_loss, _, _, aux_out = vqvae(local_map_batch, mem_condition_batch)
            else:
                logits, vq_loss, _, _ = vqvae(local_map_batch, mem_condition_batch)
                aux_out = None

            # Compute losses.
            total_loss, recon_loss, vq_loss_value, aux_loss = compute_vqvae_loss(
                target=local_map_batch,          # (B, 65, 65)
                logits=logits,                   # (B, num_classes, 65, 65)
                vq_loss=vq_loss,                # VQ loss from the model
                class_weights=train_config.get("class_weights", None),
                ignore_index=train_config.get('ignored_class_id', -100),
                beta=train_config.get("beta", 1.0),
                aux_out=aux_out,
                aux_target=oh_batch.float(),     # (B, 17)
                aux_weight=train_config.get("aux_weight", 1.0)
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()  # Update the learning rate.
            
            global_step += 1
            epoch_loss += total_loss.item()
            
            # Log metrics to wandb.
            wandb.log({
                "epoch": epoch + 1,
                "step": global_step,
                "loss/train_total": total_loss.item(),
                "loss/train_recon": recon_loss.item(),
                "loss/train_vq": vq_loss_value.item(),
                "loss/train_aux": aux_loss.item() if aux_loss is not None else 0.0
            })
            
            pbar.set_postfix({
                "Train Total": f"{total_loss.item():.4f}",
                "Train Recon": f"{recon_loss.item():.4f}",
                "Train VQ": f"{vq_loss_value.item():.4f}",
                "Train Aux": f"{aux_loss.item():.4f}" if aux_loss is not None else "0.0000"
            })
        
        avg_train_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} average train loss: {avg_train_loss:.4f}")
        
        # Evaluate on test set.
        vqvae.eval()
        conditioner.eval()
        test_total_loss = 0.0
        test_recon_loss = 0.0
        test_vq_loss = 0.0
        test_aux_loss = 0.0
        num_batches = len(test_dataloader)
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Test data measuring"):
                local_map_batch = batch["local_map"].to(device)
                rgb_views_batch = batch["rgb_views"].to(device)
                depth_views_batch = batch['depth_views'].to(device)
                oh_batch = batch["onehot_info"].to(device)
                if "rgb_embeds" in batch:
                    rgb_embeds_batch = batch["rgb_embeds"].to(device)  # (B, 4, 2048)
                else:
                    try:
                        flat_imgs = [to_pil(view) for views in rgb_views_batch for view in views]
                        flat_feats = rgb_clip_encoder.extract_fts(flat_imgs)
                        rgb_embeds_batch = flat_feats.reshape(rgb_views_batch.shape[0], 4, -1)
                    except: # first time loading
                        rgb_clip_encoder = CLIPEncoder(device)
                
                if "depth_embeds" in batch:
                    depth_embeds_batch = batch["depth_embeds"].to(device)  # (B, 4, 2048)
                else:
                    try:
                        B, V, H, W = depth_views_batch.shape
                        # 1️⃣ Flatten view dimension into batch, then permute to (batch*views, H, W, 1)
                        flat_depth = depth_views_batch.reshape(-1, H, W).unsqueeze(-1)
                        flat_feats = depth_encoder.extract_fts(flat_depth)

                        # 3️⃣ Reshape back into (B, num_views, feat_dim)
                        feat_dim = flat_feats.shape[-1]
                        depth_feats = torch.from_numpy(flat_feats).reshape(B, V, feat_dim)
                        depth_embeds_batch = torch.from_numpy(flat_feats).reshape(B, V, -1)
                    except: # first time loading
                        depth_encoder = DepthEncoder(device)

                mem_condition_batch, _ = conditioner(rgb_embeds_batch, depth_embeds_batch)
                if model_config["oh_aux_task"]:
                    logits, vq_loss, _, _, aux_out = vqvae(local_map_batch, mem_condition_batch)
                else:
                    logits, vq_loss, _, _ = vqvae(local_map_batch, mem_condition_batch)
                    aux_out = None
        
                total_loss_val, recon_loss_val, vq_loss_val, aux_loss_val = compute_vqvae_loss(
                    target=local_map_batch,
                    logits=logits,
                    vq_loss=vq_loss,
                    class_weights=train_config.get("class_weights", None),
                    ignore_index=train_config.get('ignored_class_id', -100),
                    beta=train_config.get("beta", 1.0),
                    aux_out=aux_out,
                    aux_target=oh_batch.float(),
                    aux_weight=train_config.get("aux_weight", 1.0)
                )
            
                test_total_loss += total_loss_val.item()
                test_recon_loss += recon_loss_val.item()
                test_vq_loss += vq_loss_val.item()
                test_aux_loss += aux_loss_val.item() if aux_loss_val is not None else 0.0
        
        avg_test_total = test_total_loss / num_batches
        avg_test_recon = test_recon_loss / num_batches
        avg_test_vq = test_vq_loss / num_batches
        avg_test_aux = test_aux_loss / num_batches
        print(f"Epoch {epoch+1} average test loss: Total: {avg_test_total:.4f}, Recon: {avg_test_recon:.4f}, VQ: {avg_test_vq:.4f}, Aux: {avg_test_aux:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "loss/test_total": avg_test_total,
            "loss/test_recon": avg_test_recon,
            "loss/test_vq": avg_test_vq,
            "loss/test_aux": avg_test_aux
        })
        
        # Save checkpoint
        latest_checkpoint_path = f"{MODEL_SAVE_DIR}/{task_name}_latest.pth"
        checkpoint = {
            "vqvae_state_dict": vqvae.state_dict(),
            "conditioner_state_dict": conditioner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "train_config": train_config,
            "model_config": model_config
        }
        torch.save(checkpoint, latest_checkpoint_path)
        wandb.save(latest_checkpoint_path)
        
        # Update meta data.
        # Update or add new keys while preserving existing keys.
        meta_data.update({
            "last_epoch": epoch + 1,
            "global_step": global_step,
            "latest_checkpoint": latest_checkpoint_path
        })
        
        if avg_test_total < best_loss:
            best_loss = avg_test_total
            best_checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"{task_name}_best.pth")
            torch.save(checkpoint, best_checkpoint_path)
            meta_data["best_loss"] = best_loss
            meta_data["best_checkpoint"] = best_checkpoint_path
            meta_data["best_epoch"] = epoch + 1
            wandb.save(best_checkpoint_path)
        
        with open(meta_file, "w") as f:
            json.dump(meta_data, f)

    wandb.finish()
