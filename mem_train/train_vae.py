import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from mem_train.dataset import MEM_build_Dataset  

def custom_collate_fn(batch):
    """
    Custom collate function for MEM_build_Dataset.
    
    Expects each sample (dict) to contain:
      - "local_map": Tensor of shape (1, 65, 65)
      - "rgb_views": Tensor of shape (4, 3, 1024, 1024)
      - "onehot_views": Tensor of shape (4, 17)
      - "h5_path": str
      - Optionally "blip2_embeds": Tensor of shape (4, 32, 768)
    
    Returns a dictionary with batched tensors:
      - "local_map": (B, 1, 65, 65)
      - "rgb_views": (B, 4, 3, 1024, 1024)
      - "onehot_views": (B, 4, 17)
      - "blip2_embeds": (B, 4, 32, 768) if all samples have it; otherwise, it is omitted.
      - "h5_path": list of strings
    """
    batch_dict = {}
    batch_dict["local_map"] = torch.stack([sample["local_map"] for sample in batch], dim=0)
    batch_dict["rgb_views"] = torch.stack([sample["rgb_views"] for sample in batch], dim=0)
    batch_dict["onehot_views"] = torch.stack([sample["onehot_views"] for sample in batch], dim=0)

    # Only include "blip2_embeds" if every sample has it.
    if all("blip2_embeds" in sample and sample["blip2_embeds"] is not None for sample in batch):
        batch_dict["blip2_embeds"] = torch.stack([sample["blip2_embeds"] for sample in batch], dim=0)
    
    return batch_dict

task_config = {
    "num_epochs": 10,
    "batch_size": 4
}


if __name__ == "__main__":
    IMAGE_MAP_DIR = "data/semantic_maps/gibson/image_map_pairs"
    dataset = MEM_build_Dataset(root_dir=IMAGE_MAP_DIR)
    dataloader = DataLoader(
        dataset, 
        batch_size=task_config["batch_size"], 
        shuffle=True, 
        num_workers=4,
        collate_fn=custom_collate_fn  # Use the collate function defined earlier.
    )


    for epoch in range(task_config["num_epochs"]):
        for batch in dataloader:
            print("Local map shape:", batch["local_map"].shape)    # e.g., (B, 1, 65, 65)
            print("RGB views shape:", batch["rgb_views"].shape)      # e.g., (B, 4, 3, 1024, 1024)
            print("One-hot views shape:", batch["onehot_views"].shape) # e.g., (B, 4, 17)
            if "blip2_embeds" in batch:
                print("BLIP2 embeds shape:", batch["blip2_embeds"].shape)
            break  # Remove break to iterate over the full dataset.



    # # ---------------------------
    # # Initialize wandb
    # # ---------------------------
    # wandb.init(project="transformer_cvae_map_gen", config=config)
    # # ---------------------------
    # # Instantiate model, optimizer, loss, etc.
    # # ---------------------------
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # # We'll use CrossEntropyLoss for token reconstruction.
    # # Note: CrossEntropyLoss in PyTorch expects logits with shape (N, C) and targets with shape (N).
    # criterion = nn.CrossEntropyLoss()

    # # ---------------------------
    # # Dummy Dataset (Replace with your own DataLoader)
    # # ---------------------------
    # def get_dummy_batch(batch_size, src_seq_len, tgt_seq_len, encoder_input_dim, vocab_size):
    #     # Encoder inputs: e.g., BLIP-2 tokens (float tensor)
    #     encoder_inputs = torch.randn(batch_size, src_seq_len, encoder_input_dim)
    #     # Target input tokens: dummy integers (for decoder input)
    #     tgt_input = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
    #     # Target labels: shift tgt_input by one or use separate target tokens.
    #     # For simplicity, we use the same tgt_input as target (in practice, shift for language models).
    #     tgt_labels = tgt_input.clone()
    #     return encoder_inputs, tgt_input, tgt_labels

    # # ---------------------------
    # # Training Loop
    # # ---------------------------
    # global_step = 0
    # for epoch in range(config["num_epochs"]):
    #     model.train()
    #     epoch_loss = 0.0
    #     num_batches = 100  # dummy: set number of batches per epoch (replace with len(dataloader))
    #     for batch_idx in range(num_batches):
    #         # Get dummy batch
    #         encoder_inputs, tgt_input, tgt_labels = get_dummy_batch(
    #             config["batch_size"],
    #             config["src_seq_len"],
    #             config["tgt_seq_len"],
    #             config["encoder_input_dim"],
    #             config["vocab_size"]
    #         )
    #         encoder_inputs = encoder_inputs.to(device)
    #         tgt_input = tgt_input.to(device)
    #         tgt_labels = tgt_labels.to(device)
            
    #         optimizer.zero_grad()
    #         logits, mu, logvar = model(encoder_inputs, tgt_input)
    #         # logits: (batch, tgt_seq_len, vocab_size)
    #         # Flatten logits and targets for loss computation.
    #         logits_flat = logits.view(-1, config["vocab_size"])
    #         tgt_labels_flat = tgt_labels.view(-1)
    #         recon_loss = criterion(logits_flat, tgt_labels_flat)
    #         # Compute KL divergence loss:
    #         kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1)
    #         kl_loss = kl_loss.mean()
            
    #         loss = recon_loss + kl_loss
            
    #         loss.backward()
    #         optimizer.step()
            
    #         epoch_loss += loss.item()
    #         global_step += 1
            
    #         if batch_idx % config["log_interval"] == 0:
    #             wandb.log({
    #                 "train_loss": loss.item(),
    #                 "recon_loss": recon_loss.item(),
    #                 "kl_loss": kl_loss.item(),
    #                 "epoch": epoch,
    #                 "global_step": global_step
    #             })
    #             print(f"Epoch [{epoch}/{config['num_epochs']}], Batch [{batch_idx}/{num_batches}], Loss: {loss.item():.4f}")
        
    #     avg_epoch_loss = epoch_loss / num_batches
    #     wandb.log({"avg_epoch_loss": avg_epoch_loss, "epoch": epoch})
    #     print(f"==> Epoch [{epoch}] Average Loss: {avg_epoch_loss:.4f}")

    # # Save the model checkpoint.
    # torch.save(model.state_dict(), "transformer_cvae_map_gen.pt")
    # wandb.save("transformer_cvae_map_gen.pt")
