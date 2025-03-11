import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from cVAE_transformer import TransformerCVAE  # assuming your model is saved in transformer_cvae.py

# ---------------------------
# Configuration & Hyperparameters
# ---------------------------
config = {
    "encoder_input_dim": 768,   # BLIP-2 token dimension
    "model_dim": 512,
    "latent_dim": 64,
    "enc_layers": 4,
    "dec_layers": 4,
    "num_heads": 8,
    "dropout": 0.1,
    "max_seq_len": 256,         # maximum output token sequence length
    "vocab_size": 12,           # e.g., 10 semantic classes + special tokens
    "use_attn_pooling": True,
    "batch_size": 8,
    "src_seq_len": 10,          # number of encoder tokens (e.g., from BLIP-2)
    "tgt_seq_len": 20,          # target token sequence length (map tokens)
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "log_interval": 10
}

# ---------------------------
# Initialize wandb
# ---------------------------
wandb.init(project="transformer_cvae_map_gen", config=config)

# ---------------------------
# Instantiate model, optimizer, loss, etc.
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerCVAE(
    encoder_input_dim=config["encoder_input_dim"],
    vocab_size=config["vocab_size"],
    model_dim=config["model_dim"],
    latent_dim=config["latent_dim"],
    enc_layers=config["enc_layers"],
    dec_layers=config["dec_layers"],
    num_heads=config["num_heads"],
    dropout=config["dropout"],
    max_seq_len=config["max_seq_len"],
    use_attn_pooling=config["use_attn_pooling"]
).to(device)

optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# We'll use CrossEntropyLoss for token reconstruction.
# Note: CrossEntropyLoss in PyTorch expects logits with shape (N, C) and targets with shape (N).
criterion = nn.CrossEntropyLoss()

# ---------------------------
# Dummy Dataset (Replace with your own DataLoader)
# ---------------------------
def get_dummy_batch(batch_size, src_seq_len, tgt_seq_len, encoder_input_dim, vocab_size):
    # Encoder inputs: e.g., BLIP-2 tokens (float tensor)
    encoder_inputs = torch.randn(batch_size, src_seq_len, encoder_input_dim)
    # Target input tokens: dummy integers (for decoder input)
    tgt_input = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
    # Target labels: shift tgt_input by one or use separate target tokens.
    # For simplicity, we use the same tgt_input as target (in practice, shift for language models).
    tgt_labels = tgt_input.clone()
    return encoder_inputs, tgt_input, tgt_labels

# ---------------------------
# Training Loop
# ---------------------------
global_step = 0
for epoch in range(config["num_epochs"]):
    model.train()
    epoch_loss = 0.0
    num_batches = 100  # dummy: set number of batches per epoch (replace with len(dataloader))
    for batch_idx in range(num_batches):
        # Get dummy batch
        encoder_inputs, tgt_input, tgt_labels = get_dummy_batch(
            config["batch_size"],
            config["src_seq_len"],
            config["tgt_seq_len"],
            config["encoder_input_dim"],
            config["vocab_size"]
        )
        encoder_inputs = encoder_inputs.to(device)
        tgt_input = tgt_input.to(device)
        tgt_labels = tgt_labels.to(device)
        
        optimizer.zero_grad()
        logits, mu, logvar = model(encoder_inputs, tgt_input)
        # logits: (batch, tgt_seq_len, vocab_size)
        # Flatten logits and targets for loss computation.
        logits_flat = logits.view(-1, config["vocab_size"])
        tgt_labels_flat = tgt_labels.view(-1)
        recon_loss = criterion(logits_flat, tgt_labels_flat)
        # Compute KL divergence loss:
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1)
        kl_loss = kl_loss.mean()
        
        loss = recon_loss + kl_loss
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        global_step += 1
        
        if batch_idx % config["log_interval"] == 0:
            wandb.log({
                "train_loss": loss.item(),
                "recon_loss": recon_loss.item(),
                "kl_loss": kl_loss.item(),
                "epoch": epoch,
                "global_step": global_step
            })
            print(f"Epoch [{epoch}/{config['num_epochs']}], Batch [{batch_idx}/{num_batches}], Loss: {loss.item():.4f}")
    
    avg_epoch_loss = epoch_loss / num_batches
    wandb.log({"avg_epoch_loss": avg_epoch_loss, "epoch": epoch})
    print(f"==> Epoch [{epoch}] Average Loss: {avg_epoch_loss:.4f}")

# Save the model checkpoint.
torch.save(model.state_dict(), "transformer_cvae_map_gen.pt")
wandb.save("transformer_cvae_map_gen.pt")
