import torch
import torch.nn as nn
import torch.nn.functional as F
from blip2_conditioned_VQVAE import TopdownMapEncoder, TopdownMapDecoder, TopdownMapVQVAE

#############################################
# Multi-View Conditioning Modules
#############################################
class RGBDepthFusion(nn.Module):
    def __init__(self, input_dim=2048, fused_dim=1024):
        super().__init__()
        self.fc = nn.Linear(2 * input_dim, fused_dim)
    def forward(self, rgb, depth):
        x = torch.cat([rgb, depth], dim=-1)
        return F.relu(self.fc(x))

class MultiViewFusion(nn.Module):
    def __init__(self, token_dim, n_transformer_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        encoder = nn.TransformerEncoderLayer(d_model=token_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=n_transformer_layers)
    def forward(self, x):
        return self.transformer(x)

class MultiViewConditioner(nn.Module):
    def __init__(self, rgb_depth_dim=2048, fused_dim=1024, token_dim=1024, num_views=4, n_transformer_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.fusion = RGBDepthFusion(rgb_depth_dim, fused_dim)
        self.project = nn.Linear(fused_dim, token_dim) if fused_dim != token_dim else None
        self.direction_embeddings = nn.Parameter(torch.randn(num_views, token_dim))
        self.multi_view_fusion = MultiViewFusion(token_dim, n_transformer_layers, n_heads, dropout)
        self.global_fc = nn.Linear(token_dim, token_dim)

    def forward(self, rgb, depth):
        B, V, _ = rgb.shape
        tokens = []
        for i in range(V):
            fused = self.fusion(rgb[:, i], depth[:, i])
            if self.project: fused = self.project(fused)
            fused = fused + self.direction_embeddings[i]
            tokens.append(fused.unsqueeze(1))
        tokens = torch.cat(tokens, dim=1)
        fused_tokens = self.multi_view_fusion(tokens)
        global_cond = self.global_fc(fused_tokens.mean(dim=1))
        return global_cond, fused_tokens

#############################################
# Factory Function
#############################################
def create_MemMapVQVAE(cfg):
    conditioner = MultiViewConditioner(
        rgb_depth_dim=cfg["rgb_depth_dim"],
        fused_dim=cfg["fused_dim"],
        token_dim=cfg["cond_dim"],
        num_views=cfg["num_views"],
        n_transformer_layers=cfg["n_transformer_layers"],
        n_heads=cfg["n_heads"],
        dropout=cfg["dropout_p"]
    )
    encoder = TopdownMapEncoder(
        embedding_dim=cfg["embedding_dim"],
        cond_dim=cfg["cond_dim"],
        num_classes=cfg["num_classes"],
        emb_dim=cfg["emb_dim"],
        transformer_dim=cfg["transformer_dim"],
        n_transformer_layers=cfg["n_transformer_layers"],
        n_heads=cfg["n_heads"],
        use_cond_input=True,
        dropout_p=cfg["dropout_p"]
    )
    decoder = TopdownMapDecoder(
        embedding_dim=cfg["embedding_dim"],
        cond_dim=cfg["cond_dim"],
        num_classes=cfg["num_classes"],
        use_cond_input=True,
        dropout_p=cfg["dropout_p"]
    )
    vqvae = TopdownMapVQVAE(
        encoder, decoder,
        num_embeddings=cfg["num_embeddings"],
        embedding_dim=cfg["embedding_dim"],
        commitment_cost=cfg["commitment_cost"],
        oh_aux_task=cfg["oh_aux_task"],
        oh_aux_class=cfg["num_classes"] - 2
    )
    return conditioner, vqvae

if __name__ == "__main__":
    config = {
        "embedding_dim": 128, "num_embeddings": 512, "commitment_cost": 0.25,
        "cond_dim": 1024, "rgb_depth_dim": 2048, "fused_dim": 1024, "num_views": 4,
        "num_classes": 18, "emb_dim": 32, "transformer_dim": 128,
        "n_transformer_layers": 2, "n_heads": 4,
        "oh_aux_task": True, "dropout_p": 0.2
    }

    conditioner, vqvae = create_MemMapVQVAE(config)

    B = 2
    # Dummy top‑down map
    local_map = torch.randint(0, config["num_classes"], (B, 65, 65))
    # Dummy RGB+depth embeddings: shape (batch, 4 views, 2048)
    rgb_embeds = torch.randn(B, config["num_views"], config["rgb_depth_dim"])
    depth_embeds = torch.randn(B, config["num_views"], config["rgb_depth_dim"])

    mem_condition, _ = conditioner(rgb_embeds, depth_embeds)
    logits, vq_loss, perplexity, encoding_indices, aux_out = vqvae(local_map, mem_condition)

    print("Logits shape:", logits.shape)               # → (B, 18, 65, 65)
    print("VQ loss:", vq_loss.item())
    print("Perplexity:", perplexity.item())
    print("Encoding indices shape:", encoding_indices.shape)  # → (B * codebook_tokens,)
    print("Aux output shape:", aux_out.shape)          # → (B, num_classes-1)
    print("===== TEST COMPLETE =====")
