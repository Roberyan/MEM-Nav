import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import gelu
from PIL import Image
import numpy as np

# Assume these utility functions and constants are defined appropriately:
# load_blip2_model_lavis(blip2_name) returns (blip2_model, vis_processors, txt_processors)
# generate_mem_prompt(object_list) returns a text prompt based on the list of objects
# OBJECT_CATEGORIES is a dict with keys like "gibson" mapping to a list of objects.
# For demonstration purposes, I'll assume these are imported from your modules.
from mem_vae_utils import load_blip2_model_lavis, generate_mem_prompt
from constants import OBJECT_CATEGORIES

#######################################
#       AverageSelfAttention          #
#######################################
class AverageSelfAttention(nn.Module):
    def __init__(self, hidden_size):
        """
        Computes a weighted average over a sequence.
        Args:
            hidden_size: The dimension of the hidden state.
        """
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(hidden_size))
        self.softmax = nn.Softmax(dim=1)
        self.non_linearity = gelu

    def forward(self, inputs, attention_mask=None):
        # inputs: (batch, seq_len, hidden_size)
        scores = self.non_linearity(torch.matmul(inputs, self.attention_weights))  # (batch, seq_len)
        if attention_mask is not None:
            scores = scores + attention_mask
        weights = self.softmax(scores)  # (batch, seq_len)
        representations = torch.sum(inputs * weights.unsqueeze(-1), dim=1)  # (batch, hidden_size)
        return representations, weights

#######################################
#       Map Encoder (CNN)             #
#######################################
class MapEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # (65 -> 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (32 -> 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (16 -> 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# (8 -> 4)
            nn.ReLU(),
        )
        # Update the fc layers to match output size: 256 * 4 * 4 = 4096.
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
    
    def forward(self, x):
        # x: (batch, 1, 65, 65)
        h = self.conv(x)  # (batch, 256, 4, 4)
        h = h.view(h.size(0), -1)  # Flatten to (batch, 4096)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

#######################################
#       Map Decoder (CNN)             #
#######################################
class MapDecoder(nn.Module):
    def __init__(self, latent_dim=64, cond_dim=512, num_classes=18, map_size=(65,65)):
        super().__init__()
        self.fc = nn.Linear(latent_dim + cond_dim, 256 * 5 * 5)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 5 -> 10
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 10 -> 20
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 20 -> 40
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_classes, kernel_size=4, stride=2, padding=1)  # 40 -> 80
        )
        self.map_size = map_size  # (65,65); you'll crop or resize as needed.
    
    def forward(self, z, c):
        # z: (batch, latent_dim), c: (batch, cond_dim)
        x = torch.cat([z, c], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 256, 5, 5)
        x = self.deconv(x)
        # Crop or resize to 65x65 (here, we crop the top-left corner)
        x = x[:, :, :65, :65]
        return x  # (batch, num_classes, 65, 65)

#######################################
#   Conditional VAE for Map Generation  #
#######################################
class ConditionalMapVAE(nn.Module):
    def __init__(self, latent_dim=64, cond_dim=512, num_classes=18):
        super().__init__()
        self.encoder = MapEncoder(latent_dim=latent_dim)
        self.decoder = MapDecoder(latent_dim=latent_dim, cond_dim=cond_dim, num_classes=num_classes)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, map_input, condition):
        # map_input: (batch, 1, 65, 65), condition: (batch, cond_dim)
        mu, logvar = self.encoder(map_input)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z, condition)
        return logits, mu, logvar

#######################################
#   Condition Extractor from BLIP2      #
#######################################
class ConditionExtractor(nn.Module):
    def __init__(self, blip_input_dim=768, cond_dim=512):
        super().__init__()
        # We use an AverageSelfAttention module to pool over the tokens.
        self.attn = AverageSelfAttention(blip_input_dim)
        self.proj = nn.Linear(blip_input_dim, cond_dim)
    
    def forward(self, blip_embeddings):
        # blip_embeddings: (num_views, num_queries, blip_input_dim)
        # Average over queries per view:
        view_avg = blip_embeddings.mean(dim=1)  # (num_views, blip_input_dim)
        # Apply attention pooling over the views.
        # Add a batch dimension (assume one sample, then remove later)
        pooled, _ = self.attn(view_avg.unsqueeze(0))  # (1, blip_input_dim)
        pooled = pooled.squeeze(0)  # (blip_input_dim,)
        cond = self.proj(pooled)  # (cond_dim,)
        return cond

#######################################
#   Full Conditional Map Generation VAE
#######################################
class MapGenerationVAE(nn.Module):
    def __init__(self, 
                nav_task="gibson", 
                latent_dim=64, 
                cond_dim=512, 
                num_classes=18, 
                blip_input_dim=768,
                load_device="cpu"):
        super().__init__()
        self.device = load_device
        self.nav_task = nav_task
        # VAE for map generation.
        self.vae = ConditionalMapVAE(latent_dim=latent_dim, cond_dim=cond_dim, num_classes=num_classes)
        self.vae.to(self.device)
        # Condition extractor from BLIP2 embeddings.
        self.condition_extractor = ConditionExtractor(blip_input_dim, cond_dim)
        self.condition_extractor.to(self.device)
        # Load BLIP2 model and processors.
        self.blip2_model, self.vis_processors, self.txt_processors = load_blip2_model_lavis("blip2_feature_extractor")
        self.blip2_model.to(self.device)
        self.mem_prompt = generate_mem_prompt(OBJECT_CATEGORIES[nav_task])
    
    def get_blip_embeddings(self, view_rgbs):
        # view_rgbs: list of 4 PIL images.
        processed_images = [self.vis_processors["eval"](img).to(self.device) for img in view_rgbs]
        images_batch = torch.stack(processed_images, dim=0).to(device)
        text_input = self.txt_processors["eval"](self.mem_prompt)
        sample = {"image": images_batch, "text_input": [text_input]*4}
        features = self.blip2_model.extract_features(sample)
        return features.multimodal_embeds  # shape: (4, num_queries, blip_input_dim)
    
    def forward(self, map_input, blip_embeddings=None, view_rgbs=None):
        """
        Args:
            map_input: Tensor of shape (batch, 1, 65, 65) – ground truth map.
            blip_embeddings: Optional tensor of shape (batch, 4, num_queries, blip_input_dim).
            view_rgbs: Optional list (of length batch) where each element is a list of 4 PIL images.
        Returns:
            logits: (batch, num_classes, 65, 65)
            mu, logvar: latent parameters.
        """
        if blip_embeddings is None:
            if view_rgbs is None:
                raise ValueError("Either blip_embeddings or view_rgbs must be provided.")
            batch_conditions = []
            for sample_views in view_rgbs:
                emb = self.get_blip_embeddings(sample_views)  # (4, num_queries, blip_input_dim)
                cond = self.condition_extractor(emb)          # (cond_dim,)
                batch_conditions.append(cond)
            condition = torch.stack(batch_conditions, dim=0)  # (batch, cond_dim)
        else:
            # If precomputed blip_embeddings are provided.
            # Flatten them if necessary. Assume shape (batch, 4, num_queries, blip_input_dim)
            condition = self.condition_extractor(blip_embeddings.view(blip_embeddings.size(0), -1, blip_embeddings.size(-1)))
        
        logits, mu, logvar = self.vae(map_input, condition)
        return logits, mu, logvar

#######################################
#           Example Usage             #
#######################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dummy ground truth map: shape (batch, 1, 65, 65)
    batch_size = 2
    map_input = torch.randint(0, 18, (batch_size, 1, 65, 65)).float().to(device)
    
    # Create dummy view images: each sample has 4 dummy PIL images.
    def dummy_pil_image():
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(arr)
    view_rgbs = [[dummy_pil_image() for _ in range(4)] for _ in range(batch_size)]
    
    model = MapGenerationVAE(
        nav_task="gibson", 
        latent_dim=64, 
        cond_dim=512, 
        num_classes=18, 
        blip_input_dim=768,
        load_device=device)
    
    logits, mu, logvar = model(map_input, view_rgbs=view_rgbs)
    print("Logits shape:", logits.shape)  # Expected: (batch, num_classes, 65, 65)
    print("Mu shape:", mu.shape)            # Expected: (batch, latent_dim)
    print("Logvar shape:", logvar.shape)    # Expected: (batch, latent_dim)
