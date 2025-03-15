import torch
import torch.nn as nn
import torch.nn.functional as F

from mem_vae_utils import (
    load_blip2_model_lavis, 
    load_instructblip_model_lavis,
    generate_mem_prompt, 
    prepare_blip2_embeddings
)

from constants import OBJECT_CATEGORIES

class AverageSelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Learnable attention weights for each token dimension.
        self.attention_weights = nn.Parameter(torch.randn(hidden_size))
        self.softmax = nn.Softmax(dim=1)
        # Use GELU as the non-linearity.
        self.non_linearity = F.gelu

    def forward(
        self, 
        inputs, # (batch, seq_len, hidden_size)
        attention_mask=None # (optional): Mask to modify attention scores
    ):
        # Compute attention scores: (batch, seq_len)
        scores = self.non_linearity(torch.matmul(inputs, self.attention_weights))
        if attention_mask is not None:
            scores = scores + attention_mask
        # Normalize the scores to get weights.
        weights = self.softmax(scores)  # The attention weights (batch, seq_len).
        representations = torch.sum(inputs * weights.unsqueeze(-1), dim=1) # Aggregated vector of shape (batch, hidden_size).
        return representations, weights

class MemGenerator(nn.Module):
    def __init__(
        self, 
        token_dim=768, # Dimension of each token (default: 768).
        aggregated_dim=768, # Desired output condition dimension.
        use_attention=True # Whether to use Average Self-Attention pooling.
    ):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attention_pool = AverageSelfAttention(hidden_size=token_dim)
        # If aggregated_dim differs from token_dim, add a linear projection.
        self.fc = nn.Linear(token_dim, aggregated_dim) if aggregated_dim != token_dim else None

    def forward(self, x):
        x = x.unsqueeze(0) if x.dim() == 3 else x # Ensure a batch dimension: if input is (4,32,768), add batch dimension.
        
        B, num_views, num_tokens, token_dim = x.size()
        x_flat = x.reshape(B, num_views * num_tokens, token_dim) # (B, 4*32, 768) => (B, 128, 768)
        
        if self.use_attention:
            condition, weights = self.attention_pool(x_flat)
        else:
            condition, weights = x_flat.mean(dim=1), None # simple mean pooling.
        
        if self.fc is not None:
            condition = self.fc(condition) # (B, aggregated_dim)
        
        return condition, weights

class TopdownMapHybridEncoder(nn.Module):
    def __init__(
        self,
        latent_dim=128,      # Latent space dimension.
        cond_dim=768,        # Condition vector dimension from MemGenerator.
        num_classes=18,      # Number of classes in the topdown map.
        emb_dim=32,          # Embedding dimension for each class id.
        transformer_dim=128, # Dimension for transformer encoder features.
        n_transformer_layers=2,
        n_heads=4,
        use_cond_input=False
    ):
        super(TopdownMapHybridEncoder, self).__init__()
        self.use_cond_input = use_cond_input

        # Embedding layer: convert each pixel (class id) to a vector.
        self.embedding = nn.Embedding(num_classes, emb_dim)

        # Convolutional backbone to extract local features.
        # Input: (B, 65, 65, emb_dim) → permute → (B, emb_dim, 65, 65)
        self.conv1 = nn.Conv2d(emb_dim, 32, kernel_size=5, stride=2, padding=2)   # → (B, 32, 33, 33)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)          # → (B, 64, 17, 17)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)         # → (B, 128, 9, 9)
        self.bn3   = nn.BatchNorm2d(128)
        
        # Project conv features to transformer dimension if needed.
        # Final conv feature map shape: (B, 128, 9, 9)
        # We want each spatial location to be a token of dimension transformer_dim.
        self.conv_proj = nn.Conv2d(128, transformer_dim, kernel_size=1)  # (B, transformer_dim, 9, 9)
        
        # Transformer encoder to capture global context.
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        
        # After transformer, we mean-pool over the sequence.
        # The output dimension will be transformer_dim.
        # Late conditioning: we then concatenate mem_condition.
        linear_input_dim = transformer_dim + cond_dim if use_cond_input else transformer_dim
        
        # Fully connected layer to output latent mean and log-variance.
        self.fc = nn.Linear(linear_input_dim, latent_dim * 2)

    def forward(self, 
        local_map, # Topdown map, shape (B, 65, 65) (each pixel is a class id).
        mem_condition # Condition vector from MemGenerator, shape (B, cond_dim).
    ):
        # Ensure a batch dimension.
        if local_map.dim() == 2:
            local_map = local_map.unsqueeze(0)
        batch_size = local_map.size(0)
        
        # Embed local map: (B, 65, 65) → (B, 65, 65, emb_dim)
        x = self.embedding(local_map)
        # Permute to (B, emb_dim, 65, 65)
        x = x.permute(0, 3, 1, 2)
        
        # Convolutional backbone.
        x = F.leaky_relu(self.bn1(self.conv1(x)))  # → (B, 32, 33, 33)
        x = F.leaky_relu(self.bn2(self.conv2(x)))    # → (B, 64, 17, 17)
        x = F.leaky_relu(self.bn3(self.conv3(x)))    # → (B, 128, 9, 9)
        
        # Project conv features to transformer dimension.
        x = self.conv_proj(x)  # → (B, transformer_dim, 9, 9)
        
        # Reshape feature map into a sequence.
        B, C, H, W = x.size()  # Here, C == transformer_dim and H*W = 81 tokens.
        x = x.reshape(B, C, H * W)      # (B, transformer_dim, sequence_length)
        x = x.permute(2, 0, 1)          # (sequence_length, B, transformer_dim) as expected by the transformer
        
        # Apply transformer encoder.
        x = self.transformer_encoder(x)  # (sequence_length, B, transformer_dim)
        # Mean-pool over the sequence dimension.
        x = x.mean(dim=0)  # (B, transformer_dim)
        
        # Late conditioning: concatenate mem_condition.
        if self.use_cond_input:
            x = torch.cat([x, mem_condition], dim=1)  # (B, transformer_dim + cond_dim)
        
        # Produce latent parameters.
        stats = self.fc(x)
        mu, logvar = stats.chunk(2, dim=1) # (B, latent_dim) each
        return mu, logvar

class TopdownMapVAEDecoder(nn.Module):
    def __init__(
        self, 
        latent_dim=128, # Dimensionality of the latent vector
        cond_dim=768, # Dimensionality of the condition vector (from MemGenerator).
        num_classes=18, # Number of classes in the topdown map.
        use_cond_input=True
    ): 

        super(TopdownMapVAEDecoder, self).__init__()
        self.use_cond_input = use_cond_input
        
        # For late conditioning, we concatenate the latent vector with the mem_condition.
        fc_input_dim = latent_dim + cond_dim if use_cond_input else latent_dim
        
        # Fully connected layer projects the input to a flattened feature map of shape (256, 5, 5).
        self.fc = nn.Linear(fc_input_dim, 256 * 5 * 5)
        
        # Transposed convolution layers to gradually upsample the feature map:
        # - From 5x5 → 9x9,
        # - From 9x9 → 17x17,
        # - From 17x17 → 33x33,
        # - From 33x33 → 65x65.
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, num_classes, kernel_size=3, stride=2, padding=1, output_padding=0)
        # The final layer outputs logits (no activation applied).

    def forward(self, 
        z, # Latent vector of shape (B, latent_dim).
        mem_condition #  Condition vector from MemGenerator, shape (B, cond_dim).
    ):
        # Late conditioning: concatenate latent vector and condition.
        if self.use_cond_input:
            z = torch.cat([z, mem_condition], dim=1)
        batch_size = z.size(0)
        
        # Project to a flattened feature map.
        x = self.fc(z)  # (B, 256*5*5)
        x = F.leaky_relu(x)
        x = x.reshape(batch_size, 256, 5, 5)  # Reshape to (B, 256, 5, 5)
        
        # Upsample gradually:
        x = F.leaky_relu(self.bn1(self.deconv1(x)))  # → (B, 128, 9, 9)
        x = F.leaky_relu(self.bn2(self.deconv2(x)))    # → (B, 64, 17, 17)
        x = F.leaky_relu(self.bn3(self.deconv3(x)))    # → (B, 32, 33, 33)
        logits = self.deconv4(x)                       # → (B, num_classes, 65, 65)
        return logits # (B, num_classes, 65, 65)

class TopdownMapVAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim=128):
        """
        Combined conditional VAE for topdown map generation.

        Args:
            encoder (nn.Module): Encoder module.
            decoder (nn.Module): Decoder module.
            latent_dim (int): Dimensionality of the latent space.
        """
        super(TopdownMapVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        """Applies the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, local_map, mem_condition):
        """
        Args:
            local_map (Tensor): Topdown map of shape (B, 65, 65) (each pixel is a class id).
            mem_condition (Tensor): Condition vector from MemGenerator, shape (B, cond_dim).
        Returns:
            logits (Tensor): Output logits with shape (B, num_classes, 65, 65).
            mu (Tensor): Latent mean of shape (B, latent_dim).
            logvar (Tensor): Latent log-variance of shape (B, latent_dim).
        """
        # Encode the inputs to get latent parameters.
        mu, logvar = self.encoder(local_map, mem_condition)
        # Sample latent vector.
        z = self.reparameterize(mu, logvar)
        # Decode to get output logits.
        logits = self.decoder(z, mem_condition)
        return logits, mu, logvar

def create_MemMapVAE(model_config):
    """
    Creates the MemGenerator and VAE (encoder + decoder) based on a configuration dictionary.
    
    Args:
        config (dict): Configuration dictionary with keys:
            - latent_dim: int, latent space dimension.
            - cond_dim: int, condition vector dimension.
            - num_classes: int, number of classes in the topdown map.
            - emb_dim: int, embedding dimension for each class id.
            - transformer_dim: int, transformer encoder feature dimension.
            - n_transformer_layers: int, number of transformer encoder layers.
            - n_heads: int, number of attention heads in transformer.
            - encoder_use_cond_input: bool, whether to inject condition early in the encoder.
            - decoder_use_cond_input: bool, whether to inject condition early in the decoder.
            - mem_generator_token_dim: int, token dimension for MemGenerator.
            - mem_generator_aggregated_dim: int, aggregated dimension for MemGenerator.
            - mem_generator_use_attention: bool, whether to use self-attention in MemGenerator.
    
    Returns:
        tuple: (mem_generator, vae)
            - mem_generator (nn.Module): The MemGenerator model.
            - vae (nn.Module): The combined VAE model.
    """
    
    # Create the MemGenerator.
    mem_generator = MemGenerator(
        token_dim=model_config["mem_generator_token_dim"],
        aggregated_dim=model_config["mem_generator_aggregated_dim"],
        use_attention=model_config["mem_generator_use_attention"]
    )
    
    # Create the encoder.
    encoder = TopdownMapHybridEncoder(
        latent_dim=model_config["latent_dim"],
        cond_dim=model_config["cond_dim"],
        num_classes=model_config["num_classes"],
        emb_dim=model_config["emb_dim"],
        transformer_dim=model_config["transformer_dim"],
        n_transformer_layers=model_config["n_transformer_layers"],
        n_heads=model_config["n_heads"],
        use_cond_input=model_config["encoder_use_cond_input"]
    )
    
    # Create the decoder.
    decoder = TopdownMapVAEDecoder(
        latent_dim=model_config["latent_dim"],
        cond_dim=model_config["cond_dim"],
        num_classes=model_config["num_classes"],
        use_cond_input=model_config["decoder_use_cond_input"]
    )
    
    # Combine encoder and decoder into the VAE.
    vae = TopdownMapVAE(encoder, decoder, latent_dim=model_config["latent_dim"])
    
    return mem_generator, vae
    
if __name__ == "__main__":
    config = {
        "latent_dim": 128,
        "cond_dim": 768,
        "num_classes": 18,
        "emb_dim": 32,
        "transformer_dim": 128,
        "n_transformer_layers": 2,
        "n_heads": 4,
        "encoder_use_cond_input": True,   # Use late conditioning in encoder.
        "decoder_use_cond_input": True,   # Use late conditioning in decoder.
        "mem_generator_token_dim": 768,
        "mem_generator_aggregated_dim": 768,
        "mem_generator_use_attention": True
    }
    
    # Create models.
    mem_generator, vae = create_MemMapVAE(config)

    # no blip2 embedding situation, generate from rgb_views
    mem_prompt = generate_mem_prompt(OBJECT_CATEGORIES["gibson"])
    blip2_model_name="blip2_t5_instruct"
    blip2_model, vis_processors, txt_processors = load_instructblip_model_lavis(blip2_model_name)

    batch_size = 2
    local_map_batch = torch.randint(0, config["num_classes"], (batch_size, 65, 65))  # (B, 65, 65)
    rgb_views_batch = torch.randn(batch_size, 4, 3, 1024, 1024)
    blip2_embeds_batch = prepare_blip2_embeddings(blip2_model, vis_processors, txt_processors, rgb_views_batch, mem_prompt)
    mem_condition_batch, _ = mem_generator(blip2_embeds_batch)  # (B, 768)

    # Forward pass.
    logits, mu, logvar = vae(local_map_batch, mem_condition_batch)
    print("Output logits shape:", logits.shape)   # Expected: (B, num_classes, 65, 65)
    print("Latent mean shape:", mu.shape)           # Expected: (B, latent_dim)
    print("Latent logvar shape:", logvar.shape)       # Expected: (B, latent_dim)

    print("============testing=================")


