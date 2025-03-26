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
        self.attention_weights = nn.Parameter(torch.randn(hidden_size))
        self.softmax = nn.Softmax(dim=1)
        self.non_linearity = F.gelu

    def forward(self, inputs, attention_mask=None):
        scores = self.non_linearity(torch.matmul(inputs, self.attention_weights))
        if attention_mask is not None:
            scores = scores + attention_mask
        weights = self.softmax(scores)
        representations = torch.sum(inputs * weights.unsqueeze(-1), dim=1)
        return representations, weights

class MemGenerator(nn.Module):
    def __init__(self, token_dim=768, aggregated_dim=768, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attention_pool = AverageSelfAttention(hidden_size=token_dim)
        self.fc = nn.Linear(token_dim, aggregated_dim) if aggregated_dim != token_dim else None

    def forward(self, x):
        x = x.unsqueeze(0) if x.dim() == 3 else x
        
        B, num_views, num_tokens, token_dim = x.size()
        x_flat = x.reshape(B, num_views * num_tokens, token_dim)
        
        if self.use_attention:
            condition, weights = self.attention_pool(x_flat)
        else:
            condition, weights = x_flat.mean(dim=1), None
        
        if self.fc is not None:
            condition = self.fc(condition)
        
        return condition, weights

class VectorQuantizer(nn.Module):
    """
    Vector Quantization module for VQ-VAE
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        """
        Args:
            num_embeddings: Number of embeddings in the codebook
            embedding_dim: Dimension of each embedding vector
            commitment_cost: Weight for commitment loss
        """
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Create the embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape (batch_size, embedding_dim)
        Returns:
            quantized: Quantized vectors
            vq_loss: Vector quantization loss
            perplexity: Perplexity of the codebook usage
            encodings: One-hot encodings of the quantized vectors
            encoding_indices: Indices of the quantized vectors
        """
        # Flatten input if needed
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances between inputs and embeddings
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Find closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Calculate loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # Commitment loss
        q_latent_loss = F.mse_loss(quantized, inputs.detach())  # Codebook loss
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Calculate perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encodings, encoding_indices

class TopdownMapEncoder(nn.Module):
    """
    Encoder for VQ-VAE (modified from TopdownMapHybridEncoder)
    """
    def __init__(
        self,
        embedding_dim=128,    # Embedding dimension (latent space dimension)
        cond_dim=768,         # Condition vector dimension
        num_classes=18,       # Number of classes in the topdown map
        emb_dim=32,           # Embedding dimension for each class id
        transformer_dim=128,  # Dimension for transformer encoder features
        n_transformer_layers=2,
        n_heads=4,
        use_cond_input=False,
        dropout_p=0.2         # Dropout probability
    ):
        super(TopdownMapEncoder, self).__init__()
        self.use_cond_input = use_cond_input

        # Embedding layer: convert each pixel (class id) to a vector
        self.embedding = nn.Embedding(num_classes, emb_dim)

        # Convolutional backbone
        self.conv1 = nn.Conv2d(emb_dim, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(p=dropout_p) 
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(p=dropout_p)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Project conv features to transformer dimension
        self.conv_proj = nn.Conv2d(128, transformer_dim, kernel_size=1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=n_heads, dropout=dropout_p)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        
        # Linear layer for final output
        linear_input_dim = transformer_dim + cond_dim if use_cond_input else transformer_dim
        self.fc = nn.Linear(linear_input_dim, embedding_dim)

    def forward(self, local_map, mem_condition):
        # Ensure batch dimension
        if local_map.dim() == 2:
            local_map = local_map.unsqueeze(0)
        
        # Embed local map
        x = self.embedding(local_map)
        x = x.permute(0, 3, 1, 2)
        
        # Convolutional backbone
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        
        # Project conv features
        x = self.conv_proj(x)
        
        # Reshape for transformer
        B, C, H, W = x.size()
        x = x.reshape(B, C, H * W)
        x = x.permute(2, 0, 1)
        
        # Apply transformer
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Mean-pool over sequence
        
        # Late conditioning
        if self.use_cond_input:
            x = torch.cat([x, mem_condition], dim=1)
        
        # Final embedding
        z = self.fc(x)
        return z

class TopdownMapDecoder(nn.Module):
    """
    Decoder for VQ-VAE (modified from TopdownMapVAEDecoder)
    """
    def __init__(
        self, 
        embedding_dim=128,  # Embedding dimension (latent space)
        cond_dim=768,       # Condition vector dimension
        num_classes=18,     # Number of classes in the topdown map
        use_cond_input=True,
        dropout_p=0.2
    ): 
        super(TopdownMapDecoder, self).__init__()
        self.use_cond_input = use_cond_input
        
        # Late conditioning
        fc_input_dim = embedding_dim + cond_dim if use_cond_input else embedding_dim
        self.fc = nn.Linear(fc_input_dim, 256 * 5 * 5)
        self.dropout_fc = nn.Dropout(p=dropout_p) 
        
        # Transposed convolution layers
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.bn1 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout2d(p=dropout_p)
        
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(p=dropout_p)
        
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout3 = nn.Dropout2d(p=dropout_p)
        
        self.deconv4 = nn.ConvTranspose2d(32, num_classes, kernel_size=3, stride=2, padding=1, output_padding=0)

    def forward(self, z, mem_condition):
        # Late conditioning
        if self.use_cond_input:
            z = torch.cat([z, mem_condition], dim=1)
        batch_size = z.size(0)
        
        # Project to feature map
        x = self.fc(z)
        x = F.leaky_relu(x)
        x = self.dropout_fc(x)
        x = x.reshape(batch_size, 256, 5, 5)
        
        # Upsample
        x = F.leaky_relu(self.bn1(self.deconv1(x)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.deconv2(x)))
        x = self.dropout2(x)
        x = F.leaky_relu(self.bn3(self.deconv3(x)))
        x = self.dropout3(x)
        
        logits = self.deconv4(x)
        return logits

class TopdownMapVQVAE(nn.Module):
    """
    Vector-Quantized VAE for topdown map generation
    """
    def __init__(
        self, 
        encoder, 
        decoder, 
        num_embeddings, 
        embedding_dim, 
        commitment_cost,
        oh_aux_task=False, 
        oh_aux_class=17
    ):
        super(TopdownMapVQVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.oh_aux_task = oh_aux_task
        if self.oh_aux_task:
            self.aux_classifier = nn.Linear(embedding_dim, oh_aux_class)

    def forward(self, local_map, mem_condition):
        """
        Args:
            local_map: Topdown map of shape (B, 65, 65)
            mem_condition: Condition vector from MemGenerator (B, cond_dim)
        Returns:
            logits: Output logits (B, num_classes, 65, 65)
            vq_loss: Vector quantization loss
            perplexity: Perplexity of the codebook usage
            encoding_indices: Indices of the quantized vectors
        """
        # Encode
        z = self.encoder(local_map, mem_condition)
        
        # Quantize
        quantized, vq_loss, perplexity, _, encoding_indices = self.quantizer(z)
        
        # Decode
        logits = self.decoder(quantized, mem_condition)
        
        if self.oh_aux_task:
            aux_out = self.aux_classifier(z)
            return logits, vq_loss, perplexity, encoding_indices, aux_out
        
        return logits, vq_loss, perplexity, encoding_indices

def create_MemMapVQVAE(model_config):
    """
    Creates the MemGenerator and VQ-VAE based on a configuration dictionary.
    
    Args:
        model_config (dict): Configuration dictionary with model parameters
    
    Returns:
        tuple: (mem_generator, vqvae)
    """
    # Create MemGenerator
    mem_generator = MemGenerator(
        token_dim=model_config["mem_generator_token_dim"],
        aggregated_dim=model_config["mem_generator_aggregated_dim"],
        use_attention=model_config["mem_generator_use_attention"]
    )
    
    # Create Encoder
    encoder = TopdownMapEncoder(
        embedding_dim=model_config["embedding_dim"],
        cond_dim=model_config["cond_dim"],
        num_classes=model_config["num_classes"],
        emb_dim=model_config["emb_dim"],
        transformer_dim=model_config["transformer_dim"],
        n_transformer_layers=model_config["n_transformer_layers"],
        n_heads=model_config["n_heads"],
        use_cond_input=model_config["encoder_use_cond_input"],
        dropout_p=model_config["dropout_p"]
    )
    
    # Create Decoder
    decoder = TopdownMapDecoder(
        embedding_dim=model_config["embedding_dim"],
        cond_dim=model_config["cond_dim"],
        num_classes=model_config["num_classes"],
        use_cond_input=model_config["decoder_use_cond_input"],
        dropout_p=model_config["dropout_p"]
    )
    
    # Create VQ-VAE
    vqvae = TopdownMapVQVAE(
        encoder,
        decoder,
        num_embeddings=model_config["num_embeddings"],
        embedding_dim=model_config["embedding_dim"],
        commitment_cost=model_config["commitment_cost"],
        oh_aux_task=model_config["oh_aux_task"],
        oh_aux_class=model_config["num_classes"]-1
    )
    
    return mem_generator, vqvae

if __name__ == "__main__":
    # Model configuration
    config = {
        "embedding_dim": 128,        # Dimension of embedding vectors
        "num_embeddings": 512,       # Size of the codebook
        "commitment_cost": 0.25,     # Weight for commitment loss
        "cond_dim": 768,
        "num_classes": 18,
        "emb_dim": 32,
        "transformer_dim": 128,
        "n_transformer_layers": 2,
        "n_heads": 4,
        "encoder_use_cond_input": True,
        "decoder_use_cond_input": True,
        "mem_generator_token_dim": 768,
        "mem_generator_aggregated_dim": 768,
        "mem_generator_use_attention": True,
        "oh_aux_task": True,
        "dropout_p": 0.2
    }
    
    # Create models
    mem_generator, vqvae = create_MemMapVQVAE(config)

    # Testing with random data
    mem_prompt = generate_mem_prompt(OBJECT_CATEGORIES["gibson"])
    blip2_model_name = "blip2_t5_instruct"
    blip2_model, vis_processors, txt_processors = load_instructblip_model_lavis(blip2_model_name)

    batch_size = 2
    local_map_batch = torch.randint(0, config["num_classes"], (batch_size, 65, 65))
    rgb_views_batch = torch.randn(batch_size, 4, 3, 1024, 1024)
    rgb_embeds_batch = prepare_blip2_embeddings(blip2_model, vis_processors, txt_processors, rgb_views_batch, mem_prompt)
    mem_condition_batch, _ = mem_generator(rgb_embeds_batch)

    # Forward pass
    logits, vq_loss, perplexity, encoding_indices = vqvae(local_map_batch, mem_condition_batch)
    print("Output logits shape:", logits.shape)
    print("VQ loss:", vq_loss.item())
    print("Codebook perplexity:", perplexity.item())
    print("Encoding indices shape:", encoding_indices.shape)

    print("============testing=================")
