import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from gym import spaces
from mem_vae_utils import (
    load_blip2_model_lavis, 
    load_instructblip_model_lavis,
    generate_mem_prompt, 
    prepare_blip2_embeddings
)

from constants import OBJECT_CATEGORIES
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder

class VlnResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=128,
        checkpoint="NONE",
        backbone="resnet50",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
    ):
        super().__init__()
        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"depth": observation_space.spaces["depth"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint, map_location=torch.device('cpu'))

            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue

                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v

            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                # Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    def forward(self, observations, no_fc_layer=False):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            obs_depth = observations["depth"]
            if len(obs_depth.size()) == 5:
                observations["depth"] = obs_depth.contiguous().view(
                    -1, obs_depth.size(2), obs_depth.size(3), obs_depth.size(4)
                )
            x = self.visual_encoder(observations)
            x = x.flatten(start_dim=1)

        x = x.detach()
        if no_fc_layer:
            return x

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)

class DepthEncoder(object):
    def __init__(self, 
                device, 
                observation_space=spaces.Dict({"depth": spaces.Box(low=0.0, high=1.0, shape=(224, 224, 1), dtype=np.float32)}), 
                ckpt_file="data/ddppo-models/gibson-2plus-resnet50.pth", 
                backbone="resnet50", 
                batch_size=64) -> None:
        depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=256, # useless
            checkpoint=ckpt_file,
            backbone=backbone,
            trainable=False,
        )
        depth_encoder.eval()
        self.model = depth_encoder.to(device)
        self.device = device
        self.batch_size = batch_size
    
    def extract_fts(self, depth_images):
        '''
        depth_images: Tensor (batch, height, width, 1)
        '''
        fts = []
        for i in range(0, len(depth_images), self.batch_size):
            ft = self.model.visual_encoder(
                {'depth': depth_images[i: i+self.batch_size]}
            ) # (batch, 128, 4, 4)
            fts.append(ft.flatten(start_dim=1).data.cpu().numpy())
        fts = np.concatenate(fts, 0)
        return fts

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
        use_cond_input=False,
        dropout_p=0.2  # Dropout probability
    ):
        super(TopdownMapHybridEncoder, self).__init__()
        self.use_cond_input = use_cond_input

        # Embedding layer: convert each pixel (class id) to a vector.
        self.embedding = nn.Embedding(num_classes, emb_dim)

        # Convolutional backbone to extract local features.
        # Input: (B, 65, 65, emb_dim) → permute → (B, emb_dim, 65, 65)
        self.conv1 = nn.Conv2d(emb_dim, 32, kernel_size=5, stride=2, padding=2)   # → (B, 32, 33, 33)
        self.bn1   = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(p=dropout_p) 
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)          # → (B, 64, 17, 17)
        self.bn2   = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(p=dropout_p)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)         # → (B, 128, 9, 9)
        self.bn3   = nn.BatchNorm2d(128)
        
        # Project conv features to transformer dimension if needed.
        # Final conv feature map shape: (B, 128, 9, 9)
        # We want each spatial location to be a token of dimension transformer_dim.
        self.conv_proj = nn.Conv2d(128, transformer_dim, kernel_size=1)  # (B, transformer_dim, 9, 9)
        
        # Transformer encoder to capture global context.
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=n_heads, dropout=dropout_p)
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
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)))    # → (B, 64, 17, 17)
        x = self.dropout2(x)
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
        use_cond_input=True,
        dropout_p=0.2
    ): 

        super(TopdownMapVAEDecoder, self).__init__()
        self.use_cond_input = use_cond_input
        
        # For late conditioning, we concatenate the latent vector with the mem_condition.
        fc_input_dim = latent_dim + cond_dim if use_cond_input else latent_dim
        self.fc = nn.Linear(fc_input_dim, 256 * 5 * 5)
        self.dropout_fc = nn.Dropout(p=dropout_p) 
        
        # Transposed convolution layers to gradually upsample the feature map:
        # - From 5x5 → 9x9,
        # - From 9x9 → 17x17,
        # - From 17x17 → 33x33,
        # - From 33x33 → 65x65.
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.bn1 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout2d(p=dropout_p)
        
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2= nn.Dropout2d(p=dropout_p)
        
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout3 = nn.Dropout2d(p=dropout_p)
        
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
        x = self.dropout_fc(x)
        x = x.reshape(batch_size, 256, 5, 5)  # Reshape to (B, 256, 5, 5)
        
        # Upsample gradually:
        x = F.leaky_relu(self.bn1(self.deconv1(x)))  # → (B, 128, 9, 9)
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.deconv2(x)))    # → (B, 64, 17, 17)
        x = self.dropout2(x)
        x = F.leaky_relu(self.bn3(self.deconv3(x)))    # → (B, 32, 33, 33)
        x = self.dropout3(x)
        
        logits = self.deconv4(x)                       # → (B, num_classes, 65, 65)
        return logits # (B, num_classes, 65, 65)

class TopdownMapVAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim=128, oh_aux_task=False, oh_aux_class=17):
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
        self.oh_aux_task = oh_aux_task
        if oh_aux_task:
            self.aux_classifier = nn.Linear(latent_dim, oh_aux_class)

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
        
        if self.oh_aux_task:
            aux_out = self.aux_classifier(mu)
            return logits, mu, logvar, aux_out

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
        use_cond_input=model_config["encoder_use_cond_input"],
        dropout_p=model_config["dropout_p"]
    )
    
    # Create the decoder.
    decoder = TopdownMapVAEDecoder(
        latent_dim=model_config["latent_dim"],
        cond_dim=model_config["cond_dim"],
        num_classes=model_config["num_classes"],
        use_cond_input=model_config["decoder_use_cond_input"],
        dropout_p=model_config["dropout_p"]
    )
    
    # Combine encoder and decoder into the VAE.
    vae = TopdownMapVAE(
        encoder, 
        decoder, 
        latent_dim=model_config["latent_dim"], 
        oh_aux_task=model_config["oh_aux_task"],
        oh_aux_class=model_config["num_classes"]-1
    )
    
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
        "mem_generator_use_attention": True,
        "oh_aux_task": True
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
    rgb_embeds_batch = prepare_blip2_embeddings(blip2_model, vis_processors, txt_processors, rgb_views_batch, mem_prompt)
    mem_condition_batch, _ = mem_generator(rgb_embeds_batch)  # (B, 768)

    # Forward pass.
    logits, mu, logvar = vae(local_map_batch, mem_condition_batch)
    print("Output logits shape:", logits.shape)   # Expected: (B, num_classes, 65, 65)
    print("Latent mean shape:", mu.shape)           # Expected: (B, latent_dim)
    print("Latent logvar shape:", logvar.shape)       # Expected: (B, latent_dim)

    print("============testing=================")


