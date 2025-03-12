import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import gelu
from lavis.models import load_model_and_preprocess
from constants import OBJECT_CATEGORIES

def load_blip2_model_lavis(
    model_name="blip2_feature_extractor", 
    type="pretrain_vitL"
):
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=model_name, 
        model_type=type, 
        is_eval=True, device=device)

    return model, vis_processors, txt_processors

#######################################
#       AverageSelfAttention          #
#######################################
class AverageSelfAttention(nn.Module):
    def __init__(self, hidden_size):
        """
        A self-attention pooling module that computes a weighted average of
        the input sequence along the sequence dimension.
        Args:
            hidden_size: Size of the hidden state (model_dim)
        """
        super().__init__()
        # Learnable weight vector for computing attention scores.
        self.attention_weights = nn.Parameter(torch.randn(hidden_size))
        self.softmax = nn.Softmax(dim=1)
        self.non_linearity = gelu

    def forward(self, inputs, attention_mask=None):
        """
        Args:
            inputs: Tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional mask of shape (batch, seq_len)
        Returns:
            representations: Tensor of shape (batch, hidden_size), the weighted sum.
            weights: The attention weights (after softmax) of shape (batch, seq_len).
        """
        # Compute raw scores via dot product between each token and the learned weight.
        scores = self.non_linearity(torch.matmul(inputs, self.attention_weights))  # (batch, seq_len)
        if attention_mask is not None:
            scores = scores + attention_mask
        weights = self.softmax(scores)  # (batch, seq_len)
        representations = torch.sum(inputs * weights.unsqueeze(-1), dim=1)  # (batch, hidden_size)
        return representations, weights

#######################################
#       Transformer Encoder           #
#######################################
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers, num_heads, dropout=0.1):
        """
        Args:
            input_dim: Dimensionality of each token in the input sequence.
            model_dim: Hidden dimension of the transformer.
            num_layers: Number of transformer encoder layers.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)                  # (batch, seq_len, model_dim)
        x = x.transpose(0, 1)                   # (seq_len, batch, model_dim)
        enc_out = self.encoder(x)               # (seq_len, batch, model_dim)
        enc_out = enc_out.transpose(0, 1)         # (batch, seq_len, model_dim)
        return enc_out

#######################################
#       Transformer Decoder           #
#######################################
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, model_dim, num_layers, num_heads, dropout=0.1, max_seq_len=4225):
        """
        Args:
            vocab_size: Vocabulary size for output tokens (e.g., number of classes).
            model_dim: Hidden dimension of the transformer.
            num_layers: Number of decoder layers.
            num_heads: Number of attention heads.
            max_seq_len: Maximum output sequence length. Here, 65*65=4225.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, model_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # tgt: (batch, tgt_seq_len) token indices.
        batch_size, tgt_seq_len = tgt.size()
        positions = torch.arange(0, tgt_seq_len, device=tgt.device).unsqueeze(0).expand(batch_size, tgt_seq_len)
        tgt_emb = self.embedding(tgt) + self.pos_embedding(positions)  # (batch, tgt_seq_len, model_dim)
        tgt_emb = tgt_emb.transpose(0, 1)  # (tgt_seq_len, batch, model_dim)
        memory = memory.transpose(0, 1)    # (mem_seq_len, batch, model_dim)
        dec_out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        dec_out = dec_out.transpose(0, 1)  # (batch, tgt_seq_len, model_dim)
        logits = self.fc_out(dec_out)      # (batch, tgt_seq_len, vocab_size)
        return logits

#######################################
#  Transformer CVAE for Map Generation
#  Conditioned on BLIP2 Embeddings from Surrounding Views
#######################################
class MapGenerationCVAE(nn.Module):
    def __init__(self, 
                nav_task="gibson",
                blip_input_dim=768, 
                vocab_size=18,
                model_dim=512, 
                latent_dim=64,
                enc_layers=4, 
                dec_layers=4,
                num_heads=8, 
                dropout=0.1,
                max_seq_len=4225, 
                use_attn_pooling=True,
                blip2_name="blip2_feature_extractor"):
        """
        Args:
            blip_input_dim: Dimensionality of each token from BLIP2 (e.g., 768).
                           The BLIP2 embeddings from 4 views are flattened to shape (batch, 128, 768).
            vocab_size: Number of classes (output vocabulary size). E.g., 18 if 18 valid classes.
            model_dim: Transformer hidden dimension.
            latent_dim: Dimension of the latent variable z.
            max_seq_len: Output sequence length; here, 65*65 = 4225.
            use_attn_pooling: Use attention pooling for encoder output.
        """
        super().__init__()
        self.nav_task = nav_task
        self.generate_mem_prompt(OBJECT_CATEGORIES[nav_task])
        
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.use_attn_pooling = use_attn_pooling

        # Encoder: processes the flattened BLIP2 embeddings.
        self.encoder = TransformerEncoder(blip_input_dim, model_dim, num_layers=enc_layers, num_heads=num_heads, dropout=dropout)
        
        # Pooling module.
        if use_attn_pooling:
            self.pooling = AverageSelfAttention(model_dim)
        else:
            self.pooling = None
        
        # Posterior projection.
        self.fc_mu = nn.Linear(model_dim, latent_dim)
        self.fc_logvar = nn.Linear(model_dim, latent_dim)
        
        # Map latent variable to memory token for decoder.
        self.latent_to_memory = nn.Linear(latent_dim, model_dim)
        
        # Decoder: generate token sequence representing the map.
        self.decoder = TransformerDecoder(vocab_size, model_dim, num_layers=dec_layers, num_heads=num_heads, dropout=dropout, max_seq_len=max_seq_len)

        # For data with only RGB, need BLIP2 to get embeddings.
        self.blip2_model, self.vis_processors, self.txt_processors = load_blip2_model_lavis(blip2_name)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate_mem_prompt(self, objects_list):
        objects_str = ", ".join(objects_list)       
        self.mem_prompt = (
            f"Describe the environment in detail, focusing on navigable spaces, obstacles, "
            f"and key objects such as {objects_str}. "
            "Mention the layout, spatial relationships, and form a topdown map to benefit navigation."
        )

    def get_blip_embeddings(self, view_rgbs):
        """
        Extract BLIP2 embeddings from a list of raw RGB images (PIL images).
        view_rgbs: list of PIL images (for one sample).
        Returns:
            Tensor of shape (num_views * num_queries, blip_input_dim)
        """
        # Process each view using the visual processor.
        processed_images = [self.vis_processors["eval"](img) for img in view_rgbs]
        # Stack into a batch: shape (num_views, C, H, W)
        images_batch = torch.stack(processed_images, dim=0)
        text_input = self.txt_processors["eval"](self.mem_prompt)
        # For BLIP2, text_input is expected as a list. We assume same prompt for all views.
        sample = {"image": images_batch, "text_input": [text_input]*4}
        # Extract features.
        features = self.blip2_model.extract_features(sample)
        # Return the multimodal embeddings.
        return features.multimodal_embeds  # e.g., shape (num_views, num_queries, blip_input_dim)

    def forward(self, tgt_input, blip_embeddings=None, view_rgbs=None, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt_input: Tensor of shape (batch, tgt_seq_len) with token indices for the output map.
            blip_embeddings: Optional tensor of shape (batch, 128, blip_input_dim). If provided, used directly.
            view_rgbs: Optional list of lists. Each element is a list of 4 raw RGB PIL images for one sample.
                       Used to extract BLIP2 embeddings if blip_embeddings is None.
        Returns:
            logits: (batch, tgt_seq_len, vocab_size) – logits over classes for each pixel.
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        # Auto-detect whether to extract BLIP embeddings.
        if blip_embeddings is None:
            if view_rgbs is None:
                raise ValueError("Either blip_embeddings or view_rgbs must be provided.")
            # Process each sample in the batch.
            batch_embeds = []
            for sample_views in view_rgbs:
                # sample_views: list of 4 PIL images for that sample.
                emb = self.get_blip_embeddings(sample_views)  # shape (num_views, num_queries, blip_input_dim)
                # Flatten the views along the first two dimensions.
                emb = emb.reshape(-1, emb.size(-1))  # Use reshape instead of view
                batch_embeds.append(emb)
            # Stack along the batch dimension.
            blip_embeddings = torch.stack(batch_embeds, dim=0)  # shape (batch, total_tokens, blip_input_dim)
        
        # Now, blip_embeddings has shape (batch, num_tokens, blip_input_dim)
        enc_out = self.encoder(blip_embeddings)  # (batch, num_tokens, model_dim)
        
        if self.use_attn_pooling:
            pooled, _ = self.pooling(enc_out)      # (batch, model_dim)
        else:
            pooled = enc_out.mean(dim=1)
        
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        z = self.reparameterize(mu, logvar)
        
        memory_token = self.latent_to_memory(z).unsqueeze(1)  # (batch, 1, model_dim)
        logits = self.decoder(tgt_input, memory_token, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return logits, mu, logvar
#######################################
#           Example Usage             #
#######################################
if __name__ == "__main__":
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration.
    blip_input_dim = 768    # Each BLIP token dimension.
    vocab_size = 18         # Suppose we have 18 classes (including 0).
    model_dim = 512
    latent_dim = 64
    enc_layers = 4
    dec_layers = 4
    num_heads = 8
    dropout = 0.1
    max_seq_len = 65 * 65   # 4225 tokens for a 65x65 map.
    use_attn_pooling = True

    # Instantiate the model.
    model = MapGenerationCVAE(
        nav_task="gibson",
        blip_input_dim=blip_input_dim,
        vocab_size=vocab_size,
        model_dim=model_dim,
        latent_dim=latent_dim,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_seq_len=max_seq_len,
        use_attn_pooling=use_attn_pooling,
        blip2_name="blip2_feature_extractor"
    ).to(device)
    
    # Dummy target input: a sequence of tokens of length 4225.
    batch_size = 2
    tgt_seq_len = max_seq_len
    tgt_input = torch.randint(0, vocab_size, (batch_size, tgt_seq_len)).to(device)
    
    # Option 1: Provide precomputed BLIP embeddings.
    # For demonstration, we'll create a dummy tensor of shape (batch, 128, blip_input_dim)
    dummy_blip_embeddings = torch.randn(batch_size, 128, blip_input_dim).to(device)
    
    logits, mu, logvar = model(tgt_input=tgt_input, blip_embeddings=dummy_blip_embeddings)
    print("Logits shape:", logits.shape)   # Expected: (batch, 4225, vocab_size)
    print("Mu shape:", mu.shape)             # Expected: (batch, latent_dim)
    print("Logvar shape:", logvar.shape)     # Expected: (batch, latent_dim)
    
    # Option 2: Provide raw view images (for each sample, a list of 4 PIL images).
    # Here, we'll assume view_rgbs is a list of length batch_size, each element is a list of 4 dummy images.
    # For demonstration purposes, we use random tensors converted to PIL images.
    from PIL import Image
    import numpy as np
    def dummy_pil_image():
        # Create a dummy RGB image (e.g., 224x224)
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(arr)
    
    view_rgbs = [[dummy_pil_image() for _ in range(4)] for _ in range(batch_size)]
    
    logits2, mu2, logvar2 = model(tgt_input=tgt_input, blip_embeddings=None, view_rgbs=view_rgbs)
    print("Logits2 shape:", logits2.shape)
    print("Mu2 shape:", mu2.shape)
    print("Logvar2 shape:", logvar2.shape)
