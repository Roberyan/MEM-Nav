import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import gelu

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
            attention_mask: Optional mask of shape (batch, seq_len) where masked positions
                            are set to a large negative value.
        Returns:
            representations: Tensor of shape (batch, hidden_size), the weighted sum.
            weights: The attention weights (after softmax) of shape (batch, seq_len).
        """
        # Compute raw scores via dot product between each token and the learned weight.
        scores = self.non_linearity(torch.matmul(inputs, self.attention_weights))  # (batch, seq_len)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax to obtain attention weights.
        weights = self.softmax(scores)  # (batch, seq_len)
        # Weighted sum over tokens.
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
    def __init__(self, vocab_size, model_dim, num_layers, num_heads, dropout=0.1, max_seq_len=256):
        """
        Args:
            vocab_size: Vocabulary size for output tokens.
            model_dim: Hidden dimension of the transformer.
            num_layers: Number of decoder layers.
            num_heads: Number of attention heads.
            max_seq_len: Maximum output sequence length.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, model_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # tgt: (batch, tgt_seq_len) token indices
        batch_size, tgt_seq_len = tgt.size()
        positions = torch.arange(0, tgt_seq_len, device=tgt.device).unsqueeze(0).expand(batch_size, tgt_seq_len)
        tgt_emb = self.embedding(tgt) + self.pos_embedding(positions)  # (batch, tgt_seq_len, model_dim)
        tgt_emb = tgt_emb.transpose(0, 1)                               # (tgt_seq_len, batch, model_dim)
        memory = memory.transpose(0, 1)                                 # (mem_seq_len, batch, model_dim)
        dec_out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        dec_out = dec_out.transpose(0, 1)                               # (batch, tgt_seq_len, model_dim)
        logits = self.fc_out(dec_out)                                   # (batch, tgt_seq_len, vocab_size)
        return logits

#######################################
#  Transformer-based CVAE with MEM    #
#      Dual Pooling Options             #
#######################################
class TransformerCVAE(nn.Module):
    def __init__(self, encoder_input_dim, vocab_size,
                 model_dim=512, latent_dim=64,
                 enc_layers=4, dec_layers=4,
                 num_heads=8, dropout=0.1,
                 max_seq_len=256,
                 use_attn_pooling=True):
        """
        Args:
            encoder_input_dim: Dimensionality of each token in the conditional input.
            vocab_size: Size of the output vocabulary.
            model_dim: Transformer hidden dimension.
            latent_dim: Dimensionality of the latent variable z.
            use_attn_pooling: If True, use attention-based pooling; otherwise use mean pooling.
        """
        super().__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.use_attn_pooling = use_attn_pooling

        # Encoder for conditional input.
        self.encoder = TransformerEncoder(encoder_input_dim, model_dim, num_layers=enc_layers, num_heads=num_heads, dropout=dropout)
        
        # Pooling module: either attention-based or simple mean.
        if use_attn_pooling:
            self.pooling = AverageSelfAttention(model_dim)
        else:
            self.pooling = None  # We'll use mean pooling in forward()
        
        # Posterior projection: from pooled representation to Gaussian parameters.
        self.fc_mu = nn.Linear(model_dim, latent_dim)
        self.fc_logvar = nn.Linear(model_dim, latent_dim)
        
        # Project latent variable into a memory token for the decoder.
        self.latent_to_memory = nn.Linear(latent_dim, model_dim)
        
        # Decoder for output tokens.
        self.decoder = TransformerDecoder(vocab_size, model_dim, num_layers=dec_layers, num_heads=num_heads, dropout=dropout, max_seq_len=max_seq_len)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, encoder_inputs, tgt_input, tgt_mask=None, memory_mask=None):
        """
        Args:
            encoder_inputs: (batch, src_seq_len, encoder_input_dim) conditional input tokens.
            tgt_input: (batch, tgt_seq_len) decoder input tokens.
        Returns:
            logits: (batch, tgt_seq_len, vocab_size)
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        # Encode the conditional input.
        enc_out = self.encoder(encoder_inputs)  # (batch, src_seq_len, model_dim)
        
        # Pooling step: either attention-based or simple mean.
        if self.use_attn_pooling:
            pooled, attn_scores = self.pooling(enc_out)  # (batch, model_dim)
        else:
            pooled = enc_out.mean(dim=1)  # (batch, model_dim)
        
        # Compute posterior parameters.
        mu = self.fc_mu(pooled)            # (batch, latent_dim)
        logvar = self.fc_logvar(pooled)      # (batch, latent_dim)
        
        # Sample latent variable.
        z = self.reparameterize(mu, logvar)   # (batch, latent_dim)
        
        # Project latent variable into a memory token (for cross-attention).
        memory_token = self.latent_to_memory(z).unsqueeze(1)  # (batch, 1, model_dim)
        
        # Decode the target sequence using the memory token.
        logits = self.decoder(tgt_input, memory_token, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return logits, mu, logvar

#######################################
#             Example Usage           #
#######################################
if __name__ == "__main__":
    # Configuration parameters.
    encoder_input_dim = 768    # BLIP-2 embedding size per token.
    model_dim = 512            # Transformer hidden dimension.
    latent_dim = 64            # Latent variable dimension; try 64, can be increased if needed.
    enc_layers = 4             # Number of transformer encoder layers.
    dec_layers = 4             # Number of transformer decoder layers.
    num_heads = 8              # Number of attention heads.
    dropout = 0.1              # Dropout rate.
    max_seq_len = 256          # Maximum output sequence length (e.g., for a 16x16 tokenized map, 256 tokens).
    vocab_size = 12            # Vocabulary size for the output tokens (e.g., 10 semantic classes + special tokens).
    use_attn_pooling = True    # Use attention-based pooling for latent extraction.


    # Instantiate the model.
    model = TransformerCVAE(
        encoder_input_dim=encoder_input_dim,
        vocab_size=vocab_size,
        model_dim=model_dim,
        latent_dim=latent_dim,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_seq_len=max_seq_len,
        use_attn_pooling=use_attn_pooling
    )

    # Dummy inputs:
    batch_size = 8
    src_seq_len = 10   # e.g., number of tokens from a multimodal encoder
    tgt_seq_len = 20   # e.g., target sequence length

    # Random conditional input tokens.
    encoder_inputs = torch.randn(batch_size, src_seq_len, encoder_input_dim)
    # Dummy target input tokens (integer indices).
    tgt_input = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))

    # Forward pass.
    logits, mu, logvar = model(encoder_inputs, tgt_input)
    print("Logits shape:", logits.shape)  # Expected: (batch_size, tgt_seq_len, vocab_size)
    print("mu shape:", mu.shape)            # Expected: (batch_size, latent_dim)
    print("logvar shape:", logvar.shape)    # Expected: (batch_size, latent_dim)
