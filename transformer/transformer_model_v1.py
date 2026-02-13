import torch
import torch.nn as nn
from torch.nn import functional as F

"""
DOCUMENTATION OF DEVIATIONS FROM model.py:

1. Spatio-Temporal Flattening: 
   In model.py, the transformer processed temporal sequences for individual spatial points.
   In transformer_model_v1.py, we flatten the 8x26 (Time x Space) grid into a single 208-token sequence.
   This allows the model to attend to both previous time steps and neighboring spatial points within the same time step.

2. Structured Embeddings:
   Instead of a single positional embedding, we use two separate embedding layers for Time (0-7) and Space (0-25).
   These are added to each token's representation based on its position in the 8x26 grid.

3. Input Feature Dimension:
   The input projection layer is expanded to 52 features (47 latents + 3 xyz + 1 rel_time + 1 param) 
   instead of just the latent dimension.

4. Causal Masking:
   We use a standard causal mask on the flattened sequence. This naturally supports predicting 
   the "last 4 positions in the last time period" by looking at the model's outputs for the 
   preceding positions in the sequence.

5. Unified Architecture:
   Uses nn.TransformerEncoderLayer for optimized performance, following the best practice 
   suggested in the latter part of model.py.
"""

class Block(nn.Module):
    """ Transformer block using torch.nn.TransformerEncoderLayer """
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )

    def forward(self, x, mask=None):
        # is_causal=True helps with optimization in newer torch versions
        return self.layer(x, src_mask=mask, is_causal=True)

class OrderedTransformerV1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection from 52 features to embedding dimension
        self.input_projection = nn.Linear(config.INPUT_DIM, config.EMBED_SIZE, bias=config.BIAS)
        
        # Structured Embeddings
        self.time_embeddings = nn.Embedding(config.NUM_TIME, config.EMBED_SIZE)
        self.space_embeddings = nn.Embedding(config.NUM_X, config.EMBED_SIZE)
        
        # Core transformer blocks
        self.blocks = nn.ModuleList([
            Block(config.EMBED_SIZE, config.N_HEADS, config.DROPOUT) 
            for _ in range(config.N_LAYERS)
        ])
        
        # Final normalization and output head
        self.ln_f = nn.LayerNorm(config.EMBED_SIZE)
        self.output_head = nn.Linear(config.EMBED_SIZE, config.LATENT_DIM, bias=config.BIAS)
        
        # Pre-calculate and register time/space indices for the flattened sequence
        # (8, 26) grid -> 208 sequence
        time_ids = torch.arange(config.NUM_TIME).repeat_interleave(config.NUM_X)
        space_ids = torch.arange(config.NUM_X).repeat(config.NUM_TIME)
        self.register_buffer("time_ids", time_ids)
        self.register_buffer("space_ids", space_ids)

    def forward(self, x):
        # x shape: (B, T, InputDim) where T <= 208
        B, T, C = x.shape
        
        # 1. Project to embedding space
        x = self.input_projection(x)
        
        # 2. Add embeddings (slicing to handle variable length during autoregression)
        x = x + self.time_embeddings(self.time_ids[:T]) + self.space_embeddings(self.space_ids[:T])
        
        # 3. Create Causal Mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        
        # 4. Pass through blocks
        for blk in self.blocks:
            x = blk(x, mask=mask)
            
        x = self.ln_f(x)
        
        # 5. Output head: Predict next latent for each position
        return self.output_head(x)

    @torch.no_grad()
    def predict_autoregressive(self, initial_sequence, num_to_predict):
        """
        Autoregressively predict the next 'num_to_predict' latents in the sequence.
        initial_sequence: (1, seq_len, 52) where seq_len is typically 208
        """
        self.eval()
        seq = initial_sequence.clone()
        start_pos = seq.shape[1] - num_to_predict
        
        for i in range(start_pos, seq.shape[1]):
            # Predict using all tokens up to i (exclusive)
            # The output at index i-1 is the prediction for latent at index i
            outputs = self(seq[:, :i, :])
            next_latent = outputs[:, -1, :]
            
            # Update the latent features (0-46) for the current position
            seq[:, i, :self.config.LATENT_DIM] = next_latent
            
        return seq[:, -num_to_predict:, :self.config.LATENT_DIM]
