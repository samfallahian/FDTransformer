import torch
import torch.nn as nn
from torch.nn import functional as F

class Block(nn.Module):
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
        return self.layer(x, src_mask=mask, is_causal=True)

class OrderedTransformerNeurIPS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.input_projection = nn.Linear(config.INPUT_DIM, config.EMBED_SIZE, bias=config.BIAS)
        
        self.time_embeddings = nn.Embedding(config.NUM_TIME, config.EMBED_SIZE)
        self.space_embeddings = nn.Embedding(config.NUM_X, config.EMBED_SIZE)
        
        self.blocks = nn.ModuleList([
            Block(config.EMBED_SIZE, config.N_HEADS, config.DROPOUT) 
            for _ in range(config.N_LAYERS)
        ])
    
        self.ln_f = nn.LayerNorm(config.EMBED_SIZE)
        self.output_head = nn.Linear(config.EMBED_SIZE, config.LATENT_DIM, bias=config.BIAS)
    
        time_ids = torch.arange(config.NUM_TIME).repeat_interleave(config.NUM_X)
        space_ids = torch.arange(config.NUM_X).repeat(config.NUM_TIME)
        self.register_buffer("time_ids", time_ids)
        self.register_buffer("space_ids", space_ids)

    def forward(self, x):
        B, T, C = x.shape
        
        x = self.input_projection(x)
        x = x + self.time_embeddings(self.time_ids[:T]) + self.space_embeddings(self.space_ids[:T])
        
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        
        for blk in self.blocks:
            x = blk(x, mask=mask)
            
        x = self.ln_f(x)
        return self.output_head(x)
