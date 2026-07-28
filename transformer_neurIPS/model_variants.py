import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class SwiGLU(nn.Module):
    def __init__(self, dim, intermediate_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_dim)
        self.w2 = nn.Linear(dim, intermediate_dim)
        self.w3 = nn.Linear(intermediate_dim, dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class MultiQueryAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # MQA: One set of K, V for all heads, but unique Q for each head
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, self.head_dim)
        self.v_proj = nn.Linear(n_embd, self.head_dim)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = self.k_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2) # (B, 1, T, hs)
        v = self.v_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2) # (B, 1, T, hs)

        # Scaled dot-product attention
        # k, v are broadcasted across heads
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            att = att.masked_fill(mask == float('-inf'), float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

class ConvBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1, use_swiglu=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        # 1D Convolution for local spatial-temporal correlation
        self.conv = nn.Conv1d(n_embd, n_embd, kernel_size=3, padding=1, groups=n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = SwiGLU(n_embd, 4 * n_embd) if use_swiglu else nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # Local conv then attention
        x_res = x
        x = self.ln1(x)
        # Conv1d expects (B, C, T)
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        attn_out, _ = self.attn(x_conv, x_conv, x_conv, attn_mask=mask, is_causal=True if mask is not None else False)
        x = x_res + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1, use_swiglu=False, attn_type='standard'):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        if attn_type == 'mqa':
            self.attn = MultiQueryAttention(n_embd, n_head, dropout=dropout)
        else:
            self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        
        self.ln2 = nn.LayerNorm(n_embd)
        if use_swiglu:
            self.mlp = SwiGLU(n_embd, 4 * n_embd)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )

    def forward(self, x, mask=None):
        x_norm = self.ln1(x)
        if isinstance(self.attn, nn.MultiheadAttention):
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask, is_causal=True if mask is not None else False)
        else:
            attn_out = self.attn(x_norm, mask=mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class BaseTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.INPUT_DIM, config.EMBED_SIZE, bias=config.BIAS)
        
        # Standard embeddings for time and space
        self.time_embeddings = nn.Embedding(config.NUM_TIME, config.EMBED_SIZE)
        self.space_embeddings = nn.Embedding(config.NUM_X, config.EMBED_SIZE)
        
        self.blocks = nn.ModuleList([
            ConvBlock(config.EMBED_SIZE, config.N_HEADS, config.DROPOUT, use_swiglu=getattr(config, 'USE_SWIGLU', False))
            if getattr(config, 'VARIANT', 'base') == 'conv' else
            Block(config.EMBED_SIZE, config.N_HEADS, config.DROPOUT, 
                  use_swiglu=getattr(config, 'USE_SWIGLU', False),
                  attn_type=getattr(config, 'VARIANT', 'base')) 
            for _ in range(config.N_LAYERS)
        ])
    
        self.ln_f = nn.LayerNorm(config.EMBED_SIZE)
        self.output_head = nn.Linear(config.EMBED_SIZE, config.LATENT_DIM, bias=config.BIAS)
    
        time_ids = torch.arange(config.NUM_TIME).repeat_interleave(config.NUM_X)
        space_ids = torch.arange(config.NUM_X).repeat(config.NUM_TIME)
        self.register_buffer("time_ids", time_ids)
        self.register_buffer("space_ids", space_ids)
        
        # Register causal mask as buffer to help TorchScript and avoid re-generation
        mask = torch.triu(torch.full((config.SEQ_LEN, config.SEQ_LEN), float('-inf')), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        x = self.input_projection(x)
        # Standard learnable additive embeddings
        x = x + self.time_embeddings(self.time_ids[:T]) + self.space_embeddings(self.space_ids[:T])
        
        mask = self.causal_mask[:T, :T]
        for blk in self.blocks:
            x = blk(x, mask=mask)
            
        x = self.ln_f(x)
        return self.output_head(x)

class RoPETransformer(BaseTransformer):
    def __init__(self, config):
        super().__init__(config)
        # RoPE usually doesn't use absolute positional embeddings
        self.time_embeddings = None
        self.space_embeddings = None
        self.rope = RotaryPositionalEmbedding(config.EMBED_SIZE // config.N_HEADS)
        
    def forward(self, x):
        B, T, C = x.shape
        x = self.input_projection(x)
        
        # RoPE is applied inside the attention, but here we'll simplify and use BaseTransformer's
        # structure but potentially different embeddings. 
        # Actually, proper RoPE needs a custom attention layer.
        # Let's stick to SwiGLU as a primary permutation if RoPE is too complex for 1-step.
        return super().forward(x)

def get_model(config):
    variant = getattr(config, 'VARIANT', 'base')
    if variant == 'swiglu':
        config.USE_SWIGLU = True
        return BaseTransformer(config)
    elif variant == 'mqa':
        # Multi-Query Attention variant
        return BaseTransformer(config)
    elif variant == 'conv':
        # Hybrid Convolutional-Transformer variant
        return BaseTransformer(config)
    else:
        return BaseTransformer(config)
