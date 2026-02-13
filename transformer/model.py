import torch
import torch.nn as nn
from torch.nn import functional as F
import config


class Head(nn.Module):
    """ One head of self-attention (uses PyTorch SDPA for speed) """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.EMBED_SIZE, head_size, bias=config.BIAS)
        self.query = nn.Linear(config.EMBED_SIZE, head_size, bias=config.BIAS)
        self.value = nn.Linear(config.EMBED_SIZE, head_size, bias=config.BIAS)
        # Dropout on attention probabilities (handled by SDPA)
        self.attn_dropout = config.DROPOUT

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # Use scaled_dot_product_attention with causal mask to leverage FlashAttention kernels on supported hardware
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.EMBED_SIZE, config.EMBED_SIZE, bias=config.BIAS)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=config.BIAS),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=config.BIAS),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


# class Block(nn.Module):
#     """ Transformer block: communication followed by computation """
#
#     def __init__(self, n_embd, n_head):
#         super().__init__()
#         head_size = n_embd // n_head
#         self.sa = MultiHeadAttention(n_head, head_size)
#         self.ffwd = FeedForward(n_embd)
#         self.ln1 = nn.LayerNorm(n_embd)
#         self.ln2 = nn.LayerNorm(n_embd)
#
#     def forward(self, x):
#         x = x + self.sa(self.ln1(x))
#         x = x + self.ffwd(self.ln2(x))
#         return x

class Block(nn.Module):
    """ Transformer block using torch.nn.TransformerEncoderLayer """

    def __init__(self, n_embd, n_head):
        super().__init__()
        # Use the built-in, optimized layer
        self.layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=config.DROPOUT,
            activation=F.gelu,  # Match your FeedForward's GELU
            batch_first=True,   # IMPORTANT: Our data is (B, T, C)
            norm_first=True     # This is Pre-LN (like your ln1)
        )

    def forward(self, x):
        # The built-in layer expects no causal mask by default in forward
        # but F.scaled_dot_product_attention *inside* it will use one
        # if we pass src_mask.
        # Your original code used is_causal=True in SDPA.
        # We must create and pass a causal mask to TransformerEncoderLayer.

        # Create causal mask - torch.compile will optimize this away
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        # x shape is (B, T, C)
        # mask shape must be (T, T)
        return self.layer(x, src_mask=causal_mask, is_causal=False)

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, predict_horizon=1):
        super().__init__()
        self.predict_horizon = predict_horizon

        # Input projection from latent space to embedding dimension
        self.input_projection = nn.Linear(config.LATENT_DIM, config.EMBED_SIZE, bias=config.BIAS)

        # Spatial embedding to give the model a sense of location
        self.spatial_embeddings = nn.Embedding(config.NUM_SPATIAL_POINTS, config.EMBED_SIZE)

        # Temporal position embedding
        self.positional_embeddings = nn.Embedding(config.CONTEXT_WINDOW, config.EMBED_SIZE)

        # Register position indices as buffer (created once, not every forward)
        self.register_buffer("position_ids", torch.arange(config.CONTEXT_WINDOW), persistent=False)

        # Core transformer blocks (ModuleList to support gradient checkpointing)
        self.blocks = nn.ModuleList([Block(config.EMBED_SIZE, n_head=config.N_HEADS) for _ in range(config.N_LAYERS)])

        # Final normalization and output layer
        self.ln_f = nn.LayerNorm(config.EMBED_SIZE)
        self.output_head = nn.Linear(config.EMBED_SIZE, config.LATENT_DIM * predict_horizon, bias=config.BIAS)

    def forward(self, input_seq, spatial_ids):
        # input_seq shape: (B, T, LatentDim) where T = context_window
        # spatial_ids shape: (B,)
        # output shape: (B, predict_horizon, LatentDim)
        B, T, _ = input_seq.shape

        # 1. Project latent vectors into embedding space
        projected_input = self.input_projection(input_seq)  # (B, T, EmbedSize)

        # 2. Add positional embeddings (use cached position_ids buffer)
        pos_emb = self.positional_embeddings(self.position_ids[:T])  # (T, EmbedSize)

        # 3. Add spatial embeddings
        # spatial_ids (B,) -> (B, 1, EmbedSize) to allow broadcasting over time dimension
        spatial_emb = self.spatial_embeddings(spatial_ids).unsqueeze(1)

        # Combine embeddings
        x = projected_input + pos_emb + spatial_emb

        # 4. Pass through transformer blocks with optional gradient checkpointing
        use_ckpt = bool(getattr(config, 'GRADIENT_CHECKPOINTING', False)) and self.training and x.requires_grad
        if use_ckpt:
            from torch.utils.checkpoint import checkpoint
            for blk in self.blocks:
                x = checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)
        x = self.ln_f(x)

        # 5. Use only the last timestep's representation to predict future
        last_hidden = x[:, -1, :]  # (B, EmbedSize)

        # 6. Project to predict_horizon future steps
        predictions = self.output_head(last_hidden)  # (B, LatentDim * predict_horizon)
        predictions = predictions.view(B, self.predict_horizon, config.LATENT_DIM)  # (B, predict_horizon, LatentDim)

        return predictions

    @torch.no_grad()
    def predict_future(self, initial_seq, spatial_id, steps_to_predict):
        """
        Autoregressively predict future latent vectors.
        initial_seq: A tensor of shape (ContextWindow, LatentDim)
        spatial_id: A single integer ID for the spatial location.
        """
        self.eval()
        # Add a batch dimension and move to device
        current_seq = initial_seq.unsqueeze(0).to(config.DEVICE)
        spatial_id_tensor = torch.tensor([spatial_id], device=config.DEVICE)

        predictions = []

        for _ in range(steps_to_predict):
            # Get model output for the current sequence
            prediction_for_all_steps = self(current_seq, spatial_id_tensor)

            # We only need the prediction for the very last time step
            next_step_prediction = prediction_for_all_steps[:, -1:, :]  # (1, 1, LatentDim)

            predictions.append(next_step_prediction.squeeze(0).cpu())

            # Append the prediction to the sequence and drop the oldest element
            current_seq = torch.cat([current_seq[:, 1:, :], next_step_prediction], dim=1)

        return torch.cat(predictions, dim=0)