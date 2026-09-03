"""
Model variants for the latent fluid-dynamics transformer.

CAUSALITY
=========
Every path in this file is now causal by construction, and there is a runtime
probe that proves it (`train_production_transformer_deep_dive.py
--diagnostics-only`, or `probe_causality()` in that module).

Two historical leaks are fixed here. Both made teacher-forced train/val loss
look good while autoregressive rollout could never follow:

1. `nn.MultiheadAttention(..., attn_mask=None, is_causal=True)`.
   For the MODULE api, `is_causal` is only a HINT that `attn_mask` already IS
   the causal mask -- it is not an instruction, and the contract is the exact
   inverse of `F.scaled_dot_product_attention`'s (pytorch#118972). There is
   also a path where it is silently dropped with no error (pytorch#99282).
   Replaced by `CausalSelfAttention`, which calls
   `F.scaled_dot_product_attention(..., is_causal=True)` directly -- the
   FUNCTIONAL api does treat `is_causal` as an instruction -- while keeping
   nn.MultiheadAttention's exact parameter names (`in_proj_weight`,
   `in_proj_bias`, `out_proj.weight`, `out_proj.bias`) so existing checkpoints
   still load without a key remap.

2. `nn.Conv1d(n_embd, n_embd, kernel_size=3, padding=1)` in `ConvBlock`.
   Symmetric padding means every output position saw t-1, t AND t+1. Applied
   once per block, so `N_LAYERS` blocks leaked `N_LAYERS` tokens of lookahead.
   The conv is now left-padded only (same weight shape, same state-dict key).

Because `CausalSelfAttention` is parameter-compatible with the module it
replaces, OLD checkpoints reload into the FIXED architecture. Their recorded
val L2 will therefore look worse when re-evaluated than what the trainer logged
at save time -- that delta is the direct measurement of how much the leak was
worth.

TOKEN LAYOUT
============
Sequences are flattened time-major / x-minor:

    token_index = t * NUM_X + x        (t in [0, NUM_TIME), x in [0, NUM_X))

so consecutive tokens are usually SPATIAL neighbours within one time frame, and
only every NUM_X-th token crosses a time-frame boundary. Anything that wants to
reason about "one time step" must move by NUM_X tokens, not 1.

OPTIONAL BEHAVIOUR (all off by default, all parameter-count preserving)
======================================================================
    config.ATTN_IMPL      'sdpa' (default, causal-by-construction)
                          'mha_hint' reproduces the OLD leaky behaviour, kept
                          only so the leak can be measured on purpose.
    config.USE_ROPE       rotary position embeddings on q/k inside attention.
    config.PREDICT_DELTA  output head predicts the CHANGE from the same
                          x-location one time frame earlier, and the anchor is
                          added back. Makes the persistence baseline the
                          zero-output of the network, so a zero-initialised
                          head starts exactly AT persistence instead of at
                          random. Head is zero-initialised in this mode.
    config.NORMALIZE_FEATURES
                          apply per-feature standardisation before the input
                          projection, using the `feat_mean`/`feat_std` buffers.
                          Needed because prepare_data.py writes columns 47:52
                          as raw magnitudes (x in [-29,69], y/z in +-80, t in
                          [0,39], param in [5.6,17.8]) alongside latents that
                          live in ~[0,1] -- a ~200x input-variance mismatch
                          through a single nn.Linear.
"""

import math
import os

import torch
import torch.nn as nn
from torch.nn import functional as F


# --------------------------------------------------------------------------- #
# Rotary position embeddings
# --------------------------------------------------------------------------- #
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryPositionalEmbedding(nn.Module):
    """cos/sin tables for RoPE over FLAT token position.

    Built once up to `max_seq_len` and sliced, rather than recomputed per
    forward. The autoregressive rollout calls the model with ~728 distinct
    sequence lengths; recomputing an arange+outer product for each one is pure
    overhead.
    """

    def __init__(self, head_dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head dimension"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)                 # (max_seq_len, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)          # (max_seq_len, head_dim)
        # Non-persistent: derivable from head_dim/base, so it does not belong in
        # a checkpoint and must not show up as a missing key on load.
        self.register_buffer("cos_cached", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None], persistent=False)

    def forward(self, seq_len, dtype):
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"RoPE cache built for max_seq_len={self.max_seq_len} but got "
                f"seq_len={seq_len}. Raise Config.SEQ_LEN or the cache size.")
        return (self.cos_cached[:, :, :seq_len].to(dtype),
                self.sin_cached[:, :, :seq_len].to(dtype))


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    """Causal self-attention, parameter-compatible with nn.MultiheadAttention.

    Uses the FUNCTIONAL `F.scaled_dot_product_attention(..., is_causal=True)`,
    where `is_causal` is an instruction rather than a hint, so causality does
    not depend on a mask being threaded through correctly. No explicit mask is
    materialised, which also keeps the fused FlashAttention kernels eligible.

    Parameter names deliberately mirror nn.MultiheadAttention so state dicts
    are interchangeable in both directions:
        in_proj_weight  (3E, E)
        in_proj_bias    (3E,)
        out_proj.weight (E, E)
        out_proj.bias   (E,)
    """

    def __init__(self, n_embd, n_head, dropout=0.0, bias=True, rope=None):
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd={n_embd} not divisible by n_head={n_head}"
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout_p = dropout
        self.rope = rope

        self.in_proj_weight = nn.Parameter(torch.empty(3 * n_embd, n_embd))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * n_embd)) if bias else None
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # Match nn.MultiheadAttention's init so switching ATTN_IMPL does not
        # silently change the initial-loss scale.
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x, mask=None):
        # `mask` is accepted and ignored for signature compatibility with the
        # blocks' call sites; causality comes from is_causal=True below.
        B, T, C = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            cos, sin = self.rope(T, q.dtype)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # q and k always have the same length here (no KV cache), so the
        # is_causal diagonal alignment is the intended one.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(y)


class LegacyLeakyAttention(nn.Module):
    """The ORIGINAL nn.MultiheadAttention + `is_causal` hint call.

    Retained ONLY so `ATTN_IMPL='mha_hint'` can reproduce the pre-fix numbers on
    purpose (e.g. to quantify how much the leak was worth). Do not train new
    models with this.
    """

    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)

    def forward(self, x, mask=None):
        out, _ = self.attn(x, x, x, attn_mask=None, is_causal=True, need_weights=False)
        return out


class MultiQueryAttention(nn.Module):
    """MQA: one K/V head shared across all query heads.

    Already causal-correct before this rewrite (it always called functional
    SDPA), so it is unchanged apart from RoPE plumbing and an explicit bias
    flag.
    """

    def __init__(self, n_embd, n_head, dropout=0.0, bias=True, rope=None):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout_p = dropout
        self.rope = rope

        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_proj = nn.Linear(n_embd, self.head_dim, bias=bias)
        self.v_proj = nn.Linear(n_embd, self.head_dim, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            cos, sin = self.rope(T, q.dtype)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Broadcast the single K/V head across query heads for SDPA.
        k = k.expand(-1, self.n_head, -1, -1)
        v = v.expand(-1, self.n_head, -1, -1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(y)


def _build_attention(n_embd, n_head, dropout, bias, attn_type, attn_impl, rope):
    if attn_type == 'mqa':
        return MultiQueryAttention(n_embd, n_head, dropout=dropout, bias=bias, rope=rope)
    if attn_impl == 'mha_hint':
        if rope is not None:
            raise ValueError("ATTN_IMPL='mha_hint' cannot apply RoPE (it owns its own projections)")
        return LegacyLeakyAttention(n_embd, n_head, dropout=dropout)
    return CausalSelfAttention(n_embd, n_head, dropout=dropout, bias=bias, rope=rope)


# --------------------------------------------------------------------------- #
# Feed-forward
# --------------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    def __init__(self, dim, intermediate_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_dim)
        self.w2 = nn.Linear(dim, intermediate_dim)
        self.w3 = nn.Linear(intermediate_dim, dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def _build_mlp(n_embd, dropout, use_swiglu):
    if use_swiglu:
        return SwiGLU(n_embd, 4 * n_embd)
    return nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),
        nn.GELU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout),
    )


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    """Pre-LN transformer block."""

    def __init__(self, n_embd, n_head, dropout=0.0, use_swiglu=False,
                 attn_type='base', attn_impl='sdpa', bias=True, rope=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = _build_attention(n_embd, n_head, dropout, bias, attn_type, attn_impl, rope)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = _build_mlp(n_embd, dropout, use_swiglu)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ConvBlock(nn.Module):
    """Depthwise-conv + attention block.

    The conv is CAUSAL: left-padded by `kernel_size - 1` with no right padding,
    so output position t depends on inputs <= t only. The previous
    `padding=1` was symmetric and leaked t+1 into every token of every layer.

    Weight shape and the `conv` attribute name are unchanged, so conv-variant
    checkpoints from before the fix still load (into the fixed, non-leaky
    architecture).
    """

    KERNEL_SIZE = 3

    def __init__(self, n_embd, n_head, dropout=0.0, use_swiglu=False,
                 attn_type='base', attn_impl='sdpa', bias=True, rope=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.conv = nn.Conv1d(n_embd, n_embd, kernel_size=self.KERNEL_SIZE,
                              padding=0, groups=n_embd)
        self.attn = _build_attention(n_embd, n_head, dropout, bias, attn_type, attn_impl, rope)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = _build_mlp(n_embd, dropout, use_swiglu)

    def forward(self, x, mask=None):
        x_res = x
        h = self.ln1(x)
        h = h.transpose(1, 2)                              # (B, C, T)
        h = F.pad(h, (self.KERNEL_SIZE - 1, 0))            # LEFT pad only -> causal
        h = self.conv(h).transpose(1, 2)                   # (B, T, C)
        x = x_res + self.attn(h)
        x = x + self.mlp(self.ln2(x))
        return x


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class BaseTransformer(nn.Module):
    # Token-level: the trainer/eval must drive it one (t, x) token at a time.
    frame_native = False

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.latent_dim = config.LATENT_DIM
        self.num_x = config.NUM_X
        self.num_time = config.NUM_TIME
        self.predict_delta = bool(getattr(config, 'PREDICT_DELTA', False))
        self.delta_anchor_kind = str(getattr(config, 'DELTA_ANCHOR', 'persistence'))
        self.normalize_features = bool(getattr(config, 'NORMALIZE_FEATURES', False))
        self.use_rope = bool(getattr(config, 'USE_ROPE', False))
        self.attn_impl = getattr(config, 'ATTN_IMPL', 'sdpa')
        variant = getattr(config, 'VARIANT', 'base')
        bias = bool(getattr(config, 'BIAS', True))
        dropout = float(getattr(config, 'DROPOUT', 0.0))
        use_swiglu = bool(getattr(config, 'USE_SWIGLU', False))

        # Per-feature standardisation. Persistent so a checkpoint is
        # self-contained -- eval does not have to recompute training statistics.
        self.register_buffer("feat_mean", torch.zeros(config.INPUT_DIM))
        self.register_buffer("feat_std", torch.ones(config.INPUT_DIM))

        # USE_META_COLS=False zeroes the (x, y, z, t, param) columns so the model
        # sees latents only. Position still reaches it through the time/space
        # embeddings, so this is a test of whether those raw columns were
        # actively harmful rather than merely badly scaled.
        keep = torch.ones(config.INPUT_DIM)
        if not bool(getattr(config, 'USE_META_COLS', True)):
            keep[config.LATENT_DIM:] = 0.0
        self.register_buffer("feat_keep", keep, persistent=False)

        self.input_projection = nn.Linear(config.INPUT_DIM, config.EMBED_SIZE, bias=bias)

        head_dim = config.EMBED_SIZE // config.N_HEADS
        rope = None
        if self.use_rope:
            rope = RotaryPositionalEmbedding(head_dim, max_seq_len=max(2048, config.SEQ_LEN))
            self.rope = rope

        # Absolute learned embeddings. Skipped entirely under RoPE, which
        # already supplies position -- keeping both is redundant and was the
        # thing `RoPETransformer` used to try (and fail) to express.
        if self.use_rope:
            self.time_embeddings = None
            self.space_embeddings = None
        else:
            self.time_embeddings = nn.Embedding(config.NUM_TIME, config.EMBED_SIZE)
            self.space_embeddings = nn.Embedding(config.NUM_X, config.EMBED_SIZE)

        block_cls = ConvBlock if variant == 'conv' else Block
        self.blocks = nn.ModuleList([
            block_cls(config.EMBED_SIZE, config.N_HEADS, dropout=dropout,
                      use_swiglu=use_swiglu, attn_type=variant,
                      attn_impl=self.attn_impl, bias=bias, rope=rope)
            for _ in range(config.N_LAYERS)
        ])

        self.ln_f = nn.LayerNorm(config.EMBED_SIZE)
        self.output_head = nn.Linear(config.EMBED_SIZE, config.LATENT_DIM, bias=bias)

        if self.predict_delta:
            # Start the network exactly AT persistence rather than at random:
            # a zero head means output == anchor == "same x-location, previous
            # time frame". Improvement over persistence can then only go up
            # from 0, and all capacity is spent on dynamics instead of on
            # relearning the identity.
            nn.init.zeros_(self.output_head.weight)
            if self.output_head.bias is not None:
                nn.init.zeros_(self.output_head.bias)

        time_ids = torch.arange(config.NUM_TIME).repeat_interleave(config.NUM_X)
        space_ids = torch.arange(config.NUM_X).repeat(config.NUM_TIME)
        self.register_buffer("time_ids", time_ids, persistent=False)
        self.register_buffer("space_ids", space_ids, persistent=False)

        # Legacy no-op buffer: old checkpoints contain a "causal_mask" key.
        # Non-persistent and 1x1 so it costs nothing and does not reappear in
        # new checkpoints.
        self.register_buffer("causal_mask", torch.zeros(1, 1), persistent=False)

        # `ridge_A` is ALWAYS a real tensor buffer (never None), so
        # `self.delta_anchor_kind == 'ridge'` -- a plain string set once at
        # construction, not data-dependent -- is the only thing gating its
        # use in forward(). When DELTA_ANCHOR != 'ridge' this is an unused
        # 1x1 placeholder, matching `causal_mask`'s pattern above.
        if self.predict_delta and self.delta_anchor_kind == 'ridge':
            ridge_path = getattr(config, 'RIDGE_MAP_PATH', None)
            if not ridge_path or not os.path.exists(ridge_path):
                raise FileNotFoundError(
                    f"DELTA_ANCHOR='ridge' requires Config.RIDGE_MAP_PATH to "
                    f"point at a fitted ridge map -- run diagnostics first "
                    f"(linear_frame_baseline() saves it there): {ridge_path}")
            payload = torch.load(ridge_path, map_location='cpu')
            ridge_A = payload['A'].float()
            expected_d = config.NUM_X * config.LATENT_DIM
            if tuple(ridge_A.shape) != (expected_d + 1, expected_d):
                raise ValueError(
                    f"ridge map at {ridge_path} has shape {tuple(ridge_A.shape)}, "
                    f"expected {(expected_d + 1, expected_d)} for this Config's "
                    f"NUM_X={config.NUM_X}, LATENT_DIM={config.LATENT_DIM} "
                    f"(it was fit under a different data/shape configuration).")
            self.register_buffer("ridge_A", ridge_A)
        else:
            self.register_buffer("ridge_A", torch.zeros(1, 1))

    # -- feature statistics -------------------------------------------------
    def set_feature_stats(self, mean, std, eps=1e-6):
        """Install per-feature standardisation statistics (in-place)."""
        with torch.no_grad():
            self.feat_mean.copy_(mean.to(self.feat_mean.device, self.feat_mean.dtype))
            std = std.to(self.feat_std.device, self.feat_std.dtype).clone()
            # Constant columns (e.g. a param column in a single-condition split)
            # would otherwise divide by ~0 and blow the projection up.
            std[std < eps] = 1.0
            self.feat_std.copy_(std)

    # -- delta anchor -------------------------------------------------------
    def _delta_anchor(self, raw_lat):
        """Latents at the same x-location one time frame earlier.

        Output position t predicts token t+1, whose persistence anchor is token
        (t+1) - NUM_X, i.e. raw_lat[t - (NUM_X - 1)]. Positions with no such
        anchor fall back to the current token (spatial persistence).
        """
        k = self.num_x - 1
        T = raw_lat.shape[1]
        # Trace-safe equivalent of `if T <= k: return raw_lat else: ...`.
        # A Python `if` on a shape-derived value trips torch.jit.trace's
        # "Converting a tensor to a Python boolean" TracerWarning AND bakes
        # in whichever branch was taken at trace time -- any later input
        # whose T falls on the other side of the threshold (e.g. the first
        # few tokens of an AR rollout) would then silently get the wrong
        # branch. Clamping k to T instead is a single unconditional
        # expression, correct for every T: when T <= k it degenerates to
        # `raw_lat[:, :T]` concatenated with an empty slice, i.e. `raw_lat`
        # unchanged; when T > k it's exactly the original shifted-concat.
        k_eff = min(k, T)
        return torch.cat([raw_lat[:, :k_eff], raw_lat[:, :T - k_eff]], dim=1)

    def _ridge_anchor(self, raw_lat):
        """Ridge-regression-map prediction anchor, used instead of
        `_delta_anchor`'s straight persistence copy when
        `DELTA_ANCHOR='ridge'`. `linear_frame_baseline()` found this fitted
        linear map beats persistence by +69% in the same decoded-velocity
        space `evaluate()` scores in -- a strictly stronger anchor than
        copying the previous frame, so predicting the RESIDUAL on top of it
        should need less of the network's capacity spent re-deriving
        something a closed-form regression already gets mostly right.

        At target token (t+1) belonging to a complete source frame (i.e.
        t+1 >= NUM_X and its source frame lies fully within `raw_lat`),
        returns the ridge map's prediction for that frame from the frame
        immediately before it. Falls back to `_delta_anchor`'s persistence
        value wherever no complete source frame is available: the leading
        NUM_X-1 positions, or -- during AR rollout, where `curr` grows one
        token at a time -- whenever `raw_lat` ends mid-frame.

        Unlike `_delta_anchor`, this branches on a shape-derived value
        (`n_complete`), so it is only guaranteed correct under
        `torch.jit.script` (tried first by `save_scripted_model()`, and
        compiles real control flow, unlike trace). If scripting ever falls
        back to `torch.jit.trace` for a `DELTA_ANCHOR='ridge'` model, the
        traced graph would fix whichever branch was taken at trace time --
        accepted here since this is a research sweep arm, not a deployed
        inference path.
        """
        fallback = self._delta_anchor(raw_lat)
        B, T, LD = raw_lat.shape
        NX = self.num_x
        n_complete = T // NX
        if n_complete < 1:
            return fallback
        D = NX * LD
        frames = raw_lat[:, :n_complete * NX, :].reshape(B, n_complete, D)
        ones = torch.ones(B, n_complete, 1, dtype=frames.dtype, device=frames.device)
        src1 = torch.cat([frames, ones], dim=-1)
        # Same bf16/float32 guard as `decode_centroid()`: the ridge map is a
        # frozen float32 buffer, but this runs inside whatever CUDA autocast
        # region the caller is in.
        with torch.autocast(device_type='cuda', enabled=False):
            pred = src1.float() @ self.ridge_A.float()      # (B, n_complete, D) -- predicts frames 1..n_complete
        pred = pred.to(raw_lat.dtype).reshape(B, n_complete * NX, LD)
        start = NX - 1
        end = min(T, start + pred.shape[1])
        out = fallback.clone()
        out[:, start:end, :] = pred[:, :end - start, :]
        return out

    def forward(self, x):
        B, T, C = x.shape
        raw_lat = x[..., :self.latent_dim]

        h = x
        if self.normalize_features:
            h = (h - self.feat_mean) / self.feat_std
        h = h * self.feat_keep

        h = self.input_projection(h)
        if self.time_embeddings is not None:
            h = h + self.time_embeddings(self.time_ids[:T]) + self.space_embeddings(self.space_ids[:T])

        for blk in self.blocks:
            h = blk(h)

        out = self.output_head(self.ln_f(h))

        if self.predict_delta:
            if self.delta_anchor_kind == 'ridge':
                out = out + self._ridge_anchor(raw_lat)
            else:
                out = out + self._delta_anchor(raw_lat)
        return out


# --------------------------------------------------------------------------- #
# Frame-level tokenisation
# --------------------------------------------------------------------------- #
# Number of per-frame scalar features appended to the flattened latents:
# (t_index, y, z, param). `x` is excluded on purpose -- it varies WITHIN a frame
# but is identical for every frame of every sequence, so once a frame is one
# token the x column carries zero information.
FRAME_META_COLS = 4


def seq_to_frames(batch, num_x, latent_dim):
    """(B, T, 52) token sequence -> (B, T//num_x, num_x*latent_dim + 4) frames.

    Relies on the time-major/x-minor flattening (token = t*num_x + x), so a
    plain reshape recovers frames. Meta columns are read from the first token of
    each frame, where they are constant across x.

    Column indices come from prepare_data.py's layout:
        0:47 latents, 47 x, 48 y, 49 z, 50 t_index, 51 param
    """
    B, T, C = batch.shape
    if T % num_x != 0:
        raise ValueError(f"frame tokenisation needs T divisible by num_x={num_x}, got T={T}")
    n_frames = T // num_x
    lat = batch[..., :latent_dim].reshape(B, n_frames, num_x * latent_dim)
    meta = batch[:, ::num_x, :][:, :, [50, 48, 49, 51]]      # t, y, z, param
    return torch.cat([lat, meta], dim=-1)


class FrameTransformer(nn.Module):
    """One token per TIME FRAME instead of one token per (time, x) pair.

    Motivation: with token-level flattening the sequence is NUM_TIME*NUM_X tokens
    of which only every NUM_X-th crosses a time-frame boundary, so a next-token objective is
    dominated by SPATIAL continuation within a frame and spends most of its
    capacity there. Collapsing each frame into a single 1222-dim token makes the
    sequence NUM_TIME tokens long and the objective purely temporal -- which is the
    quantity the persistence baseline is actually measured on.

    Side benefit: rollout is one step per frame instead of one per token, so
    evaluation is approximately NUM_X times cheaper.

    Trade-off: there is no within-frame autoregression, so frame f is predicted
    from frames < f only (the token model additionally sees earlier x-positions
    of frame f). Whether that costs or helps is exactly what the sweep measures.
    """

    frame_native = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config.LATENT_DIM
        self.num_x = config.NUM_X
        self.num_time = config.NUM_TIME
        self.frame_dim = config.NUM_X * config.LATENT_DIM
        self.in_dim = self.frame_dim + FRAME_META_COLS
        self.predict_delta = bool(getattr(config, 'PREDICT_DELTA', False))
        self.normalize_features = bool(getattr(config, 'NORMALIZE_FEATURES', False))
        self.use_rope = bool(getattr(config, 'USE_ROPE', False))
        bias = bool(getattr(config, 'BIAS', True))
        dropout = float(getattr(config, 'DROPOUT', 0.0))
        use_swiglu = bool(getattr(config, 'USE_SWIGLU', False))
        variant = getattr(config, 'VARIANT', 'base')

        self.register_buffer("feat_mean", torch.zeros(self.in_dim))
        self.register_buffer("feat_std", torch.ones(self.in_dim))

        keep = torch.ones(self.in_dim)
        if not bool(getattr(config, 'USE_META_COLS', True)):
            keep[self.frame_dim:] = 0.0        # the 4 per-frame meta scalars
        self.register_buffer("feat_keep", keep, persistent=False)

        self.input_projection = nn.Linear(self.in_dim, config.EMBED_SIZE, bias=bias)

        head_dim = config.EMBED_SIZE // config.N_HEADS
        rope = RotaryPositionalEmbedding(head_dim, max_seq_len=max(512, config.NUM_TIME)) \
            if self.use_rope else None
        if rope is not None:
            self.rope = rope
            self.time_embeddings = None
        else:
            self.time_embeddings = nn.Embedding(config.NUM_TIME, config.EMBED_SIZE)

        self.blocks = nn.ModuleList([
            Block(config.EMBED_SIZE, config.N_HEADS, dropout=dropout,
                  use_swiglu=use_swiglu,
                  attn_type=('mqa' if variant == 'mqa' else 'base'),
                  attn_impl=getattr(config, 'ATTN_IMPL', 'sdpa'),
                  bias=bias, rope=rope)
            for _ in range(config.N_LAYERS)
        ])
        self.ln_f = nn.LayerNorm(config.EMBED_SIZE)
        self.output_head = nn.Linear(config.EMBED_SIZE, self.frame_dim, bias=bias)

        if self.predict_delta:
            nn.init.zeros_(self.output_head.weight)
            if self.output_head.bias is not None:
                nn.init.zeros_(self.output_head.bias)

        self.register_buffer("frame_ids", torch.arange(config.NUM_TIME), persistent=False)

    def set_feature_stats(self, mean, std, eps=1e-6):
        with torch.no_grad():
            self.feat_mean.copy_(mean.to(self.feat_mean.device, self.feat_mean.dtype))
            std = std.to(self.feat_std.device, self.feat_std.dtype).clone()
            std[std < eps] = 1.0
            self.feat_std.copy_(std)

    def forward(self, frames):
        """frames: (B, F, frame_dim + FRAME_META_COLS) -> (B, F, frame_dim).

        Output position f is the prediction of frame f+1, built from frames <= f.
        """
        B, F_, _ = frames.shape
        anchor = frames[..., :self.frame_dim]

        h = frames
        if self.normalize_features:
            h = (h - self.feat_mean) / self.feat_std
        h = h * self.feat_keep
        h = self.input_projection(h)
        if self.time_embeddings is not None:
            h = h + self.time_embeddings(self.frame_ids[:F_])

        for blk in self.blocks:
            h = blk(h)

        out = self.output_head(self.ln_f(h))
        if self.predict_delta:
            out = out + anchor
        return out


def get_model(config):
    """Build the model described by `config`.

    `TOKENIZATION` selects token-level (NUM_TIME*NUM_X tokens) vs frame-level
    (NUM_TIME tokens);
    `VARIANT` selects the attention/block type within that. Kept as a function
    rather than a class registry because checkpoints record these as strings and
    the leaderboard test rebuilds from them.
    """
    variant = getattr(config, 'VARIANT', 'base')
    tokenization = getattr(config, 'TOKENIZATION', 'token')

    if tokenization == 'frame':
        if variant == 'swiglu':
            config.USE_SWIGLU = True
        elif variant == 'conv':
            raise ValueError("VARIANT='conv' is token-level only; the frame "
                             "sequence has NUM_TIME tokens and a depthwise conv over "
                             "it mixes time frames without adding locality.")
        return FrameTransformer(config)
    if tokenization != 'token':
        raise ValueError(f"Unknown TOKENIZATION {tokenization!r}; expected 'token' or 'frame'")

    if variant == 'swiglu':
        # Historical behaviour: the 'swiglu' variant name implies USE_SWIGLU.
        # Mutating config here is load-bearing for checkpoint round-tripping,
        # so it is kept -- see reset_config() in tests/test_model_vs_baseline.py.
        config.USE_SWIGLU = True
    elif variant not in ('base', 'mqa', 'conv'):
        raise ValueError(
            f"Unknown VARIANT {variant!r}; expected one of "
            f"'base', 'swiglu', 'mqa', 'conv'")
    return BaseTransformer(config)
