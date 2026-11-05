"""Regression tests for the 2026-11-05 fixes (docs/CHANGES_2026-11-05.md).

All synthetic and CPU-only, a few seconds total. Tests that need the frozen
GEN3 decoder skip when `assets/Model_GEN3_05_..._scripted.pt` is absent.
"""
import copy
import os
import sys
import tempfile
import types

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_MOD_DIR = os.path.dirname(_HERE)
if _MOD_DIR not in sys.path:
    sys.path.insert(0, _MOD_DIR)

import model_variants as MV                       # noqa: E402
import train_production_transformer_deep_dive as T  # noqa: E402

HAVE_DECODER = os.path.exists(
    os.environ.get("PFD_DECODER_PATH") or T.Config.DECODER_SCRIPTED_PATH)


def tiny_cfg(**over):
    """A SimpleNamespace copy of Config with a tiny model."""
    d = dict(T.config_dict())
    d.update(EMBED_SIZE=32, N_LAYERS=1, N_HEADS=2, DROPOUT=0.0)
    d.update(over)
    return types.SimpleNamespace(**d)


def with_stats(model, cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    width = model.feat_mean.numel()
    mean = torch.randn(width, generator=g) * 0.01
    std = torch.rand(width, generator=g) * 0.03 + 0.01
    model.set_feature_stats(mean, std)
    return model


def synth_tokens(cfg, B=3, params=(5.6, 10.0, 17.8), seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, cfg.SEQ_LEN, cfg.INPUT_DIM, generator=g) * 0.02
    t = torch.arange(cfg.NUM_TIME).repeat_interleave(cfg.NUM_X).float()
    x[..., cfg.LATENT_DIM + 3] = t
    for i in range(B):
        x[i, :, cfg.LATENT_DIM + 4] = params[i % len(params)]
    return x


# --------------------------------------------------------------- fix 1 ----
def test_latent_loss_reaches_every_latent_dim():
    pred = torch.zeros(2, 5, 47, requires_grad=True)
    tgt = torch.randn(2, 5, 47)
    std = torch.rand(47) + 0.5
    T.latent_consistency_loss(pred, tgt, std).backward()
    assert (pred.grad.abs().sum(dim=(0, 1)) > 0).all()


# --------------------------------------------------------------- fix 7 ----
@pytest.mark.parametrize("delta", [False, True])
@pytest.mark.parametrize("tok", ["token", "frame"])
def test_head_conversion_is_exact(delta, tok):
    torch.manual_seed(0)
    cfg_raw = tiny_cfg(OUTPUT_NORMALIZED=False, PREDICT_DELTA=delta, TOKENIZATION=tok)
    cfg_new = tiny_cfg(OUTPUT_NORMALIZED=True, PREDICT_DELTA=delta, TOKENIZATION=tok)
    raw = with_stats(MV.get_model(cfg_raw), cfg_raw).eval()
    with torch.no_grad():                      # non-trivial head even under delta zero-init
        raw.output_head.weight.normal_(0, 0.1)
        raw.output_head.bias.normal_(0, 0.1)
    new = MV.get_model(cfg_new).eval()
    sd = copy.deepcopy(raw.state_dict())
    assert T.convert_head_to_normalized(sd, new, log=lambda *_: None)
    new.load_state_dict(sd, strict=False)
    x = synth_tokens(cfg_raw, B=2)
    if tok == "frame":
        x = MV.seq_to_frames(x, cfg_raw.NUM_X, cfg_raw.LATENT_DIM)[:, :6]
    else:
        x = x[:, :40]
    with torch.no_grad():
        torch.testing.assert_close(new(x), raw(x), rtol=1e-4, atol=1e-5)


def test_normalized_head_outputs_on_latent_scale_at_init():
    cfg = tiny_cfg(OUTPUT_NORMALIZED=True)
    m = with_stats(MV.get_model(cfg), cfg).eval()
    with torch.no_grad():
        out = m(synth_tokens(cfg, B=2)[:, :40])
    # de-normalised by feat_std (<= 0.04): far below the ~O(1) raw-head scale
    assert out.abs().mean() < 0.1


# --------------------------------------------------------------- fix 8 ----
@pytest.mark.parametrize("fp32", [True, False])
def test_fp32_head_under_autocast(fp32):
    cfg = tiny_cfg(FP32_OUTPUT_HEAD=fp32, OUTPUT_NORMALIZED=False)
    m = MV.get_model(cfg).eval()
    x = synth_tokens(cfg, B=1)[:, :20]
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = m(x)
    assert out.dtype == (torch.float32 if fp32 else torch.bfloat16)


@pytest.mark.parametrize("tok", ["token", "frame"])
def test_scripted_twin_still_exports_and_matches(tok):
    # save_scripted_model() falls back to torch.jit.trace (scripting these
    # models already failed before 2026-11-05); the export must still
    # round-trip and reproduce the eager output with the new head.
    cfg = tiny_cfg(TOKENIZATION=tok, NUM_TIME=8, SEQ_LEN=80)
    m = with_stats(MV.get_model(cfg), cfg).eval()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m_scripted.pt")
        res = T.save_scripted_model(path, m, cfg=cfg, log=lambda *_: None)
        assert res["roundtrip_ok"], res
        loaded = torch.jit.load(path)
        x = synth_tokens(cfg, B=1)
        if tok == "frame":
            x = MV.seq_to_frames(x, cfg.NUM_X, cfg.LATENT_DIM)
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), m(x), rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------- fix 3 ----
def test_stratified_rollout_indices_balanced_and_deterministic():
    cfg = tiny_cfg()
    params = [5.6] * 10 + [10.0] * 3 + [17.8] * 7
    val = torch.zeros(len(params), 2, cfg.INPUT_DIM)
    for i, p in enumerate(params):
        val[i, :, cfg.LATENT_DIM + 4] = p
    idx, labels = T.stratified_rollout_indices(val, cfg, 6)
    assert sorted(labels) == sorted(["3p6"] * 2 + ["6p4"] * 2 + ["11p4"] * 2)
    idx2, _ = T.stratified_rollout_indices(val, cfg, 6)
    assert torch.equal(idx, idx2)
    assert T.param_label(12.2) == "7p8" and T.param_label(3.3) == "U3.3"


@pytest.mark.skipif(not HAVE_DECODER, reason="needs the frozen GEN3 decoder in assets/")
def test_evaluate_reports_per_param_and_ridge():
    cfg = tiny_cfg(NUM_TIME=14, SEQ_LEN=140, VAL_CONTEXT_STEPS=12, VAL_ROLLOUT_SEQS=3)
    m = with_stats(MV.get_model(cfg), cfg).eval()
    val = synth_tokens(cfg, B=3)
    idx, labels = T.stratified_rollout_indices(val, cfg, 3)
    D = cfg.NUM_X * cfg.LATENT_DIM
    ridge = torch.cat([torch.eye(D), torch.zeros(1, D)], 0)    # == persistence
    out = T.evaluate(m, val, cfg, "cpu", roll_idx=idx, roll_labels=labels,
                     ridge_A=ridge, latent_std=m.feat_std[:cfg.LATENT_DIM])
    for lab in ("3p6", "6p4", "11p4"):
        assert f"improvement_pct/{lab}" in out
        assert abs(out[f"ridge_improvement_pct/{lab}"]) < 1e-6   # identity ridge == persistence
    assert "improvement_pct_param_mean" in out and "val_tf_latent_nmse" in out


@pytest.mark.skipif(not HAVE_DECODER, reason="needs the frozen GEN3 decoder in assets/")
def test_frame_ar_loss_includes_latent_term():
    cfg = tiny_cfg(NUM_TIME=8, SEQ_LEN=80, AR_SEQS=2, AR_FRAMES=2, LATENT_LOSS_WEIGHT=1.0)
    m = with_stats(MV.get_model(cfg), cfg)
    batch = synth_tokens(cfg, B=2)
    std = m.feat_std[:cfg.LATENT_DIM]
    a = T.frame_ar_loss(m, batch, cfg, generator=torch.Generator().manual_seed(1))
    b = T.frame_ar_loss(m, batch, cfg, generator=torch.Generator().manual_seed(1),
                        latent_std=std)
    assert float(b) > float(a)


# --------------------------------------------------------------- fix 2 ----
def test_save_checkpoint_stores_ema_weights_when_promoted_on_ema():
    cfg = tiny_cfg()
    live = MV.get_model(cfg)
    ema = copy.deepcopy(live)
    with torch.no_grad():
        for p in ema.parameters():
            p.add_(1.0)
    opt = torch.optim.AdamW(live.parameters())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x_rollout_best.pt")
        T.save_checkpoint(path, live, opt, 5, {}, save_scripted=False,
                          log=lambda *_: None, ema_model=ema, weights_from_ema=True)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["weights_source"] == "ema"
        k = "output_head.weight"
        assert torch.equal(ck["model_state_dict"][k], ema.state_dict()[k])
        assert torch.equal(ck["raw_model_state_dict"][k], live.state_dict()[k])

        path2 = os.path.join(d, "x_latest.pt")
        T.save_checkpoint(path2, live, opt, 5, {}, save_scripted=False,
                          log=lambda *_: None, ema_model=ema)
        ck2 = torch.load(path2, map_location="cpu", weights_only=False)
        assert ck2["weights_source"] == "live"
        assert torch.equal(ck2["model_state_dict"][k], live.state_dict()[k])
        assert torch.equal(ck2["ema_state_dict"][k], ema.state_dict()[k])


# --------------------------------------------------------------- warm start
def test_trainer_defaults_to_cold_start():
    args = T.build_parser().parse_args([])
    assert args.warm_start is None
