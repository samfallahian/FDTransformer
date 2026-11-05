# assets/

Large binary inputs the trainer needs but that are not produced by this repo.
`*.pt` files in this folder are git-ignored, so copy them in by hand on each machine.

| File | Needed for | Where it came from |
|---|---|---|
| `Model_GEN3_05_AttentionSE_absolute_best_scripted.pt` | **Always.** It's the frozen GEN3 AttentionSE decoder. The training loss and the `improvement_pct` metric are both computed on its decoded centroid velocity. | Used to be loaded from outside the repo at `../encoder/autoencoderGEN3/saved_models_production/`. A copy is in git on branch `LFM`: `git show LFM:encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt > assets/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt` |
| `r2_h11_ridge_distill_rollout_best.pt` | **Optional.** Only used with `./run_train.sh --warm-start` (no PATH). Cold start is the default since 2026-11-05. This checkpoint predates the latent-consistency loss, so its outputs are off the latent manifold; see `docs/CHANGES_2026-11-05.md` §3. | Branch `LFM`: `git show LFM:transformer_neurIPS/saved_models/r2_h11_ridge_distill_rollout_best.pt > assets/r2_h11_ridge_distill_rollout_best.pt` |

You can override either path with `run_train.sh --decoder PATH` / `--warm-start PATH`. The trainer itself also reads the `PFD_DECODER_PATH` env var.
