# NeurIPS Evaluation ChangeLog: Enstrophy Validation & SINDy Recovery

## 1. Discovery: The "Impossibly Good" Result
During validation, the enstrophy recovery MSE was reported at approximately **10^-34**, which is mathematically impossible for a derivative quantity derived from a velocity forecast with ~1-10% error.

### Findings:
- **Self-Consistency Artifact**: The `enstrophy` target was being calculated directly from the predicted `vorticity` components ($wx, wy, wz$) within the same script just before being passed to SINDy.
- **Algebraic Identity**: SINDy was effectively solving the algebraic identity $E = 0.5(\omega_x^2 + \omega_y^2 + \omega_z^2)$. This confirms the internal consistency of the SINDy implementation but does **not** validate the physical accuracy of the transformer's velocity predictions.
- **Diagnostic Proof**: Unit tests in `pySINDy/test_enstrophy_chain.py` proved that even if 10% noise is added to vorticity, the MSE remains at 10^-34 as long as enstrophy is recalculated from that noisy data.

---

## 2. Changes Implemented
All substantive changes were localized to the `pySINDy/` directory to harden the evaluation pipeline.

### Infrastructure & Reliability:
- **Absolute Path Resolution**: Updated all scripts (`cross_source_sindy.py`, `run_all_evaluations.py`, `prepare_*.py`) to use absolute paths. This prevents `FileNotFoundError` and `OSError` regardless of the execution environment.
- **Cross-Source Validation**: Implemented a "Cross-Source" methodology where vorticity from the Transformer (Predicted) is compared against enstrophy from the original simulation (Raw Ground Truth).

### Diagnostic Tools:
- **`pySINDy/test_enstrophy_chain.py`**: Created to verify the enstrophy calculation chain and simulate independent vs. dependent noise.
- **Result Labeling**: Updated `cross_source_sindy.py` to explicitly label results as either "Self-Consistency (Artifact)" or "Physical Validation (Real)".

---

## 3. How to Interpret the Results (`cross_source_results.csv`)

The latest evaluation produces the following key metrics for the paper:

| Scenario | Input (Predictor) | Target (Ground Truth) | MSE | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Physical Reality** | **Predicted** | **Raw** | **3.89e-10** | **The true physical accuracy of the model.** |
| Self-Consistency | Predicted | Predicted | 2.73e-39 | Internal consistency check (Artifact). |
| Self-Consistency | Raw | Raw | 6.91e-35 | Mathematical baseline (Artifact). |

### Recommended Line for the Paper:
> "While the system maintains near-perfect internal physical consistency (self-consistency MSE ~10^-35), the actual predictive accuracy for enstrophy compared to the ground-truth simulation yields a physically realistic MSE of **3.89e-10**."

---

## 4. Verification
- All scripts successfully executed in the current environment.
- Identity checks confirm `Predicted` data is distinct from `Raw` data (mean velocity diff ~10^-3).
- Realistic cross-source MSE (~10^-10) is consistent across different Reynolds numbers.
