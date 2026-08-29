## Additional completed persistence evaluation

After rerunning the pipeline under the revised exclusion protocol, including retraining the shared autoencoder, we completed a persistence comparison using the resulting 4.78 × 10⁶-parameter autoregressive checkpoint. A persistence forecast copies the final observed velocity field across every future horizon. The evaluation used all 165 available 40-frame validation sequences: 12 context frames (100 ms) followed by 28 autoregressive forecast frames (233.3 ms). The source pool contained 150 wake-targeted and 15 random-location sequences, and the agreed U* = 10 exclusion was preserved. Confidence intervals below use 10,000 bootstrap resamples over sequences. We report RMSE as the primary metric because it matches the manuscript's forecasting evaluation.

| Horizon | Model RMSE | Persistence RMSE | Persistence minus model, 95% CI | Sequences beating persistence |
| --- | --- | --- | --- | --- |
| 8.3 ms | 4.81 × 10⁻⁴ | 5.40 × 10⁻⁴ | 5.82 × 10⁻⁵ [4.64 × 10⁻⁵, 6.98 × 10⁻⁵] | 79.4% |
| 50 ms | 9.83 × 10⁻⁴ | 1.173 × 10⁻³ | 1.90 × 10⁻⁴ [1.56 × 10⁻⁴, 2.23 × 10⁻⁴] | 77.6% |
| 100 ms | 1.188 × 10⁻³ | 1.593 × 10⁻³ | 4.04 × 10⁻⁴ [3.45 × 10⁻⁴, 4.68 × 10⁻⁴] | 86.7% |
| 150 ms | 1.361 × 10⁻³ | 1.923 × 10⁻³ | 5.62 × 10⁻⁴ [4.69 × 10⁻⁴, 6.55 × 10⁻⁴] | 82.4% |
| 200 ms | 1.474 × 10⁻³ | 2.209 × 10⁻³ | 7.34 × 10⁻⁴ [6.16 × 10⁻⁴, 8.54 × 10⁻⁴] | 81.8% |
| 233.3 ms | 1.534 × 10⁻³ | 2.343 × 10⁻³ | 8.10 × 10⁻⁴ [6.80 × 10⁻⁴, 9.40 × 10⁻⁴] | 78.2% |

The model outperforms persistence at every reported horizon, with RMSE improvement increasing from 10.8% at 8.3 ms to 34.6% at 233.3 ms. At the final horizon, improvement is 34.3% on wake-targeted sequences and 37.2% on random-location sequences; the mean-difference confidence intervals remain positive for both strata. Final-horizon RMSE also improves separately for all velocity components: 36.1% for vx, 26.3% for vy, and 39.3% for vz. MAE and mean vector L2 provide independent confirmation, with final-horizon improvements of 32.3% and 32.2%, respectively.

Persistence RMSE increases approximately 4.3-fold from the first to the final horizon. Thus, these evaluated trajectories are not adequately described by copying a static or weakly changing field, and the forecasting gain is not confined to selected wake locations or one velocity component. This is an operational anti-persistence result, not a formal test of statistical stationarity, and it does not replace the planned PCA-latent, raw-cube, neural-operator, or energy-spectrum comparisons.