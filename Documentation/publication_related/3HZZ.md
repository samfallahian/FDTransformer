We thank the reviewer for a thorough and constructive review. We completed two substantive corrections: the membership audit identified stage-specific holdouts and inaccurate statements in Sections 3.3/3.5, while the corrected cross-source enstrophy MSE is $3.89\times10^{-10}$. These findings are reported below; retraining, new baselines, ablations, and manuscript updates will be completed before final publication.

## Q1. How are train, validation, and test splits constructed to avoid spatial or temporal overlap?

Two distinct protocols should have been separated more clearly.

- **Autoencoder split.** The autoencoder training and validation cubes were sampled separately, without enforcing disjointness, by time index and seed coordinate from ten source experiments. The untrimmed grid and complete-cube centroid counts used for this estimate give
  $$
    N\_{\mathrm{full}}=26\times38\times12\times1200\times10=142{,}272{,}000,
  $$
  $$
    N\_{\mathrm{valid}}=(26-4)(38-4)(12-4)\times1200\times10=71{,}808{,}000.
  $$
  Let $T$ and $V$ denote independently and uniformly sampled training and validation sets, with $n\_T=n\_V=2{,}000{,}000$. If each set contains distinct draws, the expected exact intersection is
  $$
    \mathbb{E}\lvert T\cap V\rvert=\frac{n\_Tn\_V}{N},\qquad
    \frac{\mathbb{E}\lvert T\cap V\rvert}{n\_V}=\frac{n\_T}{N}.
  $$
  If sampling is performed with replacement, the probability that a validation draw appears at least once in training is $1-(1-1/N)^{n\_T}\approx n\_T/N$, producing nearly the same estimates:

| Pool definition | $N$ | $\mathbb{E}\lvert T\cap V\rvert$ | % of $V$ | % of $N$ |
| --- | --- | --- | --- | --- |
| Untrimmed positions (10 experiments) | 142,272,000 | 28,115 | 1.406% | 0.0198% |
| Complete-cube centroids (10 experiments) | 71,808,000 | 55,704 | 2.785% | 0.0776% |

  Thus, exact collisions are 0.02–0.08% of the source pool but 1.4–2.8% of validation rows. Neighboring overlapping cubes remain distinct samples, and that overlap is intentional. Autoencoder validation therefore measures same-distribution reconstruction and model selection, not forecasting or unseen-regime generalization. During retraining, we will partition eligible experiment–time–centroid indices before sampling so training and validation rows are exactly disjoint while retaining neighborhood overlap.
- **Forecasting membership.** The fitted stages did not use one uniform experiment-level split. The $U^{\ast}=6.9$ case was excluded from the autoencoder but included in both transformer training pools, whereas $U^{\ast}=10$ was included in the autoencoder but excluded from both transformer training pools. Consequently, each case was held out at only one stage, and neither the short-horizon $U^{\ast}=6.9$ evaluation nor the 80-step $U^{\ast}=10$ reversal benchmark is a full-pipeline holdout. The precise membership and implications are summarized below.
## Q2. Was the $U^{\ast}=10$ case used in training for the 80-step benchmark?

No. The $U^{\ast}=10$ case was excluded from the 80-step transformer training pool, but it was included in the autoencoder source pool. The reversal benchmark therefore tests transfer of the temporal dynamics model to $U^{\ast}=10$, but not generalization of the complete fitted pipeline. The complete membership relevant to these two cases is:

| Fitted component | $U^{\ast}=6.9$ | $U^{\ast}=10.0$ |
| --- | --- | --- |
| Autoencoder | Excluded | Included |
| 8-step transformer | Included | Excluded |
| 80-step transformer | Included | Excluded |
| Vortex-reversal testing | Not used | Primary benchmark |
| Evaluation type | Representation transfer | Dynamics transfer |

The cases test different stages. For $U^{\ast}=6.9$, the autoencoder was untrained on the regime but both transformers saw its sequences, so this case tests representation transfer with familiar dynamics. For $U^{\ast}=10$, the autoencoder saw the regime but neither transformer did, so the reversal benchmark tests dynamics transfer with a familiar representation. Neither is an end-to-end holdout.

This audit also corrects Sections 3.3 and 3.5 of the manuscript: both describe $U^{\ast}=6.9$ as held out, and Section 3.5 specifically states that it was excluded from transformer training. Those statements are inaccurate and will be corrected before final publication.

Before final publication, we will retrain a shared autoencoder using this disjoint split with both $U^{\ast}=6.9$ and $U^{\ast}=10$ excluded. We will then rerun the 8-step study with $U^{\ast}=6.9$ excluded from transformer training and the 80-step study with $U^{\ast}=10$ excluded. This provides an end-to-end holdout for each reported benchmark. Until those reruns are complete, the current results will be labeled as stage-specific rather than full-pipeline holdouts.

## Q3. Stronger learned forecasting baselines

We agree that the current PCA/POD comparison primarily evaluates compression and is not sufficient to establish forecasting competitiveness. The most targeted controls are:

1. persistence, which is a strong baseline at 8.33 ms per frame;
2. the same transformer operating on local PCA/POD latents, isolating the nonlinear autoencoder contribution; and
3. a budget-matched transformer operating directly on the cube features, isolating the compression step.

These controls address the representation claim more directly than rushed external architectures. Broader FNO, ConvLSTM/TCN, graph, and neural-ROM comparisons not completed during discussion will be evaluated before final publication.

## Q4. Is PCA/POD global or local?

The PCA/POD comparison is applied to the same flattened $5\times5\times5$ velocity cubes, each represented by 375 velocity values. It is therefore a matched local linear-versus-nonlinear comparison at the same latent dimensionality, not a full-domain global POD. We will state this explicitly and remove wording that describes this specific baseline as global.

## Q5. Same-time spatial context during autoregressive rollout

The reviewer is correct that this distinction changes how the headline numbers should be interpreted. Tokens are ordered as
$$
[T\_1,X\_1]\rightarrow\cdots\rightarrow[T\_1,X\_{26}]\rightarrow[T\_2,X\_1]\rightarrow\cdots\rightarrow[T\_8,X\_{26}],
$$
so a token may attend to earlier $x$ positions in the same time slice. Since adjacent cubes overlap heavily, the setting with all preceding same-slice tokens supplied is closer to spatially assisted next-token prediction than to forecasting an entire future field from past fields alone.

The staircase evaluation already exposes the distinction. With the seven preceding $T\_8$ tokens supplied, velocity RMSE is $1.2181\times10^{-3}$. When only the $T\_1$ context is available and the intervening tokens must be generated autoregressively, RMSE rises to $1.1305\times10^{-2}$. We will label these as separate regimes and lead the forecasting discussion with the fully generated result.

## Q6. Sensitivity to cube size, latent dimension, overlap, and auxiliary features

The current appendix contains a latent-dimension comparison and a spatio-temporal context/density analysis, but it does not vary cube size or overlap and does not consolidate feature ablations. We agree that the operating point is insufficiently justified. A complete sensitivity section should include cube size, overlap fraction, latent dimension, and removal of coordinate, relative-time, and Reynolds-proxy channels. We will report only experiments completed by the response deadline; remaining ablations will be completed before final publication rather than presented now as finished results.

## Q7. Enstrophy-consistency error near machine precision

The reviewer was correct: the original metric measured numerical self-consistency rather than forecast fidelity. The target enstrophy was calculated from the same predicted vorticity components supplied to the SINDy fit, so the reported $10^{-34}$ values verified the algebraic identity
$$
\mathcal{E}=\tfrac12\left(\omega\_x^2+\omega\_y^2+\omega\_z^2\right)
$$
within one data source. They did not compare the prediction against independent ground truth.

We corrected the evaluation to use a cross-source comparison: predicted vorticity is evaluated against enstrophy calculated independently from the raw ground-truth field. The corrected physical-validation MSE is $3.89\times10^{-10}$. For diagnosis only, predicted-versus-predicted and raw-versus-raw self-consistency remain near $2.73\times10^{-39}$ and $6.91\times10^{-35}$, respectively. Before final publication, we will replace the original table, remove the “near machine precision” forecast-fidelity claim, and describe the self-consistency values only as implementation checks.

## Physics is diagnosed but not enforced

We view the data-driven formulation as a complementary strength for noisy experimental measurements: it does not require exact boundary conditions or impose potentially mismatched governing-equation residuals. This flexibility is not evidence of superior physical fidelity, however, and it provides no formal conservation or stability guarantees. Physics is therefore diagnosed rather than enforced. In particular, lower predicted divergence must be compared with the divergence of the measured ground-truth field to distinguish improved consistency from over-smoothing. Physics diagnostics will be normalized and reported alongside that measurement floor.

We appreciate that the review was specific enough to make these corrections actionable.