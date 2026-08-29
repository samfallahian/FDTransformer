---
title: "Learning Latent Spatio-Temporal Dynamics in Experimental Fluid Flows Using Transformers"
authors:
  - Kyle Kreth — University of North Carolina at Charlotte
  - Hesam Fallahian — University of North Carolina at Charlotte
  - Seyedmohammad Mousavisani — Northeastern University
  - Mostafa Khazaee Kuhpar — University of Massachusetts Dartmouth
  - Banafsheh Seyed-Aghazadeh — University of Massachusetts Dartmouth
source: ../main.tex
---

# Learning Latent Spatio-Temporal Dynamics in Experimental Fluid Flows Using Transformers

**Kyle Kreth**, University of North Carolina at Charlotte  
**Hesam Fallahian**, University of North Carolina at Charlotte  
**Seyedmohammad Mousavisani**, Northeastern University  
**Mostafa Khazaee Kuhpar**, University of Massachusetts Dartmouth  
**Banafsheh Seyed-Aghazadeh**, University of Massachusetts Dartmouth

## Abstract

We present a transformer-based model for forecasting transport velocity in experimentally measured, three-dimensional, three-component (3D-3C) wake flows behind a rigid circular cylinder undergoing vortex-induced vibration, acquired by time-resolved volumetric particle tracking velocimetry. Rather than compressing global flow fields, we adopt a local-global decomposition: time-resolved data are reorganized into overlapping $5\times5\times5$ velocity cubes so that a residual autoencoder first learns a nonlinear local basis that preserves physically meaningful neighborhood structure. The resulting 47-dimensional latent tokens (87.5% compression) are modeled with a causal Transformer augmented by spatial coordinates, relative time indices, and a Reynolds-number proxy. Across a multi-experiment dataset, the model yields stable autoregressive rollouts, preserves dominant spatial structure on held-out data, and improves divergence and vorticity preservation relative to PCA/POD baselines at comparable dimensionality. In a retrained 80-step benchmark validated on case $U^*$ = 10, 12 steps of context (100 ms) are sufficient to recover vortex-reversal direction across 18 events over a 667 ms window, with mean recovery time of $\sim$28.5 ms. Staircase physics evaluation further shows that as temporal jump increases, velocity RMSE rises while divergence RMSE falls, suggesting that under greater uncertainty the model sacrifices fine-scale pointwise accuracy while retaining smoother, more globally consistent predictions. Overall, the results support this local-global decomposition: neighborhood-scale flow structure is preserved at the token level while attention models longer-range wake dynamics.

# Introduction

Forecasting experimentally measured wake flows is challenging because the state is high-dimensional, nonlinear, and observed through finite volumetric samples. Useful predictors must therefore preserve local rotational structure while remaining stable under autoregressive rollout. We study this problem for three-dimensional, three-component wake flows behind a rigid circular cylinder undergoing vortex-induced vibration, using time-resolved volumetric particle tracking velocimetry to build a fully data-driven forecasting pipeline.

Rather than modeling isolated velocity samples, we represent each retained location by an overlapping local $5\times5\times5$ velocity cube, so each token carries neighborhood information about shear, rotation, and mean transport structure before compression. This local-global decomposition is deliberate: a residual autoencoder maps these 375-dimensional cubes to compact latent codes, and a causal transformer then forecasts wake dynamics through attention over those physically grounded local tokens.

Standard reduced-order models apply compression globally, which can dilute local gradient structure into domain-wide modes. By contrast, our nonlinear local basis preserves physically meaningful neighborhoods before any global reasoning occurs, and the substantial overlap between adjacent cubes provides implicit spatial continuity by construction.

Unlike approaches that impose governing-equation residuals during training, our model learns directly from measured velocity neighborhoods and is assessed with both prediction error and physics-aware diagnostics. [17](#ref-raissi2019pinn), [12](#ref-karniadakis2021piml) We therefore focus on whether this local-global representation preserves coherent vortical organization and stable short-horizon evolution on held-out experiments, with the vorticity reconstruction in Section 3.4 providing the clearest qualitative test.

# Methodology

## Data generation and pre-processing

### Experimental setup and data acquisition

Flow velocity data were generated at an academic fluid-structure-interaction laboratory. Quantitative measurements were acquired with Time-Resolved Volumetric Particle Tracking Velocimetry (TR-PTV) in the wake of a rigid circular cylinder undergoing vortex-induced vibration, then processed in DaVis 10 with Shake The Box to recover time-resolved, three-dimensional, three-component (3D-3C) Lagrangian velocities $\langle v_x, v_y, v_z \rangle$. The evaluated cases are summarized in Table 1 and characterized by freestream velocity, Reynolds number $Re = \frac{U D}{\nu}$, and reduced velocity $U^* = \frac{U}{f_n D}$ with $f_n = 0.48\,\mathrm{Hz}$; the full acquisition hardware specification is deferred to the technical appendix.

<a id="tab:flow-speeds"></a>
| Flow Speed (m/s) | Reynolds Number $Re$ | Reduced Velocity $U^*$ |
| --- | --- | --- |
| 5.832e-02 | 1.26e+03 | 5.6 |
| **7.128e-02*** | 1.54e+03 | 6.9 |
| 7.452e-02 | 1.61e+03 | 7.2 |
| 8.424e-02 | 1.82e+03 | 8.1 |
| 1.036e-01 | 2.24e+03 | 10.0 |
| 1.069e-01 | 2.31e+03 | 10.3 |
| 1.116e-01 | 2.52e+03 | 11.3 |
| 1.263e-01 | 2.73e+03 | 12.2 |
| 1.361e-01 | 2.94e+03 | 13.1 |
| 1.684e-01 | 3.64e+03 | 16.3 |
| 1.846e-01 | 3.99e+03 | 17.8 |

**Table:** Flow speeds for experiments tested. **\* indicates a hold-out sample.**

### Data preparation and quality assurance

Each file was verified to contain a complete 1200-frame sequence at 120 fps with consistent row counts. Coordinates and time indices were stored as 32-bit integers, velocity components as 32-bit floats, and all velocities were linearly normalized to $[0,1]$ using corpus-wide extrema (approximately $-0.197745$ to $0.263599\,\mathrm{m/s}$). Additional auditing and preprocessing details are provided in the technical appendix. An anonymized public release of the source code and the training, validation, evaluation, and original evaluation datasets is provided in the submission PDF under NeurIPS checklist item 5.

### Spatial mapping and cubic reconstruction

The measurement domain was trimmed near its boundaries so that each retained centroid had a complete $5\times5\times5$ neighborhood. For every valid centroid, the 3D-3C velocities at the 125 local grid points were gathered and flattened into a 375-dimensional feature vector $V \in \mathbb{R}^{375}$ for autoencoder input (Figure 1); detailed boundary-handling and vectorization notes are deferred to the technical appendix.

<a id="fig:velocity-cube"></a>
[Local $5\times5\times5$ volumetric sampling neighborhood used to form velocity cubes. Top row: three orthogonal cross-sections through the cube center — $xy$ (blue, $z=0$), $xz$ (teal, $y=0$), and $yz$ (amber, $x=0$) — each containing 25 of the 125 measurement points. Bottom: full 3D projection showing all 125 points; colored points lie on one or more cross-section planes; gray points are off-plane. The red marker indicates the target centroid at $(0,0,0)$. Each vectorization of a centroid contains a 375-dimensional feature vector ($125$ points $\times$ 3 velocity components $v_x$, $v_y$, $v_z$).](../auto-encoder/LFM_Figure2.pdf)

**Figure:** Local $5\times5\times5$ volumetric sampling neighborhood used to form velocity cubes. Top row: three orthogonal cross-sections through the cube center — $xy$ (blue, $z=0$), $xz$ (teal, $y=0$), and $yz$ (amber, $x=0$) — each containing 25 of the 125 measurement points. Bottom: full 3D projection showing all 125 points; colored points lie on one or more cross-section planes; gray points are off-plane. The red marker indicates the target centroid at $(0,0,0)$. Each vectorization of a centroid contains a 375-dimensional feature vector ($125$ points $\times$ 3 velocity components $v_x$, $v_y$, $v_z$).

## Residual autoencoder with squeeze-and-excitation

### Architecture overview

We use an AttentionSE autoencoder selected from a broader architectural benchmark reported in the technical appendix.[4](#ref-he2016resnet), [6](#ref-hu2018senet) The model maps each 375-float velocity cube to a 47-dimensional latent code (87.5% compression) through fully connected encoder/decoder stages with SEResidualBlocks, ELU activations, a Tanh-bounded bottleneck, and a linear output layer.

### Training and data sampling

The source pool comprises 10 experiments excluding the hold-out case. Samples were drawn over random time indices, experiments, and seed coordinates to form 2,000,000 training rows and 2,000,000 validation rows, with each row containing the 375 floats defining one velocity cube. Optimization used Adam with RMSE reconstruction loss, a weak latent L2 penalty ($\lambda=0.00005$), and an initial learning rate of $1\times 10^{-4}$ that decayed to $1\times 10^{-5}$ once validation RMSE plateaued. Training was performed on a single NVIDIA H200 GPU for approximately 24 hours using mixed precision, with a batch size of 200,000 selected as the largest stable value that fit device memory while avoiding GPU underutilization.

### Performance metrics

We evaluate the latent representation through reconstruction RMSE, latent-dimension ablation, comparison to a linear POD/PCA baseline, and physical consistency via divergence and vorticity fidelity. The detailed ablations and AE-versus-PCA comparisons are deferred to the technical appendix.[15](#ref-murata2020nonlinear)

## Spatio-temporal transformer setup (WAFT)

### Dataset preparation and feature engineering

Latent codes are arranged as sequences of 8 time steps over 26 adjacent $x$-coordinates at fixed $(y, z)$, yielding an $8\times26$ grid of 52-dimensional tokens. Each token concatenates 47 latent features with $(x,y,z)$, a relative time index, an experiment parameter, and the grid is flattened from $[T_1,X_1]$ through $[T_8,X_{26}]$ by traversing all 26 spatial locations within each time step before advancing in time. A dedicated 80-step variant used for long-horizon vortex-reversal validation preserves the same representation while extending the sequence length; full windowing and retraining details are moved to the technical appendix.

This implements the paper's local-global architecture directly. The autoencoder first compresses overlapping local cubes into latent tokens that retain neighborhood-scale shear, rotation, and transport structure; the transformer then attends over these physically grounded tokens to model longer-range spatio-temporal relationships. Because adjacent cubes share 100 of 125 measurements, the tokenization also induces implicit spatial continuity in latent space rather than presenting the transformer with isolated point samples.

### Model architecture: WAFT (World-model for Autoregressive Fluid Transport)

***WAFT*** (World-model for Autoregressive Fluid Transport) adds learned time and space embeddings to a linear projection of each token, then applies a 6-block causal Transformer with 8 attention heads and embedding dimension of 256. A triangular mask enforces autoregressive dependence only on earlier tokens in the flattened sequence, thereby following the arrow of time.[19](#ref-vaswani2017attention), [16](#ref-radford2019language) The trained model contains approximately $4.7\times10^6$ parameters. We ran all experiments on a server equipped with dual AMD EPYC 9224 processors, 768 GiB of system memory, and one NVIDIA H200 NVL GPU with 143 GB of GPU memory. For transformer training, we used a batch size of 512, eight data-loader workers with pinned host memory enabled, and bf16 mixed precision to conserve GPU memory; the final transformer run required approximately 72 hours on this single H200. Together with the approximately 24-hour autoencoder run reported above, the final reported training runs correspond to roughly 96 single-GPU hours in total.

# Evaluation and results

Predicted latents were decoded through the AE decoder and denormalized to compute physical-velocity RMSE.

## Model performance

#### Error distribution.

Evaluation across the $y$–$z$ coordinate space indicates consistent accuracy across the experimental volume: across the ten evaluated reduced-velocity cases, velocity-space RMSE ranges from $1.4360\times10^{-3}$ at $U^*=5.6$ to $3.9328\times10^{-3}$ at $U^*=17.8$, with generally increasing error as flow speed rises; the full per-experiment RMSE table is provided in the technical appendix.

#### Autoregressive performance: staircase evaluation.

To assess stability for multi-step rollout, a “staircase” evaluation was used in which the model predicts one fixed target token in the $T_8$ slice under progressively reduced same-slice context. In the implementation's zero-based flattened indexing, the target token is sequence index 189, corresponding to $(T_8,X_8)$, while the immediately preceding available $T_8$ context occupies indices 182–188, corresponding to $(T_8,X_1)$ through $(T_8,X_7)$. When all seven preceding $T_8$ tokens are provided, the task is a single-token next-step prediction; when fewer are provided, the model autoregressively generates the missing intermediate $T_8$ tokens needed to reach that same fixed target token. This style of decreasing-context autoregressive rollout is commonly used to surface exposure bias and long-horizon error accumulation in sequence models, complementing training-time strategies such as scheduled sampling and Professor Forcing.[1](#ref-bengio2015scheduled), [13](#ref-lamb2016professor) These metrics are computed directly in the 47-dimensional latent space (MSE); because the latent features are normalized, the numerical values are smaller than the physical-velocity RMSE reported above.

Autoregressive prediction of the fixed $T_8$ target remains stable as same-slice context is reduced, with velocity-space RMSE increasing from $1.2181\times10^{-3}$ when $T_1$–$T_7$ are provided to $1.1305\times10^{-2}$ when only $T_1$ is available. Latent-space MSE rises only modestly over the same sweep, consistent with controlled long-horizon degradation; the full staircase evaluation table is provided in the technical appendix.

## Robustness analysis: data corruption study

A latent-space corruption sweep, in which a percentage of latent features was replaced with random noise in $[-1,1]$, shows that the model is highly sensitive to small perturbations but degrades more gradually once corruption is large. This two-regime response suggests that small latent perturbations can quickly move a tightly compressed state away from combinations seen during training, whereas under heavier corruption the untouched coordinate, time-index, and experiment-parameter channels still preserve enough coarse structure to slow further error growth. Detailed RMSE values and deterioration visualizations are provided in the technical appendix.

## Absolute-value reconstruction

To further assess physical fidelity beyond rollout error, we evaluated absolute predicted velocity fields against ground-truth measurements on the held-out case $U^*$ = 6.9. For this analysis, predictions were generated and reconstructed for horizons $T=2$ through $T=8$, decoded back to velocity space, and compared directly with the corresponding ground-truth fields. Error was summarized using component-wise mean absolute error (MAE) for $v_x$, $v_y$, and $v_z$.

Overall, the absolute-value comparison indicates stable reconstruction quality across short- and mid-range horizons. Across $T=2$ to $T=8$, the MAE remains on the order of $10^{-4}$ to $10^{-3}$ for all three velocity components, with $v_y$ and $v_z$ consistently showing lower absolute error than $v_x$. The lowest errors are observed at early horizons (notably around $T=2$), while a gradual increase is visible toward $T=8$, consistent with expected autoregressive error accumulation. Even at the longest horizon, however, the reconstructed fields retain close agreement with measured ground truth, indicating that the model preserves the dominant spatial structure and magnitude of the velocity field on held-out data.

Component-wise MAE remains uniformly low across $T_2$ through $T_8$, with $v_y$ and $v_z$ consistently below $v_x$ and only modest growth at longer horizons, reinforcing that absolute reconstructions remain stable over short-to-moderate rollouts. Representative planar and three-dimensional comparisons at $T_5$ likewise show close agreement between predicted and measured fields, preserving the dominant spatial structure and magnitude of the wake; the full table and figure are provided in the technical appendix.

## Coherent vortical structure validation

To evaluate whether predicted fields retain physically meaningful rotational structure, we compare spanwise-vorticity reconstructions from predicted and ground-truth velocities on the held-out case $U^*$ = 6.9. Figure 2 shows that the predicted planar slices and 3D isosurfaces preserve the dominant alternating vortical regions, their relative positioning, and the coherence of the streamwise cores, with discrepancies concentrated in fine-scale deformation near high-gradient boundaries. Because vorticity amplifies small velocity-gradient errors, this is the clearest main-paper evidence that the local-global architecture matters: preserving these derivative-sensitive structures suggests that the overlapping local tokens retain neighborhood-scale gradients and rotation before the transformer models longer-range wake dynamics. A fuller panel-by-panel interpretation is deferred to the technical appendix.

<a id="fig:isosurface-vorticity"></a>
[Two-dimensional instantaneous vorticity $w_z$ of predicted and ground-truth on $x$–$y$ at $z = 10$ and $z = -10$.](../More_Validation/2d.pdf)

[Three-dimensional instantaneous vorticity $w_z$ of predicted and ground-truth.](../More_Validation/3d.pdf)

**Figure:** Isosurface of vorticity reconstruction quality on held-out case $U^*$ = 6.9 at horizon $T_5$, shown in planar and 3D views.

## 80-step vortex-reversal validation

### Vortex-core identification and event set

The 80-step validation study uses a retrained version of ***WAFT*** to test whether the learned latent dynamics remain physically meaningful over a substantially longer horizon. As above, training excludes the held-out case $U^*$ = 6.9; the long-horizon benchmark is then validated in depth on case $U^*$ = 10. At 120 fps, an 80-step window corresponds to $\tfrac{80}{120}\,\mathrm{s}=0.667\,\mathrm{s}$ of temporal coverage; in this setting the model receives 12 steps (100 ms) of context and then autoregressively predicts the remaining 68 steps (566.7 ms) along the same 26-coordinate $x$-line. Candidate reversal times were identified from the $U^*$ = 10 trajectory using local $y$-extrema and then localized in the volumetric velocity snapshots by computing the full vorticity field together with the $Q$-criterion, a standard vortex-identification measure that marks regions where local rotation dominates strain; the formula used here is given in Equation 1 of the technical appendix.[10](#ref-jeong1995vortex), [8](#ref-hunt1988eddies)

For each timestep, we restricted the search to the interaction region $x \in [-30,30]$, identified the peak vorticity magnitude, and defined the vortex core as all grid points with $|\boldsymbol{\omega}| \geq 0.9\,|\boldsymbol{\omega}|_{\max}$. Event location was then reported as the vorticity-magnitude-weighted centroid of that core region, rather than as a single grid point. This region-based definition is the basis for the event set used in the long-horizon benchmark and ensures that the validation tracks physically extended rotational structures rather than isolated pointwise extrema.

### Spatio-temporal sensitivity sweep

To determine the minimum information required for accurate reversal prediction, we performed a sweep over both temporal context and spatial density. Temporal context $T$ ranged from 1 to 60 timesteps, spatial support $C$ ranged from 5 to 26 $x$-coordinates per timestep, and the evaluation metric was line-based $z$-vorticity, $\omega_z \approx \partial v_y/\partial x$, computed after decoding predicted latents back to velocity space. The sweep showed that prediction quality is controlled primarily by temporal context: the 12-step (100 ms) setting acts as an inflection point for reliable sign-flip prediction, while very short contexts tend to continue the incoming trend.

Spatial thinning still matters when the $x$-line is reduced aggressively, but once moderate spatial support is retained, the largest gains come from adding temporal context rather than adding more coordinates. This motivates the operating point used in the event-level study below, where the model is given a 12-step context together with the full 26-coordinate line before entering autoregressive rollout.
Representative sparse-evaluation panels are deferred to the technical appendix (Section 3.5.2); they show that a 12-step (100 ms) context with the full 26-coordinate line already captures the reversal window reliably, while longer context mainly sharpens the timing of the zero-crossing.

### Event-level recovery dynamics

Using recovery time defined as the interval from the end of the context window until the predicted sign matched the ground truth and the local RMSE fell below $10\%$ of the local peak vorticity, the retrained 80-step model correctly captured reversal direction in all 18 automatically localized events, with mean recovery time of $\sim$28.5 ms and range 17–42 ms. The full event-level validation summary table is provided in the technical appendix.

A representative step-292 reversal event shows that, despite a vertical offset, the predicted trace crosses the reversal window immediately after the 12-step context boundary and then follows the sign changes of the reference signal. This behavior highlights regime-level recovery rather than exact amplitude matching, and the full event-level figure with 95% confidence intervals is provided in the technical appendix.

<a id="fig:vortex-core-temporal-correlation"></a>
[Temporal autocorrelation of vortex-core vorticity magnitude across reversal events in case $U^*$ = 10. Boxplots summarize Pearson correlation at the identified $\geq 90%$ core points for timestep lags from $\pm1$ to $\pm50$.](../80_time_step_data/core_temporal_correlation.pdf)

**Figure:** Temporal autocorrelation of vortex-core vorticity magnitude across reversal events in case $U^*$ = 10. Boxplots summarize Pearson correlation at the identified $\geq 90%$ core points for timestep lags from $\pm1$ to $\pm50$.

The autocorrelation plot shows that the vortex core is not quasi-static across the forecast horizon. Each boxplot summarizes the event-level distribution of Pearson correlation values at a fixed lag, with the median shown by the center line and the interquartile spread shown by the box. Median Pearson correlation falls steadily from short lags to nearly zero by $\pm50$ steps, while the spread across events remains substantial at every lag. This decay provides an important physical backdrop for the 80-step benchmark: the model is not merely extrapolating a frozen structure, but instead tracking a core whose coherence materially degrades over the same horizon on which the forecast is evaluated.

## Staircase physics evaluation

To evaluate physics consistency under varying temporal gaps, we performed an interpolation staircase study across four transition groups: Jump 1 ($T_t\rightarrow T_{t+1}$), Jump 2 ($T_t\rightarrow T_{t+2}$), Jump 3 ($T_t\rightarrow T_{t+3}$), and Fixed $T_1$ context ($T_1\rightarrow T_j$ for later targets). For each transition, we computed direct velocity RMSE, divergence RMSE, and sparse-regression reconstruction errors for enstrophy.

For physics-aware evaluation, we compute the local enstrophy density, defined from the predicted velocity field $\mathbf{u}=(u,v,w)$ and vorticity $\boldsymbol{\omega}=\nabla\times\mathbf{u}$ as:

$$
\begin{aligned}
\Omega &= \tfrac{1}{2}\lVert\boldsymbol{\omega}\rVert^2 = \tfrac{1}{2}\left(\omega_x^2 + \omega_y^2 + \omega_z^2\right).
\end{aligned}
$$

Enstrophy provides a sensitive measure of local rotational intensity and is used here to assess whether predicted wake fields preserve coherent vortical structure and sharp gradient regions. Across the evaluated staircase transition families, mean direct velocity RMSE spans $6.299\times10^{-3}$ to $9.009\times10^{-3}$, while mean divergence RMSE spans $7.697\times10^{-4}$ to $1.462\times10^{-3}$, indicating a trade-off between pointwise predictive accuracy and global incompressibility consistency as temporal gap increases. Mean enstrophy-consistency error remains near machine precision throughout, ranging from $9.267\times10^{-35}$ to $5.224\times10^{-34}$; the full staircase interpolation tables are provided in the technical appendix.

# Conclusion

We present ***WAFT***, a latent forecasting pipeline that encodes overlapping local $5\times5\times5$ velocity neighborhoods into 47-dimensional latent tokens and models their evolution with a causal transformer on experimental wake-flow data. Because adjacent cubes share 100 of 125 measurements (about $80\%$), neighboring latent codes are correlated by construction, which creates an implicit spatial continuity constraint without explicitly enforcing one. Each token already encodes neighborhood-scale gradients and rotational structure, so attention can focus on longer-range inter-vortex and wake-scale dynamics rather than reconstructing local structure from scratch.

This local-first, multi-scale design is supported most directly by Section 3.4, where the predicted fields retain alternating vortical regions and streamwise-core coherence despite vorticity's sensitivity to local gradient error (Figure 2). The AE-versus-PCA comparisons in the technical appendix further show that, at comparable dimensionality, the local autoencoder preserves divergence and vorticity more faithfully than linear global baselines (Figure 5), consistent with a representation that does not sacrifice local gradient structure to fit a single global basis. The staircase results suggest the same mechanism during rollout: as temporal jump increases, velocity RMSE rises while divergence RMSE falls, indicating that once fine-scale pointwise evolution becomes uncertain, the model regresses toward smoother and more globally consistent fields with smaller gradient-induced divergence error. Together with stable short-to-moderate-horizon rollouts, event-level vortex-reversal prediction, and staircase enstrophy consistency (Table 8, Table 9), these results indicate that ***WAFT*** learns physically coherent short-horizon wake dynamics from data, while broader generalization to other geometries, regimes, and substantially longer horizons remains open.

# Limitations

This study is evaluated on experimentally measured wake flows from a rigid circular cylinder undergoing vortex-induced vibration, with held-out cases drawn from the same acquisition pipeline. As a result, the current evidence does not establish generalization to other geometries, sensing layouts, Reynolds regimes, or substantially longer forecast horizons. The method is also purely data-driven: although the reported diagnostics show favorable divergence and vorticity behavior, we do not provide formal stability guarantees or error bounds, and performance still depends on dataset coverage, preprocessing choices, and decoder fidelity.

# Broader impact

On the positive side, compact latent forecasting models could help experimental fluid-mechanics workflows by accelerating exploratory analysis, hypothesis screening, and reduced-order prediction from expensive volumetric measurements. On the negative side, overconfident use of such models in engineering design, monitoring, or control could encourage users to trust forecasts outside the measured operating regime; accordingly, these results should be treated as a research-stage modeling study rather than as a validated surrogate for safety-critical deployment.

## References

- <a id="ref-bengio2015scheduled"></a> **[1]** Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. *arXiv preprint arXiv:1506.03099*, 2015.
- <a id="ref-brunton2019koopman"></a> **[2]** Steven L. Brunton and J. Nathan Kutz. *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control*. Cambridge University Press, 2019.
- <a id="ref-fallahian2024gan"></a> **[3]** Mohammadali Fallahian, Mohsen Dorodchi, and Kyle Kreth. Gan-based tabular data generator for constructing synopsis in approximate query processing: Challenges and solutions. *Machine Learning and Knowledge Extraction*, 6(1):171–198, 2024.
- <a id="ref-he2016resnet"></a> **[4]** Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 770–778, 2016.
- <a id="ref-hendrycks2016gelu"></a> **[5]** Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). *arXiv preprint arXiv:1606.08415*, 2016.
- <a id="ref-hu2018senet"></a> **[6]** Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 7132–7141, 2018.
- <a id="ref-huang2017densenet"></a> **[7]** Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 4700–4708, 2017.
- <a id="ref-hunt1988eddies"></a> **[8]** J. C. R. Hunt, A. A. Wray, and P. Moin. Eddies, streams, and convergence zones in turbulent flows. Technical Report CTR-S88, Center for Turbulence Research, 1988.
- <a id="ref-ioffe2015batchnorm"></a> **[9]** Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *Proceedings of the 32nd International Conference on Machine Learning*, pages 448–456, 2015.
- <a id="ref-jeong1995vortex"></a> **[10]** Jinhee Jeong and Fazle Hussain. On the identification of a vortex. *Journal of Fluid Mechanics*, 285:69–94, 1995.
- <a id="ref-jumper2021alphafold"></a> **[11]** John Jumper, Richard Evans, Alexander Pritzel, et al. Highly accurate protein structure prediction with alphafold. *Nature*, 596:583–589, 2021.
- <a id="ref-karniadakis2021piml"></a> **[12]** George E Karniadakis, Ioannis G Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. *Nature Reviews Physics*, 3(6):422–440, 2021.
- <a id="ref-lamb2016professor"></a> **[13]** Alex M. Lamb, Anirudh Goyal, Ying Zhang, Saizheng Zhang, Aaron C. Courville, and Yoshua Bengio. Professor forcing: A new algorithm for training recurrent networks. *arXiv preprint arXiv:1610.09038*, 2016.
- <a id="ref-mezic2013koopman"></a> **[14]** Igor Mezić. *Analysis of Fluid Flows via Spectral Properties of the Koopman Operator*, volume 45 of *Annual Review of Fluid Mechanics*. Springer, 2013.
- <a id="ref-murata2020nonlinear"></a> **[15]** Takaaki Murata, Kai Fukami, and Koji Fukagata. Nonlinear mode decomposition with convolutional neural networks for fluid dynamics. *Journal of Fluid Mechanics*, 882:A13, 2020.
- <a id="ref-radford2019language"></a> **[16]** Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. *OpenAI Technical Report*, 2019.
- <a id="ref-raissi2019pinn"></a> **[17]** Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378:686–707, 2019.
- <a id="ref-ronneberger2015unet"></a> **[18]** Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In *Medical Image Computing and Computer-Assisted Intervention*, pages 234–241, 2015.
- <a id="ref-vaswani2017attention"></a> **[19]** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances in Neural Information Processing Systems*, pages 5998–6008, 2017.
- <a id="ref-yoon2018gain"></a> **[20]** Jinsung Yoon, James Jordon, and Mihaela van der Schaar. Gain: Missing data imputation using generative adversarial nets. In *Proceedings of the 35th International Conference on Machine Learning*, volume 80 of *Proceedings of Machine Learning Research*, pages 5689–5698. PMLR, 2018.
- <a id="ref-zagoruyko2016wide"></a> **[21]** Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. *arXiv preprint arXiv:1605.07146*, 2016.

# Technical Appendix

The appendix below is included in the submission PDF, and its numbering mirrors the corresponding sections of the main manuscript.

# Methodology

## Data generation and pre-processing

### Experimental setup and data acquisition

This appendix subsection expands the concise description from Section 2.1.1 of the main manuscript. Flow measurements were acquired at an academic fluid-structure-interaction laboratory using a LaVision TR-PTV setup on the wake of a rigid circular cylinder undergoing vortex-induced vibration. The acquisition hardware used a $300 \times 100\,\mathrm{mm}^2$ LED array (FLASHLIGHT 300), a four-camera Minishaker box equipped with 8, 12, and 16 mm lenses, synchronized triggering through a LaVision programmable timing unit, and Sony IMX252LLR/LQR CMOS cameras recording at 2048$\times$1088 resolution through a Core DVR. Raw image sequences were processed in DaVis 10 with Shake The Box to recover time-resolved 3D-3C Lagrangian particle velocities $\langle v_x,v_y,v_z \rangle$.

### Data preparation and quality assurance

Each file was audited to confirm a complete temporal sequence $\mathbf{T}=\{1,\ldots,1200\}$ and consistent row counts. Data were acquired at 120 fps, corresponding to 1200 time steps over 10 seconds with an 8.33 ms interval between frames. Global extrema of $v_x$, $v_y$, and $v_z$ were computed once over the corpus and used to linearly normalize all velocity components to $[0,1]$ (approximately $-0.197745$ to $0.263599\,\mathrm{m/s}$). For storage efficiency, spatial and temporal coordinates were cast to 32-bit integers and velocities to 32-bit floats before model training.

### Spatial mapping and cubic reconstruction

The measurement domain was trimmed near its boundaries so that every retained centroid supported a complete local neighborhood. For each valid centroid, a $5\times5\times5$ coordinate stencil was formed, the 3D-3C velocities at all 125 grid points were gathered, and the result was flattened into a feature vector $V\in\mathbb{R}^{375}$ (125 points $\times$ 3 velocity components). This vectorization step produces the fixed-width local state used by the autoencoder while preserving the underlying cube geometry of the local-neighborhood construction.

## Residual autoencoder with squeeze-and-excitation

### Architecture overview

This appendix subsection expands the brief model-selection summary from Section 2.2.1 of the main manuscript by collecting the full comparative auto-encoder benchmark table and accompanying model-selection note. Although the Baseline model achieved the lowest relative MSE in the initial comparative evaluation, extended training of the two top-performing candidates led to the selection of AttentionSE as the final model used in the remainder of the paper.

<a id="tab:autoencoder-summary"></a>
| # | Model | Basis / Paper | Rel. MSE | Key Feature | Best For |
| --- | --- | --- | --- | --- | --- |
| **1** | **Baseline** | **ResNet [4](#ref-he2016resnet)** | **1.334e-03** | **Residual blocks (LN + ELU)** | **General reconstruction** |
| 2 | Deep | ResNet [4](#ref-he2016resnet) | 3.190e-03 | More residual blocks per stage | Hierarchical matching |
| 3 | Wide | Wide ResNet [21](#ref-zagoruyko2016wide) | 4.207e-03 | Increased width (512-256-128) | High-bandwidth features |
| 4 | GELU | GELU [5](#ref-hendrycks2016gelu) | 1.887e-03 | GELU activations | Turbulent flow stochasticity |
| **5** | **AttentionSE** | **SENet [6](#ref-hu2018senet)** | **1.359e-03** | **SE gating** | **Channel-wise weighting** |
| 6 | Dense | DenseNet [7](#ref-huang2017densenet) | 2.675e-03 | Concatenated feature flow | Gradient preservation |
| 7 | BatchNorm | BatchNorm [9](#ref-ioffe2015batchnorm) | 4.290e-03 | Batch normalization | Large-batch stability |
| 8 | Skip/U-Net | U-Net [18](#ref-ronneberger2015unet) | 1.423e-03 | Encoder–decoder skips | High-res preservation |
| 9 | Bottleneck | ResNet bottleneck [4](#ref-he2016resnet) | 2.576e-03 | Reduce–transform–expand | Efficient compression |
| 10 | Gated Attention | Transformer [19](#ref-vaswani2017attention) | 1.746e-03 | Self-attention gating | Global relationships |

**Table:** Auto-encoder results summary with primary architectural references.

### Performance metrics

This technical appendix expands the performance-metric discussion from Section 2.2.3 of the main manuscript by providing the detailed PCA ablation and AE-versus-PCA physical-fidelity comparisons. The goal is to preserve the same logical numbering as the main paper while moving the lower-priority visual and implementation-level detail out of the core submission.

An ablation study (Figure 4) compares reconstruction RMSE across PCA latent dimensions $d \in \{1,\ldots,256\}$ against the fixed 47-dimensional AttentionSE autoencoder. For each PCA dimension, inputs $x_i \in \mathbb{R}^{375}$ are projected and reconstructed as $\hat{x}_i^{(d)} = U_d U_d^{\mathsf{T}} x_i$, and the resulting reconstruction RMSE is computed over the evaluation set. The AE at $d=47$ is plotted as a reference line, and PCA requires comparable or larger latent dimensionality to match the AE error.

<a id="fig:ablation-study"></a>
![Reconstruction-RMSE comparison between a PCA latent-dimension sweep and the fixed 47-dimensional AttentionSE autoencoder reference.](../auto-encoder/ablation_study.png)

**Figure:** Reconstruction-RMSE comparison between a PCA latent-dimension sweep and the fixed 47-dimensional AttentionSE autoencoder reference.

In reduced-order modeling, linear POD/PCA provides the optimal rank-$d$ linear reconstruction in the least-squares sense by minimizing

$$
\min_{U_d} \left\lVert X - U_d U_d^{\mathsf{T}} X \right\rVert_F^2,
$$

but nonlinear autoencoders can match or exceed POD performance at fewer dimensions in practice. To assess physical consistency, we also compute the divergence of reconstructed velocity fields (Figure 5 ((a))),

$$
\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z},
$$

and compare its distribution against the PCA baseline; the AE shows improved adherence to the incompressibility constraint.

The vorticity fidelity analysis in Figure 5 ((b)) shows better preservation of high-gradient structures that linear ROMs tend to smooth. In our code, each sample is a $5\times 5\times 5$ velocity cube reshaped into $(u,v,w)$, and vorticity is computed via central differences on the interior stencil:

$$
\omega = \nabla \times \mathbf{u}, \qquad |\omega| = \sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}.
$$

The script averages $|\omega|$ over the interior grid points, compares reconstructed values against the original, and reports absolute-error distributions for AE and PCA. This comparison highlights the AE's advantage in retaining small-scale rotational features that are attenuated by linear compression. Even the AE outliers remain below the 75th percentile of the PCA error distribution, indicating tighter error tails in addition to lower central tendency.

<a id="fig:divergence-vorticity"></a>
![Divergence comparison between the autoencoder reconstruction and linear ROM baseline.](../auto-encoder/divergence_comparison.png)

![Vorticity fidelity comparison showing preservation of high-gradient flow structures; points closer to the 1:1 line indicate better reconstruction fidelity.](../auto-encoder/vorticity_fidelity.png)

**Figure:** Side-by-side comparison of divergence and vorticity fidelity for AE vs. PCA reconstructions.

## Spatio-temporal transformer setup (WAFT)

### Dataset preparation and feature engineering

This appendix subsection expands the compact description from Section 2.3.1 of the main manuscript. Samples are constructed as sequences of 8 time steps at fixed $(y,z)$ locations over 26 adjacent $x$-coordinates, yielding an $8\times26\times52$ tensor per example. Each token concatenates 47 latent features with the physical coordinates $(x,y,z)$, a relative time index, and a scalar experiment parameter used as a Reynolds-number proxy. In implementation, the grid is flattened as

$$
[T_1,X_1] \rightarrow [T_1,X_2] \rightarrow \cdots \rightarrow [T_1,X_{26}] \rightarrow [T_2,X_1] \rightarrow \cdots \rightarrow [T_8,X_{26}],
$$

so attention first sees the full 26-point spatial line at a given time step before advancing to the next step. For the dedicated 80-step vortex-reversal benchmark, the same representation is retained while the temporal span is extended to $\tfrac{80}{120}\,\mathrm{s}=0.667\,\mathrm{s}$; the model receives 12 steps (100 ms) of context and autoregressively predicts the remaining 68 steps along the same 26-coordinate line.

### Model architecture: WAFT (World-model for Autoregressive Fluid Transport)

The transformer uses learned time embeddings for the 8 time indices and learned space embeddings for the 26 $x$ indices, added to a linear projection of each 52-dimensional token. The backbone contains 6 encoder blocks, 8 attention heads, embedding dimension 256, and a triangular causal mask so that predictions at $(T_t,X_j)$ depend only on earlier positions in the flattened ordering. The resulting model contains approximately $4.7\times10^6$ trainable parameters.

# Evaluation and results

## Model performance

This appendix subsection expands the brief performance summary from Section 3.1 of the main manuscript by collecting the full velocity-space RMSE table across reduced-velocity cases.

<a id="tab:rmse-per-experiment"></a>
| Reduced Velocity $U^*$ | RMSE (Velocity Units) |
| --- | --- |
| 5.6 | 1.4360e-03 |
| 7.2 | 1.8134e-03 |
| 8.1 | 2.1019e-03 |
| 10.0 | 2.4484e-03 |
| 10.3 | 2.5014e-03 |
| 11.3 | 3.0021e-03 |
| 12.2 | 2.9725e-03 |
| 13.1 | 3.1018e-03 |
| 16.3 | 3.7093e-03 |
| 17.8 | 3.9328e-03 |

**Table:** Velocity-space RMSE per reduced velocity.

#### Autoregressive performance: staircase evaluation.

This appendix subsection also collects the staircase evaluation table underlying the autoregressive summary from Section 3.1 of the main manuscript.

<a id="tab:transformer-staircase"></a>
| Context Window | Prediction MSE (Latent) | RMSE (Velocity Units) |
| --- | --- | --- |
| Given T1–7 | 1.4876e-04 | 1.2181e-03 |
| Given T1–6 | 1.4895e-04 | 2.3380e-03 |
| Given T1–5 | 1.5420e-04 | 3.7169e-03 |
| Given T1–4 | 1.6033e-04 | 5.5444e-03 |
| Given T1–3 | 1.6800e-04 | 7.7733e-03 |
| Given T1–2 | 1.7683e-04 | 9.8584e-03 |
| Given T1–1 | 1.9047e-04 | 1.1305e-02 |

**Table:** Staircase evaluation: latent-space MSE and velocity-space RMSE for predicting $T_8$ with decreasing context.

## Robustness analysis: data corruption study

This appendix subsection expands the brief robustness summary from Section 3.2 of the main manuscript by collecting the full latent-corruption table and deterioration visualizations. A corruption sweep replaced a percentage of input latent features with random noise in $[-1,1]$, matching the latent range imposed by the autoencoder's final-layer Tanh activation. Table 5 reports the resulting physical-velocity RMSE.
The error curve suggests a two-stage failure mode tied to the token design. Because the 47 latent values are a highly compressed summary of each local 3D velocity cube, perturbing even a small fraction of them can move the state away from combinations seen during training and produce the sharp initial RMSE jump. At larger corruption levels, however, the model still receives the untouched spatial coordinates, relative time index, and experiment parameter, so it retains enough coarse structural information to degrade more gradually rather than collapsing immediately.

<a id="tab:transformer-corruption"></a>
| Corruption % | RMSE (Velocity) | Performance Impact |
| --- | --- | --- |
| 0% | 3.1e-03 | Baseline (Clean) |
| 0.01% | 9.789e-03 | 3.16$\times$ baseline |
| 0.05% | 1.8070e-02 | 5.83$\times$ baseline |
| 0.1% | 3.3240e-02 | 10.72$\times$ baseline |
| 0.5% | 7.5948e-02 | 24.50$\times$ baseline |
| 1% | 1.25e-01 | 40.32$\times$ baseline |
| 10% | 5.8e-01 | 187.10$\times$ baseline |
| 50% | 1.02e+00 | 329.03$\times$ baseline |
| 100% | 1.14e+00 | 367.74$\times$ baseline |

**Table:** Robustness to latent corruption: physical-velocity RMSE and relative impact versus corruption level.

These results indicate high sensitivity to low-level noise (0% to 1% corruption) but graceful degradation at higher corruption levels, suggesting that the model retains partial predictive ability from structured inputs (coordinates and parameter) even when latent state information is heavily degraded.

<a id="fig:transformer-deterioration"></a>
![Spatial visualization of prediction deterioration under latent corruption.](../Transformer/corruption_visualization_flow.png)

![Aggregate error growth versus corruption level.](../Transformer/corruption_deterioration_plot.png)

**Figure:** Deterioration under latent-space corruption for the Transformer.

## Absolute-value reconstruction

This appendix subsection expands the brief summary from Section 3.3 of the main manuscript by collecting the detailed component-wise absolute-value reconstruction table and representative spatial comparison figure for the held-out case $U^*$ = 6.9. Table 6 reports component-wise MAE for $v_x$, $v_y$, and $v_z$ across horizons $T_2$ through $T_8$, and Figure 7 shows representative planar and three-dimensional comparisons at $T_5$.

<a id="tab:absolute-mae-4p4"></a>
| Horizon | MAE($v_x$) | MAE($v_y$) | MAE($v_z$) |
| --- | --- | --- | --- |
| $T_2$ | 9.600240e-04 | 5.718183e-04 | 5.390088e-04 |
| $T_3$ | 8.991537e-04 | 5.334945e-04 | 5.312167e-04 |
| $T_4$ | 9.094232e-04 | 5.353106e-04 | 5.384399e-04 |
| $T_5$ | 9.326175e-04 | 5.402197e-04 | 5.452439e-04 |
| $T_6$ | 8.995433e-04 | 5.462192e-04 | 5.493159e-04 |
| $T_7$ | 9.035515e-04 | 5.553967e-04 | 5.655736e-04 |
| $T_8$ | 9.497145e-04 | 5.890368e-04 | 5.950545e-04 |

**Table:** Component-wise MAE for absolute-value reconstruction on held-out case $U^*$ = 6.9 across horizons $T_2$ through $T_8$.

<a id="fig:absolute-recon-spatial-t5"></a>
[Predicted versus ground-truth absolute velocity on the $x$–$y$ plane at $T_5$.](../Transformer/plane_xy_abs_value_pred_vs_gt_T_5.pdf)

[Three-dimensional view of predicted and ground-truth absolute velocity fields at $T_5$.](../Transformer/xyz_3d_abs_value_pred_vs_gt_T_5.pdf)

**Figure:** Absolute-value reconstruction quality on held-out case $U^*$ = 6.9 at horizon $T_5$, shown in planar and 3D views.

## Coherent vortical structure validation

This appendix subsection expands the concise discussion from Section 3.4 of the main manuscript by unpacking the corresponding planar-slice and 3D isosurface comparisons. In the planar slices at $z=10$ and $z=-10$, the predicted $\omega_z$ field preserves the location, sign, and overall extent of the strongest contours, with the largest discrepancies confined to fine-scale deformation near high-gradient boundaries. In the 3D isosurfaces, the prediction likewise retains the alternating positive and negative structures, their relative phase alignment, and the streamwise coherence of the dominant vortex cores.

These view-specific comparisons matter because vorticity amplifies local velocity-gradient errors. The close agreement across both planar and volumetric views therefore supports the interpretation that the model reconstructs physically meaningful rotational organization rather than only matching pointwise velocity magnitude, while the residual differences are consistent with autoregressive accumulation and derivative sensitivity.

## 80-step vortex-reversal validation

For event localization in the long-horizon benchmark, we use the standard $Q$-criterion together with vorticity. With velocity-gradient tensor $\nabla \mathbf{u}$, its symmetric and antisymmetric parts are defined as

$$
\begin{aligned}
\mathbf{S} &= \tfrac{1}{2}\left(\nabla \mathbf{u} + (\nabla \mathbf{u})^{\top}\right), \\
  \boldsymbol{\Omega} &= \tfrac{1}{2}\left(\nabla \mathbf{u} - (\nabla \mathbf{u})^{\top}\right), \\
  Q &= \tfrac{1}{2}\left(\lVert \boldsymbol{\Omega} \rVert^2 - \lVert \mathbf{S} \rVert^2\right).
\end{aligned}
$$

Regions with $Q>0$ are those in which local rotation dominates strain, making the criterion useful for vortex identification in conjunction with the vorticity field.

### Spatio-temporal sensitivity sweep

This appendix provides the representative event-level sparse-evaluation panels underlying the spatio-temporal sensitivity sweep summarized in Section 3.5.2 of the main manuscript. The comparison visualizes three settings: $(T,C)=(1,5)$, $(12,26)$, and $(40,26)$, where $T$ is the number of context timesteps and $C$ is the number of retained $x$-coordinates per timestep.

<a id="fig:vortex-reversal-sparse"></a>
[Representative sparse-evaluation comparison for the retrained 80-step ***WAFT*** on case $U^*$ = 10. Three settings are shown: $(T,C)=(1,5)$, $(12,26)$, and $(40,26)$.](../80_time_step_data/sparse_evaluation_comparison.pdf)

**Figure:** Representative sparse-evaluation comparison for the retrained 80-step ***WAFT*** on case $U^*$ = 10. Three settings are shown: $(T,C)=(1,5)$, $(12,26)$, and $(40,26)$.

The sparse-evaluation panels make the sensitivity result concrete at the level of an individual event. With only one step and five spatial coordinates, the prediction is noisy and poorly anchored, whereas a 12-step (100 ms) context with the full 26-coordinate line already captures the reversal window reliably. Expanding the context to 40 steps further sharpens the timing of the zero-crossing, which is consistent with the interpretation that deeper temporal context mainly improves phase precision after the minimum reliable threshold has been crossed.

### Event-level recovery dynamics

This appendix subsection provides the representative event-level recovery figure underlying the discussion in Section 3.5.3 of the main manuscript. The step-292 example illustrates what success looks like in this benchmark.

<a id="tab:vortex-reversal-summary"></a>
| Quantity | Value |
| --- | --- |
| Retrained window length | 80 steps (667 ms) |
| Given context | 12 steps (100.0 ms) |
| Autoregressive rollout | 68 steps (566.7 ms) |
| Spatial support per step | 26 $x$-coordinates |
| Evaluated reversal events | 18 |
| Correct reversal detections | 18/18 |
| Mean recovery time | $\sim$28.5 ms (3.4 steps) |
| Recovery range | 17–42 ms |

**Table:** Summary of vortex-reversal validation for the retrained 80-step ***WAFT*** on case $U^*$ = 10.

<a id="fig:vortex-reversal-step292"></a>
[Representative vortex-reversal forecast for case $U^*$ = 10 at step 292 (negative-to-positive). Ground-truth and predicted vorticity are shown over the two identified core $(y,z)$ locations, with the context boundary marked at 12 steps (100 ms).](../80_time_step_data/step292_neg_to_pos_2yz_1200dpi.pdf)

**Figure:** Representative vortex-reversal forecast for case $U^*$ = 10 at step 292 (negative-to-positive). Ground-truth and predicted vorticity are shown over the two identified core $(y,z)$ locations, with the context boundary marked at 12 steps (100 ms).

The shaded gray and red envelopes in Figure 9 denote 95% confidence intervals for the ground-truth and predicted vorticity traces, respectively, so the figure reports uncertainty in addition to the central trajectories. Although the predicted red trace is vertically offset relative to the black ground-truth trace, it crosses through the reversal window immediately after the context boundary and then follows the sign changes of the reference signal. This distinction matters because the event objective is regime prediction—detecting the rotational reversal at the right time—rather than exact amplitude matching of a derivative quantity that is especially sensitive to decoder error.

## Staircase physics evaluation

This appendix subsection expands the brief summary from Section 3.6 of the main manuscript by collecting the full staircase interpolation tables. In both tables, $\mu \pm \sigma$ denotes the mean $\pm$ standard deviation over the $N$ evaluated transitions in the corresponding group. Here, divergence RMSE is computed as $\mathrm{RMSE}_{\nabla\cdot\mathbf{u}}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}\left((\nabla\cdot\mathbf{u}_{\mathrm{pred}})_i-(\nabla\cdot\mathbf{u}_{\mathrm{ref}})_i\right)^2}$, with $\nabla\cdot\mathbf{u}=\partial u/\partial x + \partial v/\partial y + \partial w/\partial z$. As temporal jump size increases, velocity RMSE rises while divergence RMSE decreases, indicating a trade-off between pointwise predictive accuracy and global incompressibility consistency. Across all groups, enstrophy consistency errors remain near machine precision, suggesting that reconstructed fields preserve demanding rotational structure even as forecast horizon increases.

<a id="tab:staircase-summary"></a>
| Transition | N | RMSE ($\mu\pm\sigma$) | Div. RMSE ($\mu\pm\sigma$) |
| --- | --- | --- | --- |
| Jump 1 | 70 | 6.299e-03 $\pm$ 1.159e-03 | 1.462e-03 $\pm$ 6.952e-04 |
| Jump 2 | 60 | 6.943e-03 $\pm$ 1.275e-03 | 1.201e-03 $\pm$ 5.902e-04 |
| Jump 3 | 50 | 7.539e-03 $\pm$ 1.432e-03 | 1.076e-03 $\pm$ 5.216e-04 |
| Fixed $T_1$ | 40 | 9.009e-03 $\pm$ 1.621e-03 | 7.697e-04 $\pm$ 3.636e-04 |

**Table:** Staircase interpolation summary (Part A): direct forecast error and incompressibility metrics.

<a id="tab:staircase-summary-physics"></a>
| Transition | N | Enstrophy MSE ($\mu\pm\sigma$) |
| --- | --- | --- |
| Jump 1 | 70 | 4.220e-34 $\pm$ 1.199e-33 |
| Jump 2 | 60 | 5.224e-34 $\pm$ 1.471e-33 |
| Jump 3 | 50 | 4.762e-34 $\pm$ 1.505e-33 |
| Fixed $T_1$ | 40 | 9.267e-35 $\pm$ 1.766e-34 |

**Table:** Staircase interpolation summary (Part B): enstrophy-consistency errors from PySINDy reconstruction metrics.

# NeurIPS Paper Checklist

## Claims

**Question:** Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
**Answer:** **Yes**
**Justification:** The abstract and Introduction summarize the proposed latent autoencoder–transformer pipeline, the experimental setting, and the main empirical findings, and those claims are supported by the Methodology, Evaluation and Results, and Conclusion sections.
**Guidelines:**
- The answer **N/A** means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A **No** or **N/A** answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## Limitations

**Question:** Does the paper discuss the limitations of the work performed by the authors?
**Answer:** **Yes**
**Justification:** The paper includes a dedicated “Limitations” section that states the dataset-specific scope, the restricted forecast regime, and the absence of formal stability or error guarantees.
**Guidelines:**
- The answer **N/A** means that the paper has no limitation while the answer **No** means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate “Limitations” section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## Theory assumptions and proofs

**Question:** For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
**Answer:** **N/A**
**Justification:** This submission is empirical and does not present new theorems, lemmas, or formal proofs.
**Guidelines:**
- The answer **N/A** means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## Experimental result reproducibility

**Question:** Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
**Answer:** **Yes**
**Justification:** The paper specifies the data acquisition process, preprocessing pipeline, input construction, latent and transformer architectures, optimizer, training sample sizes, and the evaluation protocols used for the reported benchmarks.
**Guidelines:**
- The answer **N/A** means that the paper does not include experiments.
- If the paper includes experiments, a **No** answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/ or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## Open access to data and code

**Question:** Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?
**Answer:** **Yes**
**Justification:** The paper includes a code-and-data availability statement directing readers to this anonymized public release. Resources provided for review are hosted under the anonymized Kaggle namespace [https://www.kaggle.com/datasets/lfmpaper](https://www.kaggle.com/datasets/lfmpaper):
- Source code: [`fdtransformer-source-code`](https://www.kaggle.com/datasets/lfmpaper/fdtransformer-source-code)
- Training dataset: [`training-dataset-training-data-h5`](https://www.kaggle.com/datasets/lfmpaper/training-dataset-training-data-h5)
- Validation dataset: [`validation-data-validation-data-h5`](https://www.kaggle.com/datasets/lfmpaper/validation-data-validation-data-h5)
- Evaluation dataset: [`evaluation-dataset-evaluation-data-h5`](https://www.kaggle.com/datasets/lfmpaper/evaluation-dataset-evaluation-data-h5)
- Original evaluation data: [`original-data-for-evaluation`](https://www.kaggle.com/datasets/lfmpaper/original-data-for-evaluation)

The paper and technical appendix document the associated preprocessing, model settings, and evaluation protocol.
**Guidelines:**
- The answer **N/A** means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines (<https://neurips.cc/public/guides/CodeSubmissionPolicy>) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so **No** is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (<https://neurips.cc/public/guides/CodeSubmissionPolicy>) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## Experimental setting/details

**Question:** Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?
**Answer:** **Yes**
**Justification:** The Methodology section describes the held-out setting, preprocessing, sampling procedure, latent dimensionality, architecture choices, optimizer, and the main evaluation setups needed to understand the reported results.
**Guidelines:**
- The answer **N/A** means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## Experiment statistical significance

**Question:** Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
**Answer:** **Yes**
**Justification:** The Evaluation and Results section now reports uncertainty in several forms: Figure 9 includes 95% confidence intervals for the colored trajectories, Figure 3 summarizes the spread of Pearson correlations across events with boxplots, and Tables 8 and 9 report mean $\pm$ standard deviation with the variability source defined in the text.
**Guidelines:**
- The answer **N/A** means that the paper does not include experiments.
- The authors should answer **Yes** if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).
- If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## Experiments compute resources

**Question:** For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
**Answer:** **Yes**
**Justification:** The paper reports the compute workers and host memory (dual AMD EPYC 9224 CPUs, 768 GiB RAM, one NVIDIA H200 NVL GPU with 143 GB memory), the autoencoder training time (approximately 24 hours), the transformer training time (approximately 72 hours), and an approximate total reported training budget of 96 single-GPU hours for the final runs.
**Guidelines:**
- The answer **N/A** means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## Code of ethics

**Question:** Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?
**Answer:** **Yes**
**Justification:** The work analyzes experimentally measured fluid flows and does not involve personal data, crowdsourcing, or other elements that appear inconsistent with the NeurIPS Code of Ethics.
**Guidelines:**
- The answer **N/A** means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer **No**, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## Broader impacts

**Question:** Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
**Answer:** **Yes**
**Justification:** The paper includes a dedicated “Broader Impact” section discussing beneficial scientific uses of the method and the risk of overconfident use outside the measured operating regime.
**Guidelines:**
- The answer **N/A** means that there is no societal impact of the work performed.
- If the authors answer **N/A** or **No**, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## Safeguards

**Question:** Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?
**Answer:** **N/A**
**Justification:** The paper does not release a high-risk generative model, scraped dataset, or similarly dual-use asset that would require a special controlled-release safeguard discussion.
**Guidelines:**
- The answer **N/A** means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## Licenses for existing assets

**Question:** Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
**Answer:** **N/A**
**Justification:** The core dataset described in the paper is experimentally collected by the authors, and the draft does not rely on redistributed third-party code, datasets, or models whose licenses must be documented here.
**Guidelines:**
- The answer **N/A** means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, <paperswithcode.com/datasets> has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## New assets

**Question:** Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
**Answer:** **Yes**
**Justification:** The paper introduces and publicly releases new source code and experimentally collected datasets through anonymized Kaggle links. The main text and technical appendix document the acquisition process, preprocessing pipeline, intended split roles, and dataset-specific limitations for these released assets.
**Guidelines:**
- The answer **N/A** means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/ code/ model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## Crowdsourcing and research with human subjects

**Question:** For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
**Answer:** **N/A**
**Justification:** The paper does not involve crowdsourcing or research with human subjects.
**Guidelines:**
- The answer **N/A** means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## Institutional review board (IRB) approvals or equivalent for research with human subjects

**Question:** Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
**Answer:** **N/A**
**Justification:** The paper does not involve crowdsourcing or research with human subjects, so IRB-style approval is not applicable.
**Guidelines:**
- The answer **N/A** means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## Declaration of LLM usage

**Question:** Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does *not* impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

**Answer:** **N/A**
**Justification:** LLMs are not part of the core methodology or experimental contribution of this paper.
**Guidelines:**
- The answer **N/A** means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.