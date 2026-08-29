We thank the reviewer for identifying presentation gaps, especially missing related work and unclear organization. We verified the tokenization, reconstruction, and corruption analyses reported below; manuscript restructuring, ablations, and external baselines will be completed before final publication.

## Presentation and organization

The acquisition is not absent, but it is insufficiently surfaced. The technical appendix already documents the four-camera LaVision TR-PTV/Shake The Box setup, 120-fps, 1200-frame records, normalization, and preprocessing. We will promote a concise summary to the main text and add the governing setting
$$
\partial\_t\mathbf{u}+(\mathbf{u}\cdot\nabla)\mathbf{u}
=-\rho^{-1}\nabla p+\nu\nabla^2\mathbf{u},
\qquad
\nabla\cdot\mathbf{u}=0,
$$
while stating that these equations describe the measured flow but are not enforced by the model. We will also define “ground truth” as the experimentally reconstructed 3D–3C reference field, including measurement/reconstruction uncertainty, and give corrected train/evaluation membership.

In the same revision, we will add a Related Work section and reorganize the evaluation so compression, short-horizon forecasting, long-horizon events, robustness, and physics diagnostics are separated. Conventional error-versus-horizon curves and spatial error maps will accompany the staircase view, with discussion of error concentration in gradient-sensitive regions.

## Physical motivation and breadth of the VIV dataset

We agree that the physical motivation should be stated more clearly. Although the
benchmark is restricted to a single circular-cylinder geometry, it does not
represent a single stationary or weakly varying wake state. The eleven experimental conditions span
$$
U^{\ast}=5.6\text{--}17.8.
$$
covering canonical vortex-induced vibration (VIV) behavior from pre-lock-in through lock-in and post-lock-in conditions.

The reduced velocity is not merely a normalized freestream velocity. It can be written as
$$
U^{\ast}=\frac{U}{f\_nD}
=\frac{T\_n}{T\_c},
$$
where $T\_n=1/f\_n$ is the structural natural period and $T\_c=D/U$ is the
convective time scale. It therefore represents the ratio between the structural and flow time scales and governs the synchronization that underlies VIV.

Unlike the wake of a stationary cylinder, VIV is governed by two-way fluid–structure coupling: the unsteady wake excites structural motion, while the moving cylinder modifies flow separation, vortex formation, shedding phase, and the resulting fluid forcing. As $U^{\ast}$ varies, the coupled system exhibits different vibration amplitudes, synchronization behavior, phase relationships, and wake organization. The purpose of the present benchmark is therefore to evaluate latent forecasting across previously unseen dynamical operating conditions of a coupled fluid-structure system while maintaining a controlled geometry and measurement configuration.

To better demonstrate this physical diversity, the revised manuscript will include the measured VIV response curves (vibration amplitude and dominant frequency versus $U^{\ast}$), identify the training and held-out operating conditions on these response maps, and provide representative vorticity or Q-criterion visualizations illustrating the corresponding wake evolution.

Finally, we emphasize the intended scope of the benchmark. The present experiments reproduce the canonical response of a classical VIV system that has been extensively documented in the literature, suggesting that the learned dynamics reflect established VIV physics rather than facility-specific behavior. Nevertheless, because all data are acquired using a single geometry and experimental configuration, and because $Re$ and $U^{\ast}$
 co-vary with freestream velocity, we do not claim generalization to different geometries, sensing layouts, experimental facilities, or independently varied Reynolds-number regimes.

## Q1. Vortex size versus the $5\times5\times5$ neighborhood

We may not have made one distinction clear enough. The neighborhood tensor
$$
\mathbf{V}\_{t,i}
=\left[\mathbf{u}\_t(\boldsymbol{\xi}\_i+\boldsymbol{\delta})\right]\_{
\boldsymbol{\delta}\in\lbrace-2,-1,0,1,2\rbrace^{3}}
\in\mathbb{R}^{5\times5\times5\times3}
$$
is the tokenization stencil, not the transformer's complete receptive field. It is encoded as
$$
\mathbf{z}\_{t,i}=\mathcal{E}(\mathbf{V}\_{t,i})\in\mathbb{R}^{47},
\qquad
\mathbf{q}\_{t,i}=[\mathbf{z}\_{t,i};\boldsymbol{\xi}\_i;\tau\_t;r]
\in\mathbb{R}^{52},
$$
where $\boldsymbol{\xi}\_i=(x\_i,y\_i,z\_i)^{\mathsf T}$, $\tau\_t$ is relative time, and $r$ is the experiment parameter.

The transformer receives the time-major matrix
$$
\mathbf{Q}^{(T)}=[\mathbf{q}\_{1,1},\ldots,\mathbf{q}\_{T,26}]^{\mathsf T}
\in\mathbb{R}^{26T\times52},
\qquad
n(t,i)=26(t-1)+i.
$$

For $T\in\lbrace8,80\rbrace$, $\mathbf{Q}^{(T)}$ contains 208 or 2,080 tokens. The 80-step model uses 312 context tokens and generates 1,768; the staircase varies same-slice context within 208 tokens. Causal attention therefore represents vortices larger than one stencil jointly across space and time.

The current results do not isolate stencil choice. Before final publication, we will document and compare feasible settings
$$
\boldsymbol{\theta}=(s,\rho,d\_z,\mathcal{F}\_{\mathrm{aux}}),
$$
covering stencil width, overlap, latent dimension, and auxiliary features without precommitting to unevaluated values.

## Q2 and Q3. FNO/Transolver comparisons and the simple transformer backbone

PCA/POD evaluates compression rather than forecasting. Targeted controls are persistence, the same transformer on PCA/POD latents, and a budget-matched transformer on raw cubes, isolating nonlinear compression while holding the sequence model fixed.

The conventional causal transformer is deliberate: the hypothesis concerns the representation, not a specialized attention mechanism. Fixing the backbone attributes differences more directly to the local nonlinear basis.

The latent-corruption study provides complementary evidence: replacing all 9,776 latent values across the 208-token history while retaining coordinate, time, and experiment channels increases RMSE from $3.1\times10^{-3}$ to $1.14$ ($367.7\times$). Thus, the conventional backbone materially uses encoded flow history rather than reducing to a metadata lookup, although this does not replace direct FNO or Transolver comparisons.

FNO and Transolver remain important baselines. We will not present rushed comparisons; fair evaluations not completed during discussion will be completed before final publication.

## Q4. Why call the model a world model?

We use “world model” in the predictive latent-dynamics sense. With measured flow state $\mathbf{y}\_t$, latent state $\mathbf{z}\_t$, and physical context $\mathbf{c}\_t$, the pipeline has the standard encoder–dynamics–decoder form
$$
\mathbf{z}\_t=\mathcal{E}(\mathbf{y}\_t),
\qquad
\widehat{\mathbf{z}}\_{t+1}
=\mathcal{F}(\mathbf{z}\_{t-k:t},\mathbf{c}\_{t-k:t}),
\qquad
\widehat{\mathbf{y}}\_{t+1}=\mathcal{D}(\widehat{\mathbf{z}}\_{t+1}).
$$
Action-conditioned planning is not the only precedent: Ross et al., “When do World Models Successfully Learn Dynamical Systems?” (2025), study latent dynamics for physical systems including a Kármán vortex street, while Luo et al., “EO-WM” (2026), frame exogenously driven Earth-observation forecasting as world modeling. We therefore use *latent physical world model* for learned representation and transition dynamics, while distinguishing this predictive setting from planning or control.

## Novelty and significance

The novelty claim is not a new transformer primitive, but a local-chart representation for experimental volumetric flow dynamics. For each grid centroid $\mathbf{x}$, the encoder maps a complete local neighborhood into a nonlinear coordinate,
$$
\mathbf{z}\_{\mathbf{x}}
=\mathcal{E}\left(\mathbf{V}\_{\mathcal{N}(\mathbf{x})}\right),
\qquad
\mathbf{V}\_{\mathcal{N}(\mathbf{x})}\in\mathbb{R}^{125\times3},
\qquad
\mathbf{z}\_{\mathbf{x}}\in\mathbb{R}^{47}.
$$
For centroids separated by one grid point along a coordinate direction,
$$
\left|\mathcal{N}(\mathbf{x})\cap
\mathcal{N}(\mathbf{x}+\mathbf{e}\_i)\right|
=4\times5\times5=100,
\qquad
\frac{100}{125}=0.80.
$$
Adjacent tokens therefore encode overlapping local charts of the measured flow rather than independent point samples.

The transformer advances these neighborhood-conditioned coordinates jointly across space and time,
$$
\left\lbrace\mathbf{z}\_{t,\mathbf{x}}\right\rbrace\_{t,\mathbf{x}}
\longrightarrow
\left\lbrace\widehat{\mathbf{z}}\_{t+1,\mathbf{x}}\right\rbrace\_{\mathbf{x}},
$$
rather than forecasting isolated velocity points directly in the ambient $375$-dimensional cube space. This anticipates the difficulty of learning evolution near a nonlinear flow-state manifold by supplying overlapping, neighborhood-conditioned coordinates. It is an inductive bias, not a formal guarantee that every predicted latent remains on-manifold.

Centroid-only decoding then recovers one velocity per location without post-hoc overlap averaging,
$$
\widehat{\mathbf{u}}\_{t+1}(\mathbf{x})
=\mathcal{S}\_{62}\left(
\mathcal{D}(\widehat{\mathbf{z}}\_{t+1,\mathbf{x}})
\right),
$$
so the spatial inductive bias originates in representation learning rather than smoothing during field assembly. The $367.7\times$ clean-to-fully-corrupted RMSE increase reported above further shows that prediction materially depends on this organized latent history. We will present this as a system-level and experimental contribution on real TR-PTV wake measurements, while retaining the stated limits of one geometry, one acquisition pipeline, and incomplete comparison with neural operators.