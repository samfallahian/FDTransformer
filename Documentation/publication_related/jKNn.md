We thank the reviewer for the detailed review. We corrected the enstrophy evaluation to $3.89\times10^{-10}$, verified centroid-only field assembly, and confirmed that global corruption of 208 tokens is equivalently corruption of 9,776 latent values. These findings are reported now; remaining baseline, energy, and noise controls will be completed before final publication.

## Q3. Why were the enstrophy errors near machine precision?

The reviewer was correct: the original values did not measure physical forecast fidelity. The previous pipeline calculated the enstrophy target directly from the predicted vorticity components and then fit the same algebraic relationship,
$$
\mathcal{E}=\tfrac12\left(\omega\_x^2+\omega\_y^2+\omega\_z^2\right).
$$
Consequently, the reported $10^{-34}$ MSE values measured numerical self-consistency within one predicted data source. They did not compare predictions with independent ground truth.

We corrected the evaluation using a cross-source protocol: the predictors are the vorticity components from the transformer output, while the enstrophy target is calculated independently from the raw ground-truth field. The corrected physical-validation MSE is $3.89\times10^{-10}$. Predicted-versus-predicted and raw-versus-raw self-consistency remain near $2.73\times10^{-39}$ and $6.91\times10^{-35}$, respectively, but these values are now labeled only as algebraic implementation checks. We will replace the table and remove the abstract/conclusion claim that forecast enstrophy is preserved near machine precision.

## Q1. Stronger forecasting baselines

We agree that the PCA/POD comparison in the submitted paper evaluates local compression more directly than forecasting. The most diagnostic forecasting baselines are:

1. persistence at each horizon;
2. the same transformer trained on local PCA/POD latents; and
3. a budget-matched transformer trained directly on the flattened velocity cubes.

These isolate whether performance comes from the local nonlinear representation, from compression generally, or simply from the transformer. ConvLSTM/TCN and neural-operator baselines remain valuable broader comparisons; those not completed during discussion will be evaluated carefully before final publication.

## Scope and dynamical breadth of the VIV benchmark

We agree that the study does not demonstrate transfer to another geometry,
facility, or sensing layout. We will narrow the generalization claims
accordingly. However, the fixed circular-cylinder geometry does not correspond
to a single stationary wake condition. The dataset contains eleven independently
acquired experimental cases spanning
$$
U^{\ast}=5.6\text{--}17.8.
$$
Reduced velocity compares two physical time scales:
$$
U^{\ast}=\frac{U}{f\_nD}
=\frac{T\_n}{T\_c},
$$
where $T\_n=1/f\_n$ is the structural natural period and $T\_c=D/U$ is the
convective time scale. Changes in $U^{\ast}$ therefore alter the synchronization
between structural motion and wake formation.

This is fundamentally different from a stationary-cylinder forecast. In VIV,
the wake force drives the body, while the moving boundary feeds back on
separation, vortex formation, shedding phase, and subsequent loading. The
experimental range consequently contains changes in vibration response,
frequency synchronization, and wake organization across the VIV regime. In the
revision, we will document this dynamical breadth using the measured
response-amplitude and dominant-frequency curves and representative
vorticity/$Q$-criterion fields.

Because $Re$ and $U^{\ast}$ both change with freestream velocity in the present
experiments, their individual influences are not independently isolated. Our
claim will therefore be limited to transfer across coupled VIV operating
conditions within one geometry and acquisition pipeline.

## Q2. How are overlapping decoded cubes assembled?

The production evaluation uses strictly centroid-only reconstruction; there is no averaging or blending of overlapping decoded cubes. For an interior grid location $\mathbf{x}$, let the transformer predict
$$
\widehat{\mathbf{z}}\_{t+1,\mathbf{x}}
=\mathcal{T}\left(\mathbf{Z}\_{\mathrm{context}}\right)\_{\mathbf{x}}
\in\mathbb{R}^{47}.
$$
The decoder output is reshaped as
$$
\widehat{\mathbf{V}}\_{t+1,\mathbf{x}}
=\mathrm{reshape}\_{125\times3}
\left(\mathcal{D}(\widehat{\mathbf{z}}\_{t+1,\mathbf{x}})\right)
\in\mathbb{R}^{125\times3}.
$$
Neighbor index $c=62$ is the zero-offset entry $(\Delta x,\Delta y,\Delta z)=(0,0,0)$. The reconstructed velocity is therefore
$$
\widehat{\mathbf{u}}\_{t+1}(\mathbf{x})
=\mathcal{S}\_{62}\left(\widehat{\mathbf{V}}\_{t+1,\mathbf{x}}\right)
=\widehat{\mathbf{V}}\_{t+1,\mathbf{x}}[62,:]
=\widehat{\mathbf{v}}\_{62}.
$$
The field is the collection of these one-to-one centroid predictions.

| Operation | Autoencoder training | Transformer/diagnostics |
| --- | --- | --- |
| Decode $125\times3$ cube | All 375 values in loss | Full cube produced |
| Select zero-offset row $62$ | Included in full-cube loss | Sole field value |
| Other 124 rows | Included in full-cube loss | Discarded |
| Average overlapping outputs | Not applicable | Not performed |

Every interior point serves as the geometric center of its own cube, so no point receives multiple values requiring aggregation. This avoids the low-pass filtering that averaging or weighted overlap blending could introduce; the reported divergence, vorticity, and enstrophy diagnostics are already computed from the centroid-only field.

The paper separately uses “centroid” for a vorticity-weighted vortex-core location. We will distinguish that diagnostic statistic from the geometric cube center selected above.

## Lower divergence may reflect smoothing

Agreed. The submission-portal abstract incorrectly described the predicted transport velocity as “divergence-free,” although the model neither imposes nor guarantees a divergence-free constraint. We will remove that wording and state that divergence is evaluated diagnostically rather than enforced. Moreover, measured TR-PTV fields are not exactly divergence-free, so the divergence magnitude of the raw ground-truth field must serve as the reference floor. Before final publication, this control will be paired with prediction-versus-truth energy spectra; no assembly-smoothing correction is required because the production reconstruction is centroid-only.

## Latent-coordinate corruption sensitivity

We agree that “graceful” was inaccurate for the low-corruption regime, but the intervention does not corrupt only one cube. It acts globally across the complete history presented to the transformer. With $T=8$ time steps, $X=26$ spatial cube centroids, latent width $d\_z=47$, and context width $d\_c=5$,
$$
N\_{\mathrm{tok}}=TX=8\times26=208,
\qquad
N\_Z=TXd\_z=9{,}776,
\qquad
N\_C=TXd\_c=1{,}040.
$$
These are equivalent descriptions of the same corrupted history:
$$
208\text{ tokens}
\quad\Longleftrightarrow\quad
208\times47=9{,}776\text{ latent values}.
$$
The five coordinate, time, and experiment features per token remain uncorrupted.

For corruption fraction $p$, the implementation selects
$$
m(p)=\mathrm{round}(pN\_Z),
\qquad
\lVert\mathbf{M}\rVert\_0=m(p),
\qquad
U\_{t,x,k}\sim\mathcal{U}[-1,1],
$$
and applies
$$
\widetilde{\mathbf{Z}}\_p
=(\mathbf{1}-\mathbf{M})\odot\mathbf{Z}
+\mathbf{M}\odot\mathbf{U},
\qquad
\widetilde{\mathbf{X}}\_p
=\left[\widetilde{\mathbf{Z}}\_p\mid\mathbf{C}\right].
$$

| Corruption | Expected overwritten scalars | Velocity RMSE | Relative RMSE |
| --- | --- | --- | --- |
| 0% | 0 | $3.100\times10^{-3}$ | $1.00\times$ |
| 0.01% | 0.98 ($\approx1$) | $9.789\times10^{-3}$ | $3.16\times$ |
| 0.05% | 4.89 ($\approx5$) | $1.807\times10^{-2}$ | $5.83\times$ |
| 0.1% | 9.78 ($\approx10$) | $3.324\times10^{-2}$ | $10.72\times$ |
| 0.5% | 48.88 ($\approx49$) | $7.595\times10^{-2}$ | $24.50\times$ |
| 1% | 97.76 ($\approx98$) | $1.250\times10^{-1}$ | $40.32\times$ |
| 10% | 977.60 ($\approx978$) | $5.800\times10^{-1}$ | $187.10\times$ |
| 50% | 4,888 | $1.020$ | $329.03\times$ |
| 100% | 9,776 | $1.140$ | $367.74\times$ |

At 100% corruption,
$$
\widetilde{\mathbf{X}}\_{p=1}
=\left[\mathbf{U}\mid\mathbf{C}\right],
\qquad
\mathbf{U}\perp\mathbf{Z},
$$
so all $9{,}776$ latent scalars across all 26 cubes and all 8 time steps are replaced. The model is blind to the true fluid history but retains all 208 sets of physical-context features. The large clean-to-100% gap demonstrates that latent history materially informs prediction rather than the model acting only as a coordinate/parameter lookup. The plateau near RMSE $\approx1.14$ shows bounded saturation under complete history loss, but random latents still enter the model; a zero-latent control is needed to isolate the metadata-conditioned prior.

Although values remain within the Tanh range, $\widetilde{\mathbf{Z}}\_p$ is generally off-manifold; this is activation replacement, not measurement noise. LLM corruption studies show qualitatively similar deterioration, but are not numerical benchmarks. Unlike the corrected SINDy artifact, this test directly measures prediction failure under controlled latent-history loss.

Before final publication, we will report random-mask variability and a logarithmic 0.01–1% inset, plus a zero-latent control $\widetilde{\mathbf{Z}}=\mathbf{0}$ and additive velocity noise at realistic TR-PTV levels.

## Uncited and irrelevant references

Agreed. We will audit the bibliography for entries that are not cited in the text.