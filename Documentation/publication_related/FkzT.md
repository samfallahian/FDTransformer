We thank the reviewer for a careful and balanced reading. We completed the stage-membership audit, corrected the enstrophy evaluation to $3.89\times10^{-10}$, and verified centroid-only field assembly. These findings are reported now; retraining, additional diagnostics, and framing updates will be completed before final publication.

## W1. Novelty is primarily in the data rather than the methodology

We agree that the individual components—an autoencoder, local tokenization, and a causal transformer—are established tools. Our intended contribution is not a new transformer primitive. It is the representation and empirical finding: overlapping local 3D velocity neighborhoods are compressed into a nonlinear basis before sequence modeling, and that representation is tested on noisy experimental volumetric TR-PTV wake data with gradient-sensitive diagnostics.

The reviewer categorized the contribution as use-inspired, which is the lens under which we believe the work is strongest. The scientific question is whether a local nonlinear representation preserves experimentally measured wake structure better than a matched linear representation and remains useful under rollout. We will revise the novelty statement to make this system-level and empirical claim explicit, while avoiding language that suggests a broadly new transformer architecture.

We would also welcome a specific pointer to prior work that combines overlapping local-cube nonlinear compression with causal latent forecasting on experimental volumetric wake measurements; such a reference would materially improve the positioning.

## W2. Scope and generalization

An audit prompted by the reviews revealed complementary but misaligned stage-specific holdouts. The $U^{\ast}=6.9$ case was excluded from the autoencoder but included in both transformer training pools, whereas $U^{\ast}=10$ was included in the autoencoder but excluded from both transformer training pools. Thus, the current $U^{\ast}=6.9$ result tests representation transfer but not dynamics transfer, while the $U^{\ast}=10$ result tests dynamics transfer but not representation transfer. Neither is an end-to-end holdout.

This audit also corrects Sections 3.3 and 3.5 of the manuscript, which inaccurately describe $U^{\ast}=6.9$ as held out and excluded from transformer training. We will correct those statements before final publication.

Before final publication, we will retrain a shared autoencoder excluding both $U^{\ast}=6.9$ and $U^{\ast}=10$. We will also partition eligible experiment–time–centroid indices before sampling so its training and validation sets are exactly disjoint. We will then designate $U^{\ast}=6.9$ as the end-to-end holdout for the 8-step study and $U^{\ast}=10$ as the end-to-end holdout for the 80-step study. This addresses both cross-stage leakage and exact duplicate rows between autoencoder training and validation; intentional overlap among distinct neighboring cubes remains part of the representation.

These reruns will evaluate end-to-end transfer across previously unseen operating conditions of a single coupled VIV system, rather than transfer to different geometries or experimental facilities. However, fixing the cylinder geometry does not imply that all cases belong to a single flow regime. The eleven experimental conditions span
$$
U^{\ast}=5.6\text{--}17.8,
$$
covering both lock-in and off-lock-in responses observed experimentally. Since
$$
U^{\ast}=\frac{U}{f\_nD}
=\frac{T\_n}{T\_c},
 $$
changing $U^{\ast}$ changes the ratio between the structural natural time scale and the flow convective time scale. Consequently, the coupled fluid–structure system exhibits substantially different vibration amplitudes, synchronization behavior, vortex shedding patterns, and wake organization across the dataset. Unlike the wake of a stationary cylinder, the evolving flow both drives and is modified by the structural motion, producing distinct coupled dynamics rather than simple variations of a fixed wake.

To better illustrate this point, the revised manuscript will include the measured VIV response diagram, indicating the training and held-out operating conditions together with their vibration amplitudes and dominant frequencies. This will clarify that the proposed model is evaluated on previously unseen dynamical regimes of the same physical system, rather than merely unseen snapshots from a single regime.

To further address the reviewer's concern regarding physical fidelity, we will additionally compare the predicted and measured turbulent kinetic energy spectra for the held-out cases (and corresponding POD energy distributions, space permitting). These diagnostics complement the pointwise error metrics by assessing whether the learned dynamics preserve the experimentally observed distribution of energy across the dominant flow scales.

## W3. Does lower divergence indicate physics or over-damping?

We agree that lower divergence alone does not establish a better physical forecast. The decisive control is the divergence of the measured ground-truth field, which is not exactly zero because TR-PTV measurements contain finite noise and reconstruction error. If predicted divergence falls below that measurement floor while kinetic energy is also attenuated, the appropriate interpretation is over-smoothing, not improved incompressibility.

The enstrophy analysis also required correction. The original $10^{-34}$ values were produced by comparing enstrophy with the same predicted vorticity source from which it had just been calculated; they measured algebraic self-consistency, not prediction-versus-truth fidelity. We corrected the pipeline to compare predicted vorticity with enstrophy independently calculated from the raw ground-truth field. The corrected cross-source MSE is $3.89\times10^{-10}$. Before final publication, we will replace the original values and remove the machine-precision forecast-fidelity claim.

We also agree that normalized divergence/enstrophy metrics and prediction-versus-truth kinetic-energy spectra are the appropriate diagnostics for separating stable structure from damping.

## W4. Theoretical insight and continuity

We do not claim a formal stability theorem and will not imply one. The current evidence is empirical: short-horizon autoregressive degradation, the 80-step reversal traces, and vortex-core autocorrelation. These should be described as empirical stability diagnostics rather than guarantees.

The continuity claim should be stated precisely. Adjacent input cubes share $100/125=0.80$ (80%) of their measurements, creating correlated local tokens, but this is not a partition-of-unity reconstruction. Production evaluation is strictly centroid-only:
$$
\boxed{c=62},
\qquad
\widehat{\mathbf{u}}\_{t+1}(\mathbf{x})
=\mathcal{S}\_{c}\left(
\mathcal{D}(\widehat{\mathbf{z}}\_{t+1,\mathbf{x}})
\right)
=(\widehat{v}\_{x,c},\widehat{v}\_{y,c},\widehat{v}\_{z,c})^{\mathsf T},
$$
where the boxed center index $c=62$ corresponds to $(\Delta x,\Delta y,\Delta z)=(0,0,0)$ and $\mathcal{S}\_{c}$ selects the streamwise, transverse, and spanwise components $(\widehat{v}\_{x,c},\widehat{v}\_{y,c},\widehat{v}\_{z,c})$ from the decoded $125\times3$ cube. No averaging or blending of overlapping outputs is performed. Thus, overlap supplies a representation-level neighborhood correlation without introducing assembly smoothing; we will document this distinction explicitly.

## W5. Related work

Agreed. The absence of a Related Work section is a substantive presentation gap. Before final publication, we will position the work against neural operators such as FNO, transformer PDE surrogates such as Transolver and Universal Physics Transformers, latent reduced-order forecasting, graph/mesh methods, and Koopman-style models. We will also state clearly that our present evidence does not establish superiority over those families.

## W6. Abstract and framing corrections

The submission-portal abstract uses “divergence-free transport velocity,” while the manuscript does not claim or implement a divergence-free constraint. Before final publication, we will correct that wording and replace broad generalization language with a precise description of the current stage-specific evaluations and planned end-to-end holdout reruns within one experimental setup.

We are grateful that the review identifies concrete changes that sharpen both the scientific claim and its scope.