### Methods: Data Generation and Pre-processing

#### Experimental Setup and Data Acquisition
Flow *velocity* data were generated at the [Fluid-Structure Interaction (FSI) Lab](https://fsilab.sites.umassd.edu/about/) at the University of Massachusetts Dartmouth. Quantitative flow measurements were acquired using a Time-Resolved Volumetric Particle Tracking *Velocimetry* (TR-PTV) system. The experimental setup utilized a LaVision configuration consisting of:
*   **Illumination:** A 300 × 100 mm² LED array (FLASHLIGHT 300).
*   **Imaging:** A four-camera Minishaker box equipped with 8, 12, and 16 mm focal length lenses.
*   **Synchronization:** Cameras and illumination were triggered simultaneously via a LaVision Programmable Timing Unit (PTU).
*   **Data Acquisition:** High-speed video was acquired using Sony IMX252LLR/LQR CMOS cameras (2048×1088 resolution) and recorded uncompressed via a Core DVR.

The raw image sequences were processed using the **Shake The Box (STB)** algorithm within the DaVis 10 software environment. This enabled time-resolved, three-dimensional, three-component (3D-3C) Lagrangian particle *velocity* measurements $\langle v_x, v_y, v_z \rangle$ within the volume.

#### Data Preparation and Quality Assurance
Prior to model training, the raw Lagrangian data underwent a multi-stage pre-processing pipeline to ensure spatial consistency and numerical stability:

1.  **Health Assessment:** Each data file was audited to verify the presence of a complete temporal sequence ($T = \{1, \dots, 1200\}$) and consistent row counts.
2.  **Statistical Analysis of Extremes:** Global extrema for all *velocity* components ($v_x, v_y, v_z$) were determined to establish bounds for normalization.
3.  **Data Type Correction:** To optimize computational efficiency, spatial/temporal coordinates were cast to 32-bit integers, and *velocity* components to 32-bit floats.
4.  **Normalization:** *Velocity* components were scaled to a $[0, 1]$ range using a linear transformation based on global extremes (approximately $-1.98$ to $2.64$ m/s), ensuring stable convergence during neural network training.

#### Spatial Mapping and Cubic Reconstruction
To transition from discrete point measurements to a structured representation suitable for deep learning, the data were reorganized into local volumetric neighborhoods.

As illustrated in **Figure 1** (*experimental_box.png*), a sampling volume was defined by trimming the edges of the total measurement domain. This trimming ensures that every selected centroid possesses a complete surrounding neighborhood.

For each valid centroid, a $5 \times 5 \times 5$ coordinate neighborhood was established (**Figure 2**, *mapping_visualization.png*). This neighborhood consists of the target centroid and its 124 nearest neighbors across the grid. The 3D-3C *velocity* components for all 125 points were gathered and flattened into a 375-dimensional feature vector $\mathbf{V} \in \mathbb{R}^{375}$ (125 points $\times$ 3 *velocity* components). These "*velocity* cubes" capture the local flow structure, providing the high-dimensional input required for manifold learning.

---

### Visualizations and Artifacts
The following high-resolution artifacts (1200 DPI) have been generated in the `Documentation/` directory:

1.  `Documentation/experimental_box.png`: A 3D schematic of the TR-PTV measurement domain, identifying the "Valid Sampling Region" (interior volume) and centroids used for data extraction.
2.  `Documentation/mapping_visualization.png`: A 3D breakdown of the $5 \times 5 \times 5$ sampling neighborhood, illustrating the mapping of 125 spatial points to the 375-dimensional feature vector $\mathbf{V}$.

The generating script, `generate_mapping_viz.py`, is available in the project root and is configured for publication-quality output.
