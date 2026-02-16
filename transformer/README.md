# Transformer Dataset Layout (HDF5)

This directory contains the logic for preparing and consuming the datasets used by the Transformer models. The data is stored in HDF5 format for efficient random access and storage.

## 📂 File Hierarchy & Nesting

The datasets follow a hierarchical structure that maps physical space and time into a format suitable for sequence modeling.

### 1. Training & Validation Files
`training_data.h5` | `validation_data.h5`

```text
ROOT (File)
├── 🏷️ Attributes
│   ├── feature_description: "0-46: latent, 47: x, 48: y, 49: z, 50: relative_time, 51: parameter_value"
│   └── x_coords: [-50, -46, ..., 49] (26 values)
└── 📊 Datasets
    └── data (N, 8, 26, 52) [float32]
        └── [Sample ID] (N)
            └── [Time Step] (8 consecutive steps)
                └── [X-Coordinate] (26 fixed points)
                    └── [Features] (52 values)
```

### 2. Evaluation File (The Contrast)
`evaluation_data.h5`

The evaluation file is structured similarly to the training data but includes an additional `originals` dataset. This dataset stores the **ground-truth physical velocities** for the final time step in the window, allowing for direct accuracy assessment in physical space.

```text
ROOT (File)
├── 🏷️ Attributes
│   ├── ... (same as above)
│   └── originals_description: "Original vx, vy, vz for the 8th timestep (t_idx=7)"
└── 📊 Datasets
    ├── data (N, 8, 26, 52) [float32]
    └── originals (N, 26, 3) [float32]
        └── [Sample ID]
            └── [X-Coordinate]
                └── [vx, vy, vz] (3 values)
```

---

## 🧩 Feature Mapping (Index 0-51)

Each "token" in the sequence is a vector of 52 features:

| Indices | Name | Description |
| :--- | :--- | :--- |
| **0 - 46** | **Latents** | 47-dimensional latent representation from the Autoencoder. |
| **47** | **X** | Physical X-coordinate. |
| **48** | **Y** | Physical Y-coordinate (fixed for the sample). |
| **49** | **Z** | Physical Z-coordinate (fixed for the sample). |
| **50** | **Time** | Relative time index within the window (0.0 to 7.0). |
| **51** | **Param** | The experimental parameter value (e.g., 7.8). |

---

## 🔄 Sequence Flattening

While the HDF5 stores data as `(8, 26, 52)`, the Transformer processes it as a flattened sequence of **208 tokens**. The data is flattened such that it completes one spatial row (X) before moving to the next time step (T):

**Sequence Order:**
`[T0, X0] → [T0, X1] → ... → [T0, X25] → [T1, X0] → ... → [T7, X25]`

This allows the model to learn both spatial correlations (along X) and temporal dynamics simultaneously.

---

## 🛠️ How to Access the Data

### Using `h5py` (Python)
```python
import h5py

with h5py.File("training_data.h5", "r") as f:
    # Get metadata
    print(f.attrs['feature_description'])
    
    # Access a specific sample
    sample = f['data'][0] # Shape: (8, 26, 52)
    
    # Flatten for Transformer
    flattened = sample.reshape(208, 52)
```

### Using `TransformerDataset` (PyTorch)
```python
from dataset import TransformerDataset
from torch.utils.data import DataLoader

ds = TransformerDataset("training_data.h5")
loader = DataLoader(ds, batch_size=32, shuffle=True)

for batch in loader:
    # batch shape: (Batch, 8, 26, 52)
    inputs = batch.view(-1, 208, 52)
    # ... training loop ...
```
