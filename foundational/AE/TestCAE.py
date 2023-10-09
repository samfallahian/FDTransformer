import numpy as np
import torch

from CAE import CAE  # Ensure that this module is available

# Initialize default device as CPU
device = torch.device("cpu")

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
else:
    device = torch.device("mps")  # Update device if MPS is available

    # Create a Tensor directly on the mps device
    x = torch.ones(5, device=device)
    # Or
    x = torch.ones(5, device="mps")

    # Any operation happens on the GPU
    y = x * 2


class TestCAE:

    def setup_method(self, method):
        self.model = CAE().to(device)
        self.model.eval()  # Set the model to evaluation mode
        self.test_data = torch.tensor(np.random.rand(1000, 125, 3), dtype=torch.float32).to(device)

    def test_run(self):
        with torch.no_grad():
            outputs = self.model(self.test_data)
        # Add assertions here as necessary

    def test_device_is_mps(self):
        assert device.type == "mps", f"Expected device to be MPS, but got {device.type}"

    def test_device_is_mps(self):
        assert device.type == "mps", f"Expected device to be MPS, but got {device.type}"



