import numpy as np
import torch
from CAE import CAE

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

class TestCAE:
    def __init__(self, model):
        self.model = model
        self.model.eval()  # Set the model to evaluation mode
        self.test_data = torch.tensor(np.random.rand(1000, 125, 3), dtype=torch.float32).to(device)

    def run_test(self):
        with torch.no_grad():
            outputs = self.model(self.test_data)
        # Additional processing or assertions can be done here

model = CAE().to(device)
test = TestCAE(model)
test.run_test()
