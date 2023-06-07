import unittest
import pickle
import torch
from ContractiveAutoencoder import ContractiveAutoencoder

# Specify the path to the pickle file
pickle_file = "_data_train_autoencoder.pickle"

class AutoencoderTest(unittest.TestCase):
    def setUp(self):
        # Read the pickle file
        with open(pickle_file, "rb") as f:
            self.data = pickle.load(f)

    def test_autoencoder(self):
        input_size = 375
        hidden_size = 64
        contraction_coefficient = 1e-3

        # Create an instance of the contractive autoencoder
        autoencoder = ContractiveAutoencoder(input_size, hidden_size, contraction_coefficient)

        # Iterate over the tensors in the data and perform forward pass
        for tensor in self.data:
            # Ensure the tensor has the correct shape [1, 125, 3]
            tensor = tensor.reshape(375)


            # Convert the tensor to torch.Tensor and pass it through the autoencoder

            input_tensor = torch.Tensor(tensor.view(-1))
            encoded, decoded = autoencoder(input_tensor)

            # Ensure the encoded and decoded tensors have the correct shapes
            self.assertEqual((1, 64), (1, hidden_size))
            self.assertEqual(decoded.shape, (1, input_size))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
