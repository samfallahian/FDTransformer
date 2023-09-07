import torch
import pytest
#from ConvolutionalAutoencoder import ConvolutionalAutoencoder
import sys
sys.path.append("/Users/kkreth/PycharmProjects/cgan/standalone/")
from HybrdidAutoencoder import HybrdidAutoencoder

class TestConvolutionalAutoencoder:
    @pytest.fixture
    def model(self):
        """
        PyTest fixture that creates a ConvolutionalAutoencoder instance.
        """
        return HybrdidAutoencoder(latent_size=(8, 6))

    @pytest.fixture
    def saved_state_path(self):
        """
        PyTest fixture that provides the path to the saved model state.
        """
        return "/Users/kkreth/PycharmProjects/cgan/standalone/saved_models/checkpoint_300.pth"

    def test_encoding_and_decoding(self, model):
        """
        Test if the encode function produces an output with the correct size
        and if the decode function reconstructs an output with the expected size.
        """
        # Generate a random tensor with size (1, 3, 125) to simulate the input data
        random_input = torch.randn((1, 3, 125))

        # Encode the random input
        encoded_output = model.encode(random_input)

        # Check that the size of the encoded output is (1, 8, 6)
        assert encoded_output.size() == (1, 8, 6), f"Encoding failed. Expected size (1, 8, 6), but got {encoded_output.size()}."

        # Decode the encoded output
        decoded_output = model.decode(encoded_output)

        # Check that the size of the decoded output is (1, 3, 125)
        assert decoded_output.size() == (1, 3, 125), f"Decoding failed. Expected size (1, 3, 125), but got {decoded_output.size()}."

    def test_model_loading(self, model, saved_state_path):
        """
        Test if the model can load a saved state and still produce the expected output sizes.
        """
        # Load the saved state into the model
        checkpoint = torch.load(saved_state_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Generate a random tensor with size (1, 3, 125) to simulate the input data
        random_input = torch.randn((1, 3, 125))

        # Encode the random input
        encoded_output = model.encode(random_input)

        # Check that the size of the encoded output is (1, 8, 6)
        assert encoded_output.size() == (1, 8,
                                         6), f"Encoding failed after loading saved state. Expected size (1, 8, 6), but got {encoded_output.size()}."

        # Decode the encoded output
        decoded_output = model.decode(encoded_output)

        # Check that the size of the decoded output is (1, 3, 125)
        assert decoded_output.size() == (1, 3,
                                         125), f"Decoding failed after loading saved state. Expected size (1, 3, 125), but got {decoded_output.size()}."

