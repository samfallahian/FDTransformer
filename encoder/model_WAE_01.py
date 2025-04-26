import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

original_dim = 375
latent_dim = 47
#MODEL_FILEPATH_TEMPLATE = "/Users/kkreth/PycharmProjects/cgan/saved_models/model_at_epoch_{}.pth"

class WAE(nn.Module):
    def __init__(self):
        super(WAE, self).__init__()

        # Define the sizes for the hidden layers
        hidden_dim1 = 250  # Modify as needed
        hidden_dim2 = 150  # Modify as needed
        hidden_dim3 = 100  # Modify as needed

        # Encoder
        self.fc1 = nn.Linear(original_dim, hidden_dim1)
        self.elu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.elu2 = nn.ELU()
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.elu3 = nn.ELU()
        self.fc4 = nn.Linear(hidden_dim3, latent_dim)
        # Additional Tanh activation function layer
        self.tanh = nn.Tanh()  # New layer

        # Decoder
        self.fc5 = nn.Linear(latent_dim, hidden_dim3)
        self.elu4 = nn.ELU()
        self.fc6 = nn.Linear(hidden_dim3, hidden_dim2)
        self.elu5 = nn.ELU()
        self.fc7 = nn.Linear(hidden_dim2, hidden_dim1)
        self.elu6 = nn.ReLU()
        self.fc8 = nn.Linear(hidden_dim1, original_dim)


    def decode(self, z):
        h1 = self.elu4(self.fc5(z))
        h2 = self.elu5(self.fc6(h1))
        h3 = self.elu6(self.fc7(h2))
        return self.fc8(h3)

    def encode(self, x):
        h1 = self.elu1(self.fc1(x))
        h2 = self.elu2(self.fc2(h1))
        h3 = self.elu3(self.fc3(h2))
        return self.tanh(self.fc4(h3))

    def forward(self, x):
        mu = self.encode(x.view(-1, original_dim))
        return self.decode(mu), mu

    '''
    This static function computes a pessimistic mean squared error (MSE) between two sets of data points y_true and y_pred.
    y_true would be the ground truth or target outputs that the model should ideally produce.
    y_pred would be the outputs that the model has actually produced.
    Here is a step-by-step breakdown of what this function does:
    squared_errors = (y_true - y_pred) ** 2: It first computes the squared errors between the true and predicted values. This is done element-wise, which means each corresponding element in y_true and y_pred tensors is subtracted and then squared.
    max_squared_errors = torch.max(squared_errors, dim=1).values: It then finds the maximum squared error of each instance in the batch. The dim=1 argument is indicating that the maximum should be computed across the second dimension of the tensor (columns in this case, as Python uses zero-based indexing).
    return torch.mean(max_squared_errors): Finally, it returns the mean of these maximum squared errors. Essentially, this is computing the average of the worst case (maximum) errors in each instance of the batch.
    The reason it might be called "pessimistic" MSE is because it's not calculating the mean squared error on every element, but rather considering only the maximum (worst case) error for each instance in the batch. Thus, giving you a "pessimistic" view of how your model is performing.
    The method is defined as a static method (as denoted by @staticmethod). This means it belongs to the class, not instances of the class, and as such can't modify the class state, but only perform computations on the input parameters.
    Please, let me know if you have any additional questions!
    '''
    @staticmethod
    def pessimistic_mse(y_true, y_pred):
        # Assuming y_true and y_pred are of shape [batch_size, 3] where columns represent x, y, z
        squared_errors = (y_true - y_pred) ** 2  # Element-wise squared errors
        max_squared_errors = torch.max(squared_errors, dim=1).values  # Max squared error per instance
        return torch.max(max_squared_errors)  # Mean of max squared errors

    @staticmethod
    def _huber_loss_formula(abs_errors, squared_errors, deltas):
        return torch.where(abs_errors <= deltas, 0.5 * squared_errors, deltas * (abs_errors - 0.5 * deltas))

    @staticmethod
    def custom_huber_loss(y_true, y_pred):
        # Calculate absolute errors
        abs_errors = torch.abs(y_true - y_pred)

        # Find the smallest absolute error for each instance, to use as delta
        deltas, _ = torch.min(abs_errors, dim=1, keepdim=True)

        # Calculate squared errors
        squared_errors = (y_true - y_pred) ** 2

        # Huber loss calculation with dynamic delta
        huber_loss = WAE._huber_loss_formula(abs_errors, squared_errors, deltas)

        # Summing up the Huber loss for each dimension, and then taking the max across instances
        instance_losses = torch.sum(huber_loss, dim=1)
        return torch.max(instance_losses)

    def log_cosh_loss_formula(prediction_error):
        return torch.mean(torch.log(torch.cosh(prediction_error)))


    @staticmethod
    def log_cosh_loss(y_true, y_pred):
        prediction_error = y_true - y_pred
        scaled_error = 10 * (y_true - y_pred)  # Scaling factor depends on dataset
        prediction_error = scaled_error
        log_cosh_loss = torch.log(torch.cosh(prediction_error))
        instance_losses = torch.sum(log_cosh_loss, dim=1)
        return torch.max(instance_losses)

    def loss_function(self, recon_x, x, mu):
        recon_x_grouped = recon_x.view(-1, int(recon_x.nelement()/3), 3)
        x_grouped = x.view(-1, int(x.nelement()/3), 3)

        '''
        # In case shapes are not the expected ones, print them out.
        if recon_x_grouped.shape[0] < 63 or x_grouped.shape[0] < 63:
            print('recon_x_grouped shape:', recon_x_grouped.shape)
            print('x_grouped shape:', x_grouped.shape)
        '''
        # Reconstruction loss
        recon_loss = self.log_cosh_loss(recon_x_grouped, x_grouped)

        # MMD loss: enforcing prior = posterior in the latent space
        true_samples = torch.randn(mu.shape)
        mmd_loss = self._compute_mmd(mu, true_samples)

        # Previous code was assuming there are 63 examples / batch_size
        # Changed it to compute triplet loss tensor-wide
        # OR for the 63rd triplet of a random example if there are more than 63

        # Instead of [62] this should probably be something like [random_index, 62]
        # 'random_index' is choosing a random sample in the batch
        random_index = torch.randint(len(x_grouped), size=(1,)).item()
        triplet_63_loss = F.mse_loss(recon_x_grouped[random_index, 62], x_grouped[random_index, 62])



        return recon_loss + mmd_loss + triplet_63_loss, recon_loss, mmd_loss, triplet_63_loss

    def _compute_kernel(self, x, y):
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]
        tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
        tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)
        return torch.exp(-torch.mean((tiled_x - tiled_y)**2, dim=2) / dim * 2.0)

    def _compute_mmd(self, x, y):
        y = y.to(x.device)  # ensure both x and y are in the same device
        x_kernel = self._compute_kernel(x, x)
        y_kernel = self._compute_kernel(y, y)
        xy_kernel = self._compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
