"""
Model 07: Vector Quantized VAE (VQ-VAE)
Uses a discrete latent space with learnable codebook vectors.
Loss: Reconstruction + Codebook + Commitment losses
"""
import torch
from torch import nn
import torch.nn.functional as F

original_dim = 375
latent_dim = 47
num_embeddings = 512  # Codebook size

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs):
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances to codebook vectors
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Get nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss

class VQVAE(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(VQVAE, self).__init__()

        hidden_dim1 = 250
        hidden_dim2 = 150
        hidden_dim3 = 100

        # Encoder
        self.fc1 = nn.Linear(original_dim, hidden_dim1)
        self.elu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(hidden_dim3, latent_dim)

        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, latent_dim)

        # Decoder
        self.fc5 = nn.Linear(latent_dim, hidden_dim3)
        self.elu4 = nn.ELU()
        self.dropout4 = nn.Dropout(dropout_rate)

        self.fc6 = nn.Linear(hidden_dim3, hidden_dim2)
        self.elu5 = nn.ELU()
        self.dropout5 = nn.Dropout(dropout_rate)

        self.fc7 = nn.Linear(hidden_dim2, hidden_dim1)
        self.elu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(dropout_rate)

        self.fc8 = nn.Linear(hidden_dim1, original_dim)

    def encode(self, x):
        h1 = self.dropout1(self.elu1(self.fc1(x)))
        h2 = self.dropout2(self.elu2(self.fc2(h1)))
        h3 = self.dropout3(self.elu3(self.fc3(h2)))
        return self.fc4(h3)

    def decode(self, z):
        h1 = self.dropout4(self.elu4(self.fc5(z)))
        h2 = self.dropout5(self.elu5(self.fc6(h1)))
        h3 = self.dropout6(self.elu6(self.fc7(h2)))
        return self.fc8(h3)

    def forward(self, x):
        x = x.view(-1, original_dim)
        z_e = self.encode(x)
        z_q, vq_loss = self.vq(z_e)
        return self.decode(z_q), z_q, vq_loss

    def loss_function(self, recon_x, x, z, vq_loss):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x.view(-1, original_dim), reduction='mean')

        total_loss = recon_loss + vq_loss
        return total_loss, recon_loss, vq_loss, torch.tensor(0.0)
