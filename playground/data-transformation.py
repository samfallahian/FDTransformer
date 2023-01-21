import torch


# Define the Box-Cox transformation
def box_cox(x, lambda_):
    return (x.pow(lambda_) - 1) / lambda_


# Define the inverse Box-Cox transformation
def inv_box_cox(y, lambda_):
    return (y * lambda_ + 1).pow


# Define the rank transform function
def rank_transform(x):
    sorted_x, indices = torch.sort(x)
    ranks = torch.argsort(indices, dim=0)
    return ranks


# Define the inverse rank transform function
def inv_rank_transform(ranks):
    sorted_ranks, indices = torch.sort(ranks)
    x = torch.argsort(indices, dim=0)
    return x


# Apply the rank transform to a column of your data
data = torch.randn(100, 1)
data_rank = rank_transform(data)

# Apply the inverse rank transform to the ranked data
data_original = inv_rank_transform(data_rank)

"""
It's important to note that rank transform is not unique, so you might get different results depending on how you handle the tie values. It's also important to keep the same data type between original and transformed data.
You can use this rank transform on your tabular data before feed it into VAE-CGAN, this will help to make the distribution more unimodal and make it easier for the model to learn effectively.
"""
## Embeding
import torch
import torch.nn as nn


# Define the embedding layer
class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


# Instantiate the embedding layer
num_classes = 5
embedding_dim = 2
embedding_layer = EmbeddingLayer(num_classes, embedding_dim)

# Apply the embedding to a column of your data
data = torch.randint(0, num_classes, (100, 1))
data_embedded = embedding_layer(data)

"""
In this example, the EmbeddingLayer creates an embedding matrix that maps the integers in the input data to a dense, low-dimensional representation in a embedding_dim-dimensional space. The nn.Embedding module in Pytorch takes care of the embedding matrix initialization.

You can use this embedding method on your categorical data before feeding it into a VAE-CGAN. This will help to create a dense low-dimensional representation of categorical data, which can be easier for the model to learn effectively.

You can also use pre-trained embeddings to initialize the embedding matrix, which can be useful when you have limited data or you have similar categorical variables across different datasets.
"""
