import torch
import torch.nn as nn


class CustomLoss(nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x, x_hat):
        cross_entropy = self.cross_entropy_loss(x_hat, x)
        # here add custom loss and add it to final loss
        final_loss = cross_entropy

        return final_loss
