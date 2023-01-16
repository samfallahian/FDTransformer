import torch
import torch.nn as nn


class CustomLoss(nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, x, x_hat):
        mse_loss = self.loss(x_hat, x)
        # here add custom loss and add it to final loss
        final_loss = mse_loss

        return final_loss
