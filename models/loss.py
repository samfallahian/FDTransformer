import torch
import torch.nn as nn
from sklearn.neighbors import KernelDensity
from scipy.stats import wasserstein_distance
from utils import helpers


class CustomLoss(nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss = nn.BCELoss()
        config = helpers.Config()
        self.cfg = config.from_json("training")

    def forward(self, generated_label,  real_label, generated_data, real_data):
        loss = self.loss(generated_label, real_label)
        # here add custom loss and add it to final loss
        wass_loss = self.wass_distance(real_data, generated_data)
        final_loss = loss+wass_loss

        return final_loss

    def wass_distance(self, real, generated):
        """Use KernelDensity to estimate the probability density function (PDF) of the real and generated data. Then,
        we can use the wasserstein_distance function from the to calculate the Wasserstein distance between the two
        estimated PDFs."""
        real = real.cpu()
        generated = generated.cpu()
        real = real.detach().numpy()
        generated = generated.detach().numpy()

        # Estimate PDFs of real and generated data
        kde_real = KernelDensity(kernel='gaussian', bandwidth=self.cfg.kd_band_width).fit(real)
        kde_fake = KernelDensity(kernel='gaussian', bandwidth=self.cfg.kd_band_width).fit(generated)
        # Calculate Wasserstein distance between real and generated data
        real_pdf = kde_real.score_samples(real)
        fake_pdf = kde_fake.score_samples(generated)
        wass_distance = wasserstein_distance(real_pdf, fake_pdf)
        return wass_distance

    # @staticmethod
    # def kl_divergence(real, generated):
    #     real_logprobs = F.log_softmax(discriminator(real), dim=1)
    #     fake_logprobs = F.log_softmax(discriminator(generated), dim=1)
    #     kl_divergence = F.kl_div(fake_logprobs, real_logprobs, reduction='batchmean')
    #     return intersection / union

