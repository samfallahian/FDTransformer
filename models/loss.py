import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.neighbors import KernelDensity
from scipy.stats import wasserstein_distance
from utils import helpers


class CustomLoss(nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()
        config = helpers.Config()
        self.cfg_cgan = config.from_json("training").cgan
        self.cfg_cae = config.from_json("training").cae
        self.bec_log = nn.BCEWithLogitsLoss()
        self.bec = nn.BCELoss()
        self.mse = nn.MSELoss()

    def cgan_loss(self, generated_label, real_label, generated_data, real_data, is_generator=True):
        if self.cfg_cgan.is_critic:
            loss = self.bec_log(generated_label, real_label)
        else:
            loss = self.bec(generated_label, real_label)
        # here add custom loss and add it to final loss
        if is_generator:
            wass_loss = self.wass_distance(real_data, generated_data)
            final_loss = loss + wass_loss
        else:
            final_loss = loss
        return final_loss

    def cae_loss(self, W, data, decoded, encoded):
        mse = self.mse(decoded, data)
        dh = encoded * (1 - encoded)  # Hadamard product produces size N_batch x N_hidden
        # Sum through the input dimension to improve efficiency, as suggested in #1
        w_sum = torch.sum(Variable(W) ** 2, dim=1)
        # unsqueeze to avoid issues with torch.mv
        w_sum = w_sum.unsqueeze(1)  # shape N_hidden x 1
        contractive_loss = torch.sum(torch.mm(dh ** 2, w_sum), 0)
        final_loss = mse + contractive_loss.mul(self.cfg_cae.contractive_coef)
        return final_loss , mse

    def wass_distance(self, real, generated):
        """Use KernelDensity to estimate the probability density function (PDF) of the real and generated data. Then,
        we can use the wasserstein_distance function from the to calculate the Wasserstein distance between the two
        estimated PDFs."""
        real = real.cpu()
        generated = generated.cpu()
        real = real.detach().numpy()
        generated = generated.detach().numpy()

        # Estimate PDFs of real and generated data
        kde_real = KernelDensity(kernel='gaussian', bandwidth=self.cfg_cgan.kd_band_width).fit(real)
        kde_fake = KernelDensity(kernel='gaussian', bandwidth=self.cfg_cgan.kd_band_width).fit(generated)
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
