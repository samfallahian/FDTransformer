import torch
import torch.nn as nn
from sklearn.neighbors import KernelDensity
from scipy.stats import wasserstein_distance
from utils import helpers


class CustomLoss(nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()
        config = helpers.Config()
        self.cfg = config.from_json("training")
        self.bec_log = nn.BCEWithLogitsLoss()
        self.bec = nn.BCELoss()
        self.mse = nn.MSELoss()

    def cgan_loss(self, generated_label, real_label, generated_data, real_data, is_generator=True):
        if self.cfg.is_critic:
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

    def cae_loss(self, encoded, decoded, data, encoder_model):
        mse_loss = self.mse(decoded,data)
        jacobian = torch.autograd.functional.jacobian(encoder_model, data)
        jacobian_norm = torch.norm(jacobian[0], dim=(0, 2))
        encoded_reshaped = encoded.view(-1, 1, encoded.shape[1])
        encoded_norm = torch.norm(encoded_reshaped, dim=(2))
        contractive_loss = torch.mean(jacobian_norm ** 2 * encoded_norm ** 2)
        final_loss = mse_loss + self.cfg.contractive_coef * contractive_loss
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
