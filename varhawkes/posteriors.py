import math
import numpy as np
from scipy.stats import norm, lognorm, truncnorm

import torch


class Posterior:

    def sample_epsilon(self, size):
        """
        Sample an array of epsilons from the normal distribution, with size
        (n_samples, n_weights, n_params)
        """
        raise NotImplementedError('Must be implemented in child class')

    def g(self, eps, alpha, beta, alpa2, beta2):
        """
        Reparamaterization of the approximate log-normal posterior function
        """
        raise NotImplementedError('Must be implemented in child class')

    def logpdf(self, eps, alpha, beta, alpa2, beta2):
        """
        Log Posterior
        """
        raise NotImplementedError('Must be implemented in child class')


class LogNormalPosterior(Posterior):

    def __init__(self, device = 'cpu'):
        self.norm = torch.distributions.Normal(loc=0.0, scale=1.0)
        self.device = 'cuda' if torch.cuda.is_available() and device=='cuda' else 'cpu'

    def sample_epsilon(self, size):
        """
        Sample an array of epsilons from the normal distribution, with size
        (n_samples, n_weights, n_params)
        """
        eps_err = torch.randn(size, dtype=torch.float64, device = self.device, requires_grad=False)
        return eps_err

    def g(self, eps, alpha, beta, alpha2, beta2):
        """
        Reparamaterization of the approximate log-normal posterior function
        """
        return torch.exp(alpha + eps[:-1] * beta.exp()), torch.exp(alpha2 + eps[-1] * beta2.exp())

    def logpdf(self, eps, alpha, beta, alpha2, beta2):
        z,z2 = self.g(eps, alpha, beta, alpha2, beta2)
        z_minus_v = z - alpha
        z_minus_v2 = z2 - alpha2
        sigma = beta.exp()
        sigma2 = beta2.exp()
        log_q_phi = z_minus_v * torch.exp(- z_minus_v**2/ 2*torch.exp(2*sigma)) / (math.sqrt(2*np.pi)*torch.exp(3*sigma))
        log_q_phi2 = z_minus_v2 * torch.exp(- z_minus_v2 ** 2 / 2 * torch.exp(2 * sigma2)) / (math.sqrt(2 * np.pi) * torch.exp(3 * sigma2))

        return torch.sum(log_q_phi), torch.sum(log_q_phi2)

    def mode(self, alpha, beta, alpha2, beta2):
        return torch.exp(alpha - beta.exp() ** 2), alpha2, beta2.exp()
