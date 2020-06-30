import numpy as np

import torch


class Prior:

    def __init__(self, dim, n_params, C):
        self.dim = dim
        if not isinstance(C, torch.Tensor):
            raise ValueError('Parameter `C` should be a tensor of length `n_params`.')
        self.n_params = len(C)
        self.C = C

    def logprior(self, z, z2):
        """
        Log Prior
        """
        raise NotImplementedError('Must be implemented in child class')

    def opt_hyper(self, z, z2):
        """
        Optimal regularization weights for the current value of z
        """
        raise NotImplementedError('Must be implemented in child class')


class GaussianPrior(Prior):

    def logprior(self, z, z2):
        """
        Log Prior
        """
        return -torch.sum((z ** 2) / self.C[:-1]), -torch.sum((z2 ** 2) / self.C[:-1])

    def opt_hyper(self, z, z2):
        """
        Optimal regularization weights for the current value of z
        """
        return 2 * z ** 2, 2 * z2 **2
