import torch
import numpy as np


class HawkesModel:

    def __init__(self, excitation, verbose=False, device='cpu'):
        """
        Initialize the model

        Arguments:
        ----------
        prior : Prior
            Prior object
        excitation: excitation
            Excitation object
        """
        self.excitation = excitation
        self.n_jumps = None
        self.dim = None
        self.n_params = None
        self.n_var_params = None
        self._fitted = False
        self.verbose = verbose
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.end_time = {}
        self._cache = {}
        self._cache_integral = {}


    #def set_data(self, events, end_time):
    def set_data(self, events):
        """
        Set the data for the model
        """
        assert isinstance(events[0][0], torch.Tensor)
        self.events_samples = len(events)
        self.dim = len(events[0])
        self.n_params = self.dim * (self.dim + 1)
        # n_var_params = 2 * len(events[0]) * (len(events[0]) + 1)
        self.n_var_params = 2 * self.n_params
        self.n_jumps = len(events) * sum(map(len, events[0]))
        for i in range(self.events_samples):
            # zc_debut = max([max(num) for num in events[i] if len(num) > 0])
            # print("i is :",i)
            self.end_time[i] = max([max(num) for num in events[i] if len(num) > 0])
        self.events = events
        if not self._fitted:
            self._init_cache()
        self._fitted = True

    def _init_cache(self):

        for s in range(self.events_samples):
            self._cache[s] = [torch.zeros(
                (self.dim, self.excitation.M, len(events_i)), dtype=torch.float64, device=self.device)
                for events_i in self.events[s]]
            # print(self._cache)
            for i in range(self.dim):
                for j in range(self.dim):
                    if self.verbose:
                        print(f"\rInitialize cache {i * self.dim + j + 1}/{self.dim ** 2}     ", end='')
                    id_end = np.searchsorted(
                        self.events[s][j].cpu().numpy(),
                        self.events[s][i].cpu().numpy())
                    id_start = np.searchsorted(
                        self.events[s][j].cpu().numpy(),
                        self.events[s][i].cpu().numpy() - self.excitation.cut_off)
                    for k, time_i in enumerate(self.events[s][i]):
                        t_ij = time_i - self.events[s][j][id_start[k]:id_end[k]]
                        kappas = self.excitation.call(t_ij).sum(-1)  # (M)
                        self._cache[s][i][j, :, k] = kappas
            if self.verbose:
                print()

            self._cache_integral[s] = torch.zeros((self.dim, self.excitation.M),
                                                  dtype=torch.float64, device=self.device)
            for j in range(self.dim):
                t_diff = self.end_time[s] - self.events[s][j]
                integ_excit = self.excitation.callIntegral(t_diff).sum(-1)  # (M)
                self._cache_integral[s][j, :] = integ_excit

    def log_likelihood(self, mu, W, distb):
        """
        Log likelihood of Hawkes Process for the given parameters mu and W

        Arguments:
        ----------
        mu : torch.Tensor
            (dim x 1)
            Base intensities
        W : torch.Tensor
            (dim x dim x M) --> M is for the number of different excitation functions
            The weight matrix.
        """
        log_like = torch.zeros(self.events_samples)

        for s in range(self.events_samples):
            intens = [0.0] * self.dim
            intens_every = [0.0] * self.dim

            for i in range(self.dim):
                #print('distb is ',distb)
                intens_every[i] = mu[i] + (W[i].unsqueeze(2) * self._cache[s][i]).sum(0).sum(0) - distb
                intens[i] = torch.log(intens_every[i])
                log_like[s] += intens[i].sum()
                # print(intens[i])
            log_like[s] -= self._integral_intensity(mu, W, s)
        log_like_sum = log_like.sum(0)

        return log_like_sum

    def _integral_intensity(self, mu, W, s):
            """
            Integral of intensity function

            Argument:
            ---------
            node_i: int
                Node id
            """
            integ_ints = (W * self._cache_integral[s].unsqueeze(0)).sum(1).sum(1)
            integ_ints += self.end_time[s] * mu
            return integ_ints.sum()
