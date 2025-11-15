import torch
from scipy.stats import norm
from .base import UnivariateTest


class Moments(UnivariateTest):
    """
    Moments-based test for standard normality N(0,1).
    
    Tests normality by comparing empirical moments against theoretical
    moments of the standard normal distribution.
    
    Parameters
    ----------
    k_max : int, default=4
        Maximum moment order to test (must be even). Tests moments from
        order 2 to k_max in increments of 2.
    """
    
    def __init__(self, k_max: int = 4):
        super().__init__(sorted=True)
        self.k_max = k_max
        moments = []
        for i in range(2, k_max + 1, 2):
            moment_val = norm(loc=0, scale=1).moment(i)
            moments.append(moment_val)
        self.register_buffer(f"moments", torch.Tensor(moments).unsqueeze(1))
        self.register_buffer(f"weights", torch.arange(2, self.k_max + 1).neg().exp())

    def forward(self, x):
        x = self.prepare_data(x)
        k = torch.arange(2, self.k_max + 1, device=x.device, dtype=x.dtype).view(
            -1, 1, 1
        )
        m1 = self.dist_mean(x.mean(0)).abs_()
        if self.k_max >= 2:
            xpow = self.dist_mean((x**k).mean(1))
            xpow[::2].sub_(self.moments)
            m2 = xpow.abs_().T.matmul(self.weights)
            return m1.add_(m2)  # dist_mean already handles DDP averaging
        return m1  # dist_mean already handles DDP averaging
