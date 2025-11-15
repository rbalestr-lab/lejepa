import torch
from .base import MultivariateTest


class HV(MultivariateTest):
    """
    HV (Henze-Visagie) test statistic for multivariate normality.

    Computes a kernel-based test statistic using exponential kernels with
    a gamma parameter to control sensitivity.

    Args:
        gamma: Bandwidth parameter for the kernel (must be > 0).
               Controls the sensitivity of the test.

    Performance Note:
        This test has O(N²) computational complexity where N is the number of samples.
        For large datasets (N > 1000), consider using slicing-based tests instead,
        or subsample your data.
    """

    def __init__(self, gamma=1):
        super().__init__()
        assert gamma > 0
        self.gamma = gamma

    def forward(self, x):
        x = self.prepare_data(x)
        N, D = x.shape
        norms = x.square().sum(1)
        pair_sim = 2 * x @ x.T + norms + norms.unsqueeze(1)
        lhs = torch.exp(pair_sim.div(4 * self.gamma))
        rhs = (
            x @ x.T
            - pair_sim / (2 * self.gamma)
            + D / (2 * self.gamma)
            + pair_sim / (4 * self.gamma**2)
        )
        # Note: Constant term omitted for centered statistic
        # cst = N/((beta-1)**(D/2))
        return (lhs * rhs).sum() / N
