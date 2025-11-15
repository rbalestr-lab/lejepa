import torch
from .base import MultivariateTest


class BHEP_M(MultivariateTest):
    """
    Modified BHEP (Baringhaus-Henze-Epps-Pulley) test for multivariate normality.
    
    This is a variant of the BHEP test that uses a modified kernel with a beta
    parameter to control sensitivity. The test statistic measures departure from
    standard multivariate normality N(0, I).
    
    Parameters
    ----------
    dim : int
        Dimensionality of the data
    beta : float, default=10
        Kernel parameter controlling sensitivity. Must be > 2.
        Higher values make the test more sensitive to departures from normality.
    
    Returns
    -------
    torch.Tensor
        Test statistic (scalar). Higher values indicate greater departure from
        multivariate normality.
    
    Performance Note
    ----------------
    This test has O(N²) computational complexity where N is the number of samples.
    For large datasets (N > 1000), consider using slicing-based tests instead,
    or subsample your data.
    
    Notes
    -----
    The test computes a difference between two kernel density estimates using
    an exponential kernel weighted by the beta parameter.
    """
    
    def __init__(self, dim, beta=10):
        super().__init__()
        self.dim = dim
        assert beta > 2, "beta must be > 2"
        self.beta = beta

    def forward(self, x):
        """
        Compute the BHEP_M test statistic.
        
        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            Input data of shape (N, D) where N is number of samples and
            D is dimensionality (must match self.dim)
        
        Returns
        -------
        torch.Tensor
            Test statistic (scalar)
        """
        x = self.prepare_data(x)
        _, D = x.shape
        norms = x.square().sum(1)
        pair_sim = 2 * x @ x.T + norms + norms.unsqueeze(1)
        lhs = (
            (1 / self.beta ** (D / 2))
            * torch.exp(pair_sim.div(4 * self.beta)).sum()
            / x.size(0)
        )
        rhs = (2 / (self.beta - 0.5) ** (D / 2)) * torch.exp(
            norms / (4 * self.beta - 2)
        ).sum()
        # Note: Constant term omitted for centered statistic
        # cst = N / ((self.beta - 1) ** (D / 2))
        return lhs - rhs  # + cst

