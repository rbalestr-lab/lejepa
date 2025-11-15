import torch
from .base import UnivariateTest


class ShapiroWilk(UnivariateTest):
    """
    Shapiro-Wilk test for standard normality N(0,1).

    A correlation-based goodness-of-fit test that measures how well ordered
    sample values match the expected order statistics from a standard normal
    distribution. This variant assumes the data has been standardized.

    The test statistic is computed as:
        T = 1 - |ρ(x_(i), m_i)|

    where:
        - x_(i) are the sorted (standardized) sample values
        - m_i = Φ^(-1)(p_i) are expected order statistics from N(0,1)
        - p_i are plotting positions (depends on expectation_mode)
        - ρ is correlation, computed via cosine similarity on centered data

    Parameters
    ----------
    expectation_mode : str, default='elfving'
        Plotting position formula for expected order statistics:
        - 'elfving': p_i = (i - π/8) / (n + 1/4)
        - 'blom': p_i = (i - 3/8) / (n + 1/4)  [Blom's approximation]
        - 'rahman': p_i = i / (n + 1)

    covariance_mode : str, default='shapiro_francia'
        Method for computing correlation weights:
        - 'shapiro_francia': Simplified version assuming independent order
          statistics (V = I). Equivalent to Shapiro-Francia test.
        - 'rahman': Uses tridiagonal approximation to the covariance matrix
          of order statistics for improved accuracy.

    eps : float, default=1e-5
        Small constant for numerical stability in cosine similarity

    sorted : bool, default=False
        Whether input data is pre-sorted

    Returns
    -------
    T : torch.Tensor
        Test statistic in [0, 1]. Values closer to 0 indicate better fit to
        N(0,1). Higher values indicate departure from standard normality.
        Shape matches the batch dimensions of the input.

    Notes
    -----
    - Input data should be standardized by `prepare_data()` before testing
    - The 'shapiro_francia' mode is computationally simpler but slightly less
      powerful than 'rahman' mode
    - This formulation (1 - |ρ|) is inverted from traditional W statistic
    - Works best for sample sizes 3 ≤ n ≤ 5000
    - The weights are cached and recomputed only when sample size changes

    References
    ----------
    .. [1] Shapiro, S. S., & Wilk, M. B. (1965). "An analysis of variance test
           for normality (complete samples)". Biometrika, 52(3/4), 591-611.
    .. [2] Shapiro, S. S., & Francia, R. S. (1972). "An approximate analysis
           of variance test for normality". Journal of the American Statistical
           Association, 67(337), 215-216.
    .. [3] Blom, G. (1958). "Statistical Estimates and Transformed Beta-Variables".
           Wiley, New York.
    .. [4] Royston, P. (1992). "Approximating the Shapiro-Wilk W-test for
           non-normality". Statistics and Computing, 2(3), 117-119.

    Examples
    --------
    >>> # Test if standardized samples follow N(0,1)
    >>> x = torch.randn(100)
    >>> test = ShapiroWilk(expectation_mode='blom',
    ...                     covariance_mode='shapiro_francia')
    >>> statistic = test(x)
    >>> # statistic ≈ 0 indicates good fit to N(0,1)
    >>> # statistic > threshold indicates rejection of normality
    """

    def __init__(
        self,
        expectation_mode: str = "elfving",
        covariance_mode: str = "shapiro_francia",
        eps: float = 1e-5,
        sorted: bool = False,
    ):
        super().__init__(eps=eps, sorted=sorted)
        self.expectation_mode = expectation_mode
        self.covariance_mode = covariance_mode
        self._k = None

    def forward(self, x):
        s = self.prepare_data(x)
        if self._k is None or self._k.size(0) != x.size(0):
            with torch.no_grad():
                self._k = self.get_shapiro_weights(
                    x.size(0),
                    expectation_mode=self.expectation_mode,
                    covariance_mode=self.covariance_mode,
                    device=x.device,
                )
        extra_dims = tuple([1] * (x.ndim - 1))
        k = self._k.view(x.size(0), *extra_dims)
        return (
            1 - torch.nn.functional.cosine_similarity(k, s, dim=0, eps=self.eps).abs()
        )

    @staticmethod
    def get_shapiro_weights(
        N,
        expectation_mode="blom",  # Fixed typo
        covariance_mode="shapiro_francia",
        device="cpu",
    ):
        """
        Compute Shapiro-Wilk weights for correlation with order statistics.

        Parameters
        ----------
        N : int
            Sample size
        expectation_mode : str
            Plotting position formula
        covariance_mode : str
            Covariance approximation method
        device : str or torch.device
            Device for computation

        Returns
        -------
        a : torch.Tensor
            Normalized weights of shape (N,)
        """
        g = torch.distributions.normal.Normal(0, 1)
        grid = torch.arange(1, N + 1, dtype=torch.float, device=device)

        # Compute plotting positions
        if expectation_mode == "elfving":
            pi = grid.sub_(torch.pi / 8).div_(N + 1 / 4)
        elif expectation_mode == "blom":  # Fixed name
            pi = grid.sub_(3 / 8).div_(N + 1 / 4)
        elif expectation_mode == "rahman":
            pi = grid.div_(N + 1)
        else:
            raise ValueError(f"Unknown expectation_mode: {expectation_mode}")

        # Expected order statistics from N(0,1)
        m = g.icdf(pi)

        # Apply covariance structure
        if covariance_mode == "shapiro_francia":
            # Assume independence: V = I, so a ∝ m
            a = m
        elif covariance_mode == "rahman":
            # Tridiagonal approximation: a = V^(-1) m
            phi = g.log_prob(m).exp_()
            a = phi.square().mul_(m).mul_(2)
            cross = phi[1:] * phi[:-1]
            a[:-1] -= m[1:] * cross
            a[1:] -= m[:-1] * cross
        else:
            raise ValueError(f"Unknown covariance_mode: {covariance_mode}")

        # Normalize to unit length (required for cosine similarity interpretation)
        return torch.nn.functional.normalize(a, p=2, dim=0)
