import numpy as np
import torch
from .base import MultivariateTest
from typing import Union


class BHEP(MultivariateTest):
    """
    Beta-Henze Energy-based Projection (BHEP) test statistic.

    Computes the BHEP test statistic for multivariate normality testing
    using a Gaussian kernel with bandwidth parameter beta.

    Args:
        dim: Dimensionality of the input data
        beta: Bandwidth parameter for the Gaussian kernel (must be > 0)
              Smaller values make the test more sensitive to local deviations

    Performance Note:
        This test has O(N²) computational complexity where N is the number of samples.
        For large datasets (N > 1000), consider using slicing-based tests instead,
        or subsample your data.

    Reference:
        Henze, N., & Wagner, T. (1997). A new approach to the BHEP tests for
        multivariate normality. Journal of Multivariate Analysis, 62(1), 1-23.
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()

        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")

        self.beta = beta

    def forward(
        self, x: Union[torch.Tensor, np.ndarray], beta: float = None
    ) -> torch.Tensor:
        """
        Compute BHEP test statistic.

        Args:
            x: Input data of shape (N, D) where N is number of samples
               and D is dimensionality
            beta: Optional bandwidth parameter. If provided, overrides self.beta
                for this forward pass only.

        Returns:
            BHEP test statistic as scalar tensor. Lower values indicate
            better fit to multivariate normal distribution.

        Raises:
            ValueError: If input data is empty or malformed
        """
        x = self.prepare_data(x)
        N, D = x.shape

        # Use provided beta or fall back to instance beta
        beta_val = beta if beta is not None else self.beta

        # Validate input
        if N == 0:
            raise ValueError("Input data cannot be empty")

        # Precompute constants
        beta_squared = beta_val**2

        # Compute squared norms: ||x_i||^2
        squared_norms = x.square().sum(dim=1)  # Shape: (N,)

        # Compute pairwise squared distances: ||x_i - x_j||^2
        # Using: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i^T x_j
        pairwise_distances = (
            -2 * x @ x.T 
            + squared_norms.unsqueeze(1) 
            + squared_norms.unsqueeze(0)
        )

        # Left-hand side: (1/N^2) * sum_{i,j} exp(-beta^2/2 * ||x_i - x_j||^2)
        lhs = torch.exp(pairwise_distances * (-beta_squared / 2)).sum() / (N**2)

        # Right-hand side: 2 / (N * (1 + beta^2)^(D/2)) * sum_i exp(...)
        scaling_factor = 2 / ((1 + beta_squared) ** (D / 2))
        exponent = -beta_squared / (2 + 2 * beta_squared)
        rhs = scaling_factor * torch.exp(squared_norms * exponent).sum() / N

        # Constant term: 1 / (1 + 2*beta^2)^(D/2)
        constant = 1 / ((1 + 2 * beta_squared) ** (D / 2))

        # BHEP statistic: LHS - RHS + constant
        statistic = lhs - rhs + constant

        return statistic

    def __repr__(self) -> str:
        return f"BHEP(beta={self.beta})"
