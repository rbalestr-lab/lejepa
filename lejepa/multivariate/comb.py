import numpy as np
import torch
from .base import MultivariateTest
from typing import Union


class COMB(MultivariateTest):
    """
    Combination-based (COMB) test statistic for multivariate normality.

    Computes a test statistic using a combination of exponential and
    cosine kernels applied to the data. The statistic is scaled by sqrt(N)
    and centered around 0.

    Args:
        dim: Dimensionality of the input data
        gamma: Bandwidth parameter for the kernel (must be > 0)
               Controls the sensitivity of the test

    Performance Note:
        This test has O(N²) computational complexity where N is the number of samples.
        For large datasets (N > 1000), consider using slicing-based tests instead,
        or subsample your data.

    Reference:
        [Add citation/paper reference here]

    Note:
        This implementation may be incomplete - see the commented constant
        term in the original formulation.
    """

    def __init__(self, gamma: float = 0.1):
        super().__init__()

        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")

        self.gamma = gamma

    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Compute COMB test statistic.

        Args:
            x: Input data of shape (N, D) where N is number of samples
               and D is dimensionality

        Returns:
            COMB test statistic (scalar tensor), scaled by sqrt(N)

        Raises:
            ValueError: If input data is empty or has wrong dimensionality
        """
        x = self.prepare_data(x)
        N, D = x.shape

        # Validate input
        if N == 0:
            raise ValueError("Input data cannot be empty")

        # Compute squared norms: ||x_i||^2
        squared_norms = x.square().sum(dim=1)  # Shape: (N,)

        # Exponential term based on norm differences
        # NOTE: This uses ||x_i||^2 - ||x_j||^2, not ||x_i - x_j||^2
        norm_diff_matrix = squared_norms.unsqueeze(1) - squared_norms.unsqueeze(0)
        exp_term = torch.exp(norm_diff_matrix / (4 * self.gamma))

        # Cosine term based on inner products
        # cos(<x_i, x_j> / (2*gamma))
        inner_products = x @ x.T
        cos_term = torch.cos(inner_products / (2 * self.gamma))

        # Combined kernel matrix
        kernel_matrix = exp_term * cos_term  # Element-wise product

        # Mean over all pairwise interactions
        kernel_mean = kernel_matrix.mean()

        # QUESTION: Should we use this constant?
        # constant = (np.pi / self.gamma) ** (D / 2) * np.sqrt(N)

        # Final statistic: sqrt(N) * (kernel_mean - 1)
        statistic = np.sqrt(N) * (kernel_mean - 1)

        return statistic

    def __repr__(self) -> str:
        return f"COMB(gamma={self.gamma})"
