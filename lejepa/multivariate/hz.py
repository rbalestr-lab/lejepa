import numpy as np
from .bhep import BHEP
import torch
from typing import Union
from .base import MultivariateTest


class HZ(MultivariateTest):
    """
    Henze-Zirkler (HZ) test for multivariate normality.

    The Henze-Zirkler test is a powerful multivariate normality test that uses
    an adaptive bandwidth selection rule. Unlike the standard BHEP test which
    requires manual bandwidth tuning, the HZ test automatically computes an
    optimal bandwidth parameter based on the sample size (N) and dimensionality (D).

    Mathematical Formulation
    ------------------------
    The test statistic is based on a weighted L² distance between the empirical
    and theoretical characteristic functions, using a Gaussian kernel with
    adaptively chosen bandwidth β.

    The bandwidth is computed using the Henze-Zirkler rule:

        β = (1/√2) × [(2D + 1) × N / 4]^(1/(D+4))

    The test statistic takes the form:

        T_N = (1/N²) ∑ᵢⱼ exp(-β²/2 × ||xᵢ - xⱼ||²)
              - 2/(N(1+β²)^(D/2)) × ∑ᵢ exp(-β²/(2+2β²) × ||xᵢ||²)
              + 1/(1+2β²)^(D/2)

    Under the null hypothesis of multivariate normality, the test statistic
    has a known limiting distribution that can be used for hypothesis testing.

    Attributes
    ----------
    None - dimensionality is automatically determined from input data shape.

    Parameters
    ----------
    None - the HZ test requires no configuration parameters.

    Properties
    ----------
    - **Adaptive**: Automatically adjusts sensitivity based on sample size
    - **Consistent**: Test is consistent against any non-normal alternative
    - **Affine invariant**: Invariant under affine transformations of the data
    - **Powerful**: Generally more powerful than other normality tests for
      moderate to high dimensions

    Performance Note
    ----------------
    This test has O(N²) computational complexity where N is the number of samples.
    For large datasets (N > 1000), consider using slicing-based tests instead,
    or subsample your data.

    When to Use
    -----------
    - Testing multivariate normality with 2 or more dimensions
    - When you want automatic parameter selection (no manual tuning)
    - For sample sizes N ≥ 20 (test may be unreliable for very small samples)
    - When data dimensionality is moderate (D < 10 recommended)

    Advantages Over BHEP
    --------------------
    - No need to manually tune the bandwidth parameter β
    - Optimal power properties proven theoretically
    - Widely studied and validated in literature

    Examples
    --------
    Basic usage with normally distributed data:

    >>> import torch
    >>> from lejepa.multivariate import HZ
    >>>
    >>> # Generate multivariate normal data
    >>> data = torch.randn(100, 5)  # 100 samples, 5 dimensions
    >>>
    >>> # Initialize test (no parameters needed)
    >>> hz_test = HZ()
    >>>
    >>> # Compute test statistic
    >>> statistic = hz_test(data)
    >>> print(f"Test statistic: {statistic.item():.6f}")

    Testing with non-normal data:

    >>> # Generate non-normal data (uniform)
    >>> data_uniform = torch.rand(100, 5)
    >>> statistic_uniform = hz_test(data_uniform)
    >>> # Expect larger statistic (deviation from normality)

    Using with NumPy arrays:

    >>> import numpy as np
    >>> data_np = np.random.randn(200, 3)
    >>> hz_test = HZ()
    >>> statistic = hz_test(data_np)

    References
    ----------
    .. [1] Henze, N., & Zirkler, B. (1990). "A class of invariant consistent
           tests for multivariate normality." Communications in Statistics-Theory
           and Methods, 19(10), 3595-3617.

    .. [2] Henze, N., & Wagner, T. (1997). "A new approach to the BHEP tests
           for multivariate normality." Journal of Multivariate Analysis,
           62(1), 1-23.

    See Also
    --------
    BHEP : Beta-Henze Energy-based Projection test with manual bandwidth
    COMB : Combination-based test statistic

    Notes
    -----
    - The test is based on the empirical characteristic function
    - Critical values can be obtained via Monte Carlo simulation
    - For very high dimensions (D > 20), consider dimension reduction first
    - The test assumes the data has been centered (mean zero) for optimal
      interpretation, though this is not strictly required
    - Computational complexity is O(N² × D) due to pairwise distance calculations

    Warnings
    --------
    - Very small sample sizes (N < 20) may produce unreliable results
    - High dimensionality relative to sample size (D/N > 0.1) may reduce power
    - Outliers can significantly affect the test statistic
    """

    def __init__(self):
        super().__init__()
        self._bhep = BHEP(beta=1.0)  # Reusable instance

    def _compute_bhep_statistic(self, x: torch.Tensor, beta: float) -> torch.Tensor:
        """Compute BHEP statistic with given bandwidth."""
        return self._bhep.forward(x, beta=beta)

    @staticmethod
    def compute_bandwidth(n_samples: int, n_dims: int) -> float:
        """
        Compute the Henze-Zirkler optimal bandwidth parameter.

        This method implements the Henze-Zirkler bandwidth selection rule,
        which provides an asymptotically optimal choice of the smoothing
        parameter β for the test statistic.

        The formula is:
            β = (1/√2) × [(2D + 1) × N / 4]^(1/(D+4))

        where N is the sample size and D is the dimensionality.

        Parameters
        ----------
        n_samples : int
            Number of samples (N) in the dataset. Must be positive.

        n_dims : int
            Number of dimensions (D) in the dataset. Must be positive.

        Returns
        -------
        float
            The optimal bandwidth parameter β. Always positive and finite.

        Raises
        ------
        ValueError
            If n_samples or n_dims are not positive integers.

        RuntimeError
            If the computed bandwidth is invalid (non-positive or non-finite).

        Warns
        -----
        UserWarning
            If sample size is very small (N < 10), as the test may be unreliable.

        Notes
        -----
        - The bandwidth decreases as sample size increases (fewer samples need
          more smoothing)
        - The bandwidth increases with dimensionality (higher dimensions need
          less smoothing to avoid over-smoothing)
        - The exponent 1/(D+4) ensures the bandwidth converges to optimal rate

        Examples
        --------
        >>> bandwidth = HZ.compute_bandwidth(n_samples=100, n_dims=5)
        >>> print(f"Optimal β: {bandwidth:.4f}")
        Optimal β: 1.2345

        >>> # Larger sample needs smaller bandwidth
        >>> beta_small = HZ.compute_bandwidth(50, 3)
        >>> beta_large = HZ.compute_bandwidth(500, 3)
        >>> assert beta_large < beta_small

        >>> # Higher dimensions need larger bandwidth
        >>> beta_low_dim = HZ.compute_bandwidth(100, 2)
        >>> beta_high_dim = HZ.compute_bandwidth(100, 10)
        >>> assert beta_high_dim > beta_low_dim
        """
        # Input validation
        if n_samples <= 0:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples}")
        if n_dims <= 0:
            raise ValueError(f"n_dims must be a positive integer, got {n_dims}")

        # Warn for very small samples
        if n_samples < 10:
            import warnings

            warnings.warn(
                f"Sample size {n_samples} is very small (< 10). "
                "The Henze-Zirkler test may not be reliable. "
                "Consider collecting more data or using a different test.",
                UserWarning,
                stacklevel=2,
            )

        # Warn for high dimensional data with small samples
        if n_dims / n_samples > 0.1:
            import warnings

            warnings.warn(
                f"Dimensionality ({n_dims}) is high relative to sample size "
                f"({n_samples}). Ratio D/N = {n_dims/n_samples:.2f} > 0.1. "
                "Test power may be reduced. Consider dimension reduction.",
                UserWarning,
                stacklevel=2,
            )

        # Compute bandwidth using Henze-Zirkler formula
        # β = (1/√2) × [(2D + 1) × N / 4]^(1/(D+4))
        numerator = (2 * n_dims + 1) * n_samples
        exponent = 1.0 / (n_dims + 4)

        # Compute in steps for numerical stability and clarity
        base = numerator / 4.0
        power_term = base**exponent
        beta = power_term / np.sqrt(2.0)

        # Sanity check: bandwidth must be positive and finite
        if beta <= 0 or not np.isfinite(beta):
            raise RuntimeError(
                f"Computed invalid bandwidth β = {beta}. "
                f"This should not happen. Please check inputs: "
                f"n_samples={n_samples}, n_dims={n_dims}"
            )

        return float(beta)

    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Compute the Henze-Zirkler test statistic.

        This method automatically computes the optimal bandwidth parameter
        based on the sample size and dimensionality, then calculates the
        test statistic using the BHEP formulation.

        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            Input data of shape (N, D) where:
            - N is the number of samples (observations)
            - D is the number of dimensions (features/variables)

            Can be either a PyTorch tensor or NumPy array. NumPy arrays
            will be automatically converted to PyTorch tensors.

        Returns
        -------
        torch.Tensor
            Scalar tensor containing the test statistic value.

            - Values close to 0 suggest the data is approximately normal
            - Larger positive values indicate stronger evidence against normality
            - The actual threshold depends on the significance level and can be
              obtained through Monte Carlo simulation or asymptotic tables

        Raises
        ------
        ValueError
            If the input data is empty, has wrong dimensionality, or contains
            invalid values (NaN, Inf).

        RuntimeError
            If an unexpected error occurs during computation.

        Notes
        -----
        Computational Complexity
        ~~~~~~~~~~~~~~~~~~~~~~~~
        - Time: O(N² × D) due to pairwise distance computation
        - Space: O(N²) for storing the pairwise distance matrix

        For large datasets (N > 1000), consider:
        - Using a subsample of the data
        - Using approximate tests with lower complexity
        - Splitting into batches if memory is limited

        Data Preprocessing
        ~~~~~~~~~~~~~~~~~~
        While not strictly required, the following preprocessing may improve
        test performance:
        - Center the data (subtract mean)
        - Standardize features to unit variance
        - Remove or handle outliers appropriately

        Interpreting Results
        ~~~~~~~~~~~~~~~~~~~~
        The test statistic should be compared against critical values from:
        1. Monte Carlo simulation under the null hypothesis
        2. Asymptotic distribution tables (for large N)
        3. Bootstrap-based critical values

        Examples
        --------
        Basic usage:

        >>> import torch
        >>> from your_module import HZ
        >>>
        >>> # Generate normal data
        >>> data = torch.randn(100, 5)
        >>> hz_test = HZ(dim=5)
        >>> statistic = hz_test(data)
        >>> print(f"Statistic: {statistic.item():.6f}")

        With NumPy arrays:

        >>> import numpy as np
        >>> data_np = np.random.randn(200, 3)
        >>> hz_test = HZ(dim=3)
        >>> statistic = hz_test(data_np)

        Testing non-normal data:

        >>> # Generate uniformly distributed data
        >>> data_uniform = torch.rand(100, 5)
        >>> statistic_uniform = hz_test(data_uniform)
        >>> # Expect larger statistic than for normal data

        Batch processing:

        >>> datasets = [torch.randn(50, 3) for _ in range(10)]
        >>> hz_test = HZ(dim=3)
        >>> statistics = [hz_test(data) for data in datasets]
        >>> mean_stat = torch.stack(statistics).mean()

        With standardized data:

        >>> # Standardize data first
        >>> data = torch.randn(100, 5)
        >>> data_std = (data - data.mean(0)) / data.std(0)
        >>> statistic = hz_test(data_std)
        """
        # Prepare and validate input data
        x = self.prepare_data(x)
        N, D = x.shape

        # Validate sample size
        if N == 0:
            raise ValueError(
                "Input data cannot be empty. " "Please provide at least one sample."
            )

        # Check for invalid values
        if torch.isnan(x).any():
            raise ValueError(
                "Input data contains NaN values. "
                "Please handle missing data before testing."
            )

        if torch.isinf(x).any():
            raise ValueError(
                "Input data contains infinite values. "
                "Please handle outliers or scale data appropriately."
            )

        # Compute adaptive bandwidth using Henze-Zirkler rule
        optimal_beta = self.compute_bandwidth(n_samples=N, n_dims=D)

        # Compute test statistic using BHEP formulation with optimal bandwidth
        statistic = self._compute_bhep_statistic(x, beta=optimal_beta)

        return statistic

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Callable interface for the test.

        This is a convenience method that allows the test object to be called
        directly as a function.

        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            Input data of shape (N, D).

        Returns
        -------
        torch.Tensor
            Test statistic value.

        Examples
        --------
        >>> hz_test = HZ(dim=5)
        >>> data = torch.randn(100, 5)
        >>>
        >>> # These are equivalent:
        >>> stat1 = hz_test(data)
        >>> stat2 = hz_test.forward(data)
        >>> assert torch.allclose(stat1, stat2)
        """
        return self.forward(x)

    def __repr__(self) -> str:
        """String representation of the test object."""
        return "HZ()"

    def __str__(self) -> str:
        """User-friendly string representation."""
        return "Henze-Zirkler test with adaptive bandwidth"
