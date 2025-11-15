import torch
from ..utils import all_reduce


class SlicingUnivariateTest(torch.nn.Module):
    """
    Multivariate distribution test using random slicing and univariate test statistics.
    This module extends univariate statistical tests to multivariate data by projecting
    samples onto random 1D directions (slices) and aggregating univariate test statistics
    across all projections. The approach is based on the sliced method for comparing
    high-dimensional distributions.
    The test projects multivariate samples x ∈ ℝᴰ onto random unit vectors:
        x_projected = x @ A
    where A ∈ ℝᴰˣᴷ contains K normalized random direction vectors. A univariate
    test is then applied to each of the K projected samples, and results are aggregated.
    Args:
        univariate_test (torch.nn.Module): A univariate test module that accepts
            (*, N, K) tensors and returns (*, K) test statistics, where N is the
            number of samples and K is the number of slices.
        num_slices (int): Number of random 1D projections (slices) to use. More
            slices increase test power but add computational cost.
        reduction (str, optional): How to aggregate statistics across slices:
            - 'mean': Return the average statistic across all slices
            - 'sum': Return the sum of statistics across all slices
            - None: Return individual statistics for each slice (*, num_slices)
            Default: 'mean'.
        sampler (str, optional): Random sampling method for projection directions:
            - 'gaussian': Sample from standard normal distribution (Gaussian projections)
            Default: 'gaussian'.
        clip_value (float, optional): Minimum threshold for test statistics. Values
            below this threshold are clipped to zero. Useful for reducing noise from
            negligible deviations. Default: None (no clipping).

    Performance Note:
        This test has O(N × D × num_slices) complexity where N is the number of samples,
        D is dimensionality, and num_slices is the number of projections. This is much
        more efficient than energy-based tests (O(N²)) for large N. Consider using this
        over BHEP/COMB/HV/HZ for datasets with N > 1000. The num_slices parameter
        (default: 100) controls the trade-off between accuracy and computation time.

    Attributes:
        global_step (torch.Tensor): Counter for deterministic random seed generation,
            synchronized across distributed processes to ensure consistent projections.
    Notes:
        - Projection directions are normalized to unit vectors (L2 norm = 1).
        - In distributed training, the random seed is synchronized across all ranks
          using all_reduce to ensure identical projections on all devices.
        - The generator is cached and reused across forward passes for efficiency.
        - The global step counter increments after each forward pass to ensure
          different random projections in successive calls.
    Shape:
        - Input: (*, N, D) where * is any number of batch dimensions, N is the
          number of samples, and D is the feature dimension.
        - Output:
            - Scalar if reduction='mean' or 'sum'
            - (*, num_slices) if reduction=None
    Example:
        >>> from your_module import FastEppsPulley, SlicingUnivariateTest
        >>>
        >>> # Create univariate test
        >>> univariate_test = FastEppsPulley(t_max=5.0, n_points=21)
        >>>
        >>> # Wrap with slicing for multivariate testing
        >>> test = SlicingUnivariateTest(
        ...     univariate_test=univariate_test,
        ...     num_slices=100,
        ...     reduction='mean',
        ...     sampler='gaussian',
        ...     clip_value=0.01
        ... )
        >>>
        >>> # Test multivariate samples
        >>> samples = torch.randn(1000, 50)  # 1000 samples, 50 dimensions
        >>> statistic = test(samples)
        >>> print(f"Test statistic: {statistic.item():.4f}")
        >>>
        >>> # Batch processing
        >>> batch_samples = torch.randn(32, 1000, 50)  # 32 batches
        >>> batch_stats = test(batch_samples)  # Returns scalar (averaged over slices)
    References:
        - Rabin, J., Peyré, G., Delon, J., & Bernot, M. (2012). Wasserstein
          barycenter and its application to texture mixing. In Scale Space and
          Variational Methods in Computer Vision (pp. 435-446).
        - Bonneel, N., Rabin, J., Peyré, G., & Pfister, H. (2015). Sliced and
          Radon Wasserstein barycenters of measures. Journal of Mathematical
          Imaging and Vision, 51(1), 22-45.
    """

    def __init__(
        self,
        univariate_test,
        num_slices: int,
        reduction: str = "mean",
        sampler: str = "gaussian",
        clip_value: float = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.num_slices = num_slices
        self.sampler = sampler
        self.univariate_test = univariate_test
        self.clip_value = clip_value
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

        # Generator reuse
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device, seed):
        """Get or create generator for given device and seed."""
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x):
        """
        Apply sliced univariate test to multivariate samples.
        Args:
            x (torch.Tensor): Input samples of shape (*, N, D) where * represents
                any number of batch dimensions, N is the number of samples, and
                D is the feature dimension.
        Returns:
            torch.Tensor: Aggregated test statistic(s).
                - Scalar tensor if reduction='mean' or 'sum'
                - Shape (*, num_slices) if reduction=None
        """
        with torch.no_grad():
            # Synchronize global_step across all ranks
            global_step_sync = all_reduce(self.global_step.clone(), op="MAX")
            seed = global_step_sync.item()
            dev = dict(device=x.device)

            # Get reusable generator
            g = self._get_generator(x.device, seed)

            proj_shape = (x.size(-1), self.num_slices)
            A = torch.randn(proj_shape, **dev, generator=g)
            A /= A.norm(p=2, dim=0)
            self.global_step.add_(1)

        stats = self.univariate_test(x @ A)
        if self.clip_value is not None:
            stats[stats < self.clip_value] = 0
        if self.reduction == "mean":
            return stats.mean()
        elif self.reduction == "sum":
            return stats.sum()
        elif self.reduction is None:
            return stats
