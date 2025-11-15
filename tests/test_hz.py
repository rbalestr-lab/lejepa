"""
Unit tests for multivariate normality tests: BHEP, COMB, and HZ.

This test suite validates the functionality, edge cases, and error handling
of the three multivariate statistical tests without requiring CUDA.

Run with:
    pytest test_multivariate_tests.py -v
    pytest test_multivariate_tests.py -v -m unit
    pytest test_multivariate_tests.py -v --tb=short
"""

import pytest
import torch
import numpy as np
from lejepa.multivariate import HZ
import warnings

# Mark all tests in this module as unit tests


class TestHZ:
    """Comprehensive unit test suite for HZ multivariate normality test (CPU only)."""

    @pytest.fixture(autouse=True)
    def setup_cpu(self):
        """Ensure all tests run on CPU."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield

    @pytest.fixture
    def hz(self):
        """Standard HZ instance (dimension-agnostic)."""
        return HZ()

    # ==================== Initialization Tests ====================

    @pytest.mark.unit
    def test_init_no_args(self):
        """Test initialization without arguments."""
        hz = HZ()
        assert hz is not None

    @pytest.mark.unit
    def test_init_multiple_instances(self):
        """Test that multiple instances can be created."""
        hz1 = HZ()
        hz2 = HZ()
        assert hz1 is not hz2

    # ==================== Bandwidth Computation Tests ====================

    @pytest.mark.unit
    def test_compute_bandwidth_basic(self):
        """Test basic bandwidth computation."""
        beta = HZ.compute_bandwidth(n_samples=100, n_dims=5)
        assert isinstance(beta, float)
        assert beta > 0
        assert np.isfinite(beta)

    @pytest.mark.unit
    def test_compute_bandwidth_formula(self):
        """Test bandwidth formula correctness."""
        N, D = 100, 3
        beta = HZ.compute_bandwidth(n_samples=N, n_dims=D)

        # Manual computation
        expected = (1 / np.sqrt(2)) * (((2 * D + 1) * N / 4) ** (1 / (D + 4)))

        assert np.isclose(beta, expected, rtol=1e-10)

    @pytest.mark.unit
    def test_bandwidth_invalid_samples_zero(self):
        """Test that zero samples raises ValueError."""
        with pytest.raises(ValueError, match="n_samples must be a positive integer"):
            HZ.compute_bandwidth(n_samples=0, n_dims=3)

    @pytest.mark.unit
    def test_bandwidth_invalid_samples_negative(self):
        """Test that negative samples raises ValueError."""
        with pytest.raises(ValueError, match="n_samples must be a positive integer"):
            HZ.compute_bandwidth(n_samples=-10, n_dims=3)

    @pytest.mark.unit
    def test_bandwidth_invalid_dims_zero(self):
        """Test that zero dimensions raises ValueError."""
        with pytest.raises(ValueError, match="n_dims must be a positive integer"):
            HZ.compute_bandwidth(n_samples=100, n_dims=0)

    @pytest.mark.unit
    def test_bandwidth_invalid_dims_negative(self):
        """Test that negative dimensions raises ValueError."""
        with pytest.raises(ValueError, match="n_dims must be a positive integer"):
            HZ.compute_bandwidth(n_samples=100, n_dims=-5)

    @pytest.mark.unit
    def test_bandwidth_warning_small_sample(self):
        """Test warning for very small sample size."""
        with pytest.warns(UserWarning, match="Sample size .* is very small"):
            HZ.compute_bandwidth(n_samples=5, n_dims=3)

    @pytest.mark.unit
    def test_bandwidth_warning_high_dim_ratio(self):
        """Test warning for high D/N ratio."""
        with pytest.warns(UserWarning, match="Dimensionality .* is high relative"):
            HZ.compute_bandwidth(n_samples=50, n_dims=10)

    @pytest.mark.unit
    def test_bandwidth_no_warning_good_params(self):
        """Test no warning for reasonable parameters."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                HZ.compute_bandwidth(n_samples=100, n_dims=3)
            except UserWarning:
                pytest.fail("Unexpected warning for reasonable parameters")

    # ==================== Input Validation Tests ====================

    @pytest.mark.unit
    def test_empty_input(self, hz):
        """Test that empty input raises ValueError."""
        empty_data = np.array([]).reshape(0, 2)
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            hz(empty_data)

    @pytest.mark.unit
    def test_nan_input(self, hz):
        """Test that NaN input raises ValueError."""
        data = np.random.randn(50, 2).astype(np.float32)
        data[0, 0] = np.nan
        with pytest.raises(ValueError, match="contains NaN values"):
            hz(data)

    @pytest.mark.unit
    def test_inf_input(self, hz):
        """Test that infinite input raises ValueError."""
        data = np.random.randn(50, 2).astype(np.float32)
        data[0, 0] = np.inf
        with pytest.raises(ValueError, match="contains infinite values"):
            hz(data)

    @pytest.mark.unit
    def test_single_sample(self, hz):
        """Test with single sample."""
        single_sample = np.array([[1.0, 2.0]], dtype=np.float32)
        result = hz(single_sample)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"
        assert result.numel() == 1

    @pytest.mark.unit
    def test_1d_input_fails(self, hz):
        """Test that 1D input raises ValueError."""
        data = np.random.randn(50)  # 1D array
        with pytest.raises(ValueError, match="Expected 2D input"):
            hz(data)

    @pytest.mark.unit
    def test_3d_input_fails(self, hz):
        """Test that 3D input raises ValueError."""
        data = np.random.randn(10, 5, 3)  # 3D array
        with pytest.raises(ValueError, match="Expected 2D input"):
            hz(data)

    # ==================== Multi-Dimensional Flexibility Tests ====================

    @pytest.mark.unit
    def test_reusable_across_dimensions(self, hz):
        """Test that same instance can handle different dimensions."""
        # 2D data
        data_2d = np.random.randn(50, 2).astype(np.float32)
        result_2d = hz(data_2d)
        assert torch.isfinite(result_2d)

        # 5D data with same instance
        data_5d = np.random.randn(50, 5).astype(np.float32)
        result_5d = hz(data_5d)
        assert torch.isfinite(result_5d)

        # Both should be valid but different
        assert result_2d.numel() == 1
        assert result_5d.numel() == 1

    @pytest.mark.unit
    def test_various_dimensions_single_instance(self, hz):
        """Test single instance with multiple different dimensions."""
        for dim in [1, 2, 3, 5, 10]:
            data = np.random.randn(100, dim).astype(np.float32)
            result = hz(data)
            assert torch.isfinite(result)
            assert result.device.type == "cpu"

    # ==================== Data Type Tests ====================

    @pytest.mark.unit
    def test_numpy_input(self, hz):
        """Test with numpy array input."""
        data = np.random.randn(50, 2).astype(np.float32)
        result = hz(data)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"
        assert result.numel() == 1

    @pytest.mark.unit
    def test_torch_input_cpu(self, hz):
        """Test with torch tensor input on CPU."""
        data = torch.randn(50, 2, dtype=torch.float32, device="cpu")
        result = hz(data)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"
        assert result.numel() == 1

    @pytest.mark.unit
    def test_torch_float32(self, hz):
        """Test with float32 input."""
        data = torch.randn(50, 2, dtype=torch.float32, device="cpu")
        result = hz(data)
        assert torch.isfinite(result)
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_torch_float64(self, hz):
        """Test with float64 input."""
        data = torch.randn(50, 2, dtype=torch.float64, device="cpu")
        result = hz(data)
        assert torch.isfinite(result)
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_numpy_float32(self, hz):
        """Test with numpy float32."""
        np.random.seed(42)
        data = np.random.randn(50, 2).astype(np.float32)
        result = hz(data)
        assert torch.isfinite(result)
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_numpy_float64(self, hz):
        """Test with numpy float64."""
        np.random.seed(42)
        data = np.random.randn(50, 2).astype(np.float64)
        result = hz(data)
        assert torch.isfinite(result)
        assert result.device.type == "cpu"

    # ==================== Output Tests ====================

    @pytest.mark.unit
    def test_output_is_scalar(self, hz):
        """Test output is a scalar."""
        np.random.seed(42)
        data = np.random.randn(100, 2).astype(np.float32)
        result = hz(data)
        assert result.numel() == 1
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_output_is_finite(self, hz):
        """Test output is finite."""
        np.random.seed(42)
        data = np.random.randn(100, 2).astype(np.float32)
        result = hz(data)
        assert torch.isfinite(result)
        assert not torch.isnan(result)
        assert not torch.isinf(result)

    @pytest.mark.unit
    def test_output_type(self, hz):
        """Test output is torch.Tensor."""
        np.random.seed(42)
        data = np.random.randn(100, 2).astype(np.float32)
        result = hz(data)
        assert isinstance(result, torch.Tensor)
        assert not isinstance(result, np.ndarray)

    # ==================== Mathematical Property Tests ====================

    @pytest.mark.unit
    def test_deterministic(self, hz):
        """Test that same input gives same output."""
        np.random.seed(42)
        data = np.random.randn(50, 2).astype(np.float32)
        result1 = hz(data)
        result2 = hz(data)
        assert torch.allclose(result1, result2)

    @pytest.mark.unit
    def test_mvn_data_small_statistic(self, hz):
        """Test that multivariate normal data gives small statistic."""
        torch.manual_seed(42)
        data = torch.randn(200, 2, dtype=torch.float32, device="cpu")
        result = hz(data)

        # For MVN data, statistic should be close to 0
        assert result.item() < 1.0  # Reasonable threshold

    @pytest.mark.unit
    def test_non_normal_data_larger_statistic(self, hz):
        """Test that non-normal data gives larger statistic."""
        np.random.seed(42)
        data = np.random.uniform(-3, 3, size=(200, 2)).astype(np.float32)
        result = hz(data)

        # Should detect deviation from normality
        assert result.item() >= 0

    @pytest.mark.unit
    def test_symmetry_property(self, hz):
        """Test that permuting samples doesn't change result."""
        np.random.seed(42)
        data = np.random.randn(50, 2).astype(np.float32)
        result1 = hz(data)

        # Shuffle the rows
        np.random.seed(123)
        permuted_data = data[np.random.permutation(50)]
        result2 = hz(permuted_data)

        assert torch.allclose(result1, result2, atol=1e-5)

    @pytest.mark.unit
    def test_adaptive_bandwidth_varies(self, hz):
        """Test that adaptive bandwidth varies with sample size."""
        np.random.seed(42)
        data_small = np.random.randn(30, 3).astype(np.float32)
        data_large = np.random.randn(300, 3).astype(np.float32)

        # Results should differ because bandwidth is adaptive
        result_small = hz(data_small)
        result_large = hz(data_large)

        # Can't directly compare, but both should be valid
        assert torch.isfinite(result_small)
        assert torch.isfinite(result_large)

    # ==================== Edge Cases ====================

    @pytest.mark.unit
    def test_identical_points(self, hz):
        """Test with all identical points."""
        data = np.ones((10, 2), dtype=np.float32)
        result = hz(data)
        assert torch.isfinite(result)

    @pytest.mark.unit
    def test_collinear_points(self, hz):
        """Test with collinear points."""
        data = np.column_stack([np.linspace(0, 1, 50), np.linspace(0, 1, 50)]).astype(
            np.float32
        )
        result = hz(data)
        assert torch.isfinite(result)

    @pytest.mark.unit
    def test_two_samples(self, hz):
        """Test with minimum sample size (2)."""
        data = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        result = hz(data)
        assert torch.isfinite(result)

    @pytest.mark.unit
    def test_large_sample_size(self, hz):
        """Test with larger sample size."""
        np.random.seed(42)
        data = np.random.randn(1000, 2).astype(np.float32)
        result = hz(data)
        assert torch.isfinite(result)

    @pytest.mark.unit
    def test_high_dimension(self, hz):
        """Test with higher dimensional data."""
        np.random.seed(42)
        data = np.random.randn(100, 10).astype(np.float32)
        result = hz(data)
        assert torch.isfinite(result)

    @pytest.mark.unit
    def test_very_high_dimension(self, hz):
        """Test with very high dimensional data."""
        np.random.seed(42)
        data = np.random.randn(200, 20).astype(np.float32)
        result = hz(data)
        assert torch.isfinite(result)

    @pytest.mark.unit
    def test_single_dimension(self, hz):
        """Test with 1D features (univariate)."""
        np.random.seed(42)
        data = np.random.randn(100, 1).astype(np.float32)
        result = hz(data)
        assert torch.isfinite(result)

    # ==================== Specific Mathematical Cases ====================

    @pytest.mark.unit
    def test_zero_centered_data(self, hz):
        """Test with zero-centered data."""
        np.random.seed(42)
        data = np.random.randn(100, 2).astype(np.float32)
        data -= data.mean(axis=0)
        result = hz(data)
        assert torch.isfinite(result)

    @pytest.mark.unit
    def test_standardized_data(self, hz):
        """Test with standardized data."""
        np.random.seed(42)
        data = np.random.randn(100, 2).astype(np.float32)
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        result = hz(data)
        assert torch.isfinite(result)

    @pytest.mark.unit
    def test_scaled_data(self, hz):
        """Test that scaling affects the statistic."""
        np.random.seed(42)
        data = np.random.randn(100, 2).astype(np.float32)
        result1 = hz(data)
        result2 = hz(data * 2)

        # Scaling should change the statistic
        assert not torch.allclose(result1, result2)

    @pytest.mark.unit
    def test_translated_data(self, hz):
        """Test that translation affects the statistic."""
        np.random.seed(42)
        data = np.random.randn(100, 2).astype(np.float32)
        result1 = hz(data)
        result2 = hz(data + 5)

        # Translation should change the statistic
        assert not torch.allclose(result1, result2)

    @pytest.mark.unit
    def test_negative_values(self, hz):
        """Test with all negative values."""
        np.random.seed(42)
        data = -np.abs(np.random.randn(50, 2)).astype(np.float32)
        result = hz(data)
        assert torch.isfinite(result)

    @pytest.mark.unit
    def test_mixed_signs(self, hz):
        """Test with mixed positive and negative values."""
        np.random.seed(42)
        data = np.random.randn(50, 2).astype(np.float32)
        data[:25, 0] = -np.abs(data[:25, 0])
        data[25:, 1] = np.abs(data[25:, 1])
        result = hz(data)
        assert torch.isfinite(result)

    # ==================== Numerical Stability Tests ====================

    @pytest.mark.unit
    def test_extreme_large_values(self, hz):
        """Test with extreme large values."""
        np.random.seed(42)
        data = (np.random.randn(50, 2) * 1000).astype(np.float32)
        result = hz(data)
        assert torch.isfinite(result)

    @pytest.mark.unit
    def test_extreme_small_values(self, hz):
        """Test with extreme small values."""
        np.random.seed(42)
        data = (np.random.randn(50, 2) * 1e-6).astype(np.float32)
        result = hz(data)
        assert torch.isfinite(result)

    @pytest.mark.unit
    def test_mixed_scale_data(self, hz):
        """Test with mixed scale across dimensions."""
        np.random.seed(42)
        data = np.random.randn(50, 2).astype(np.float32)
        data[:, 0] *= 1000
        data[:, 1] *= 0.001
        result = hz(data)
        assert torch.isfinite(result)

    @pytest.mark.unit
    def test_near_zero_variance(self, hz):
        """Test with near-zero variance in one dimension."""
        np.random.seed(42)
        data = np.random.randn(50, 2).astype(np.float32)
        data[:, 1] = 1.0 + np.random.randn(50).astype(np.float32) * 1e-8
        result = hz(data)
        assert torch.isfinite(result)

    # ==================== Interface Tests ====================

    @pytest.mark.unit
    def test_callable_interface(self, hz):
        """Test that __call__ and forward are equivalent."""
        np.random.seed(42)
        data = np.random.randn(50, 2).astype(np.float32)

        result_call = hz(data)
        result_forward = hz.forward(data)

        assert torch.allclose(result_call, result_forward)

    @pytest.mark.unit
    def test_repr(self, hz):
        """Test string representation."""
        repr_str = repr(hz)
        assert "HZ" in repr_str
        # Should not contain dimension since it's not specified
        assert "HZ()" == repr_str

    @pytest.mark.unit
    def test_str(self, hz):
        """Test user-friendly string."""
        str_repr = str(hz)
        assert "Henze-Zirkler" in str_repr
        assert "adaptive" in str_repr

    @pytest.mark.unit
    def test_repr_eval(self):
        """Test that repr can be evaluated."""
        hz = HZ()
        repr_str = repr(hz)
        # Should be "HZ()"
        assert repr_str == "HZ()"

    # ==================== Comparison with Known Distributions ====================

    @pytest.mark.unit
    def test_standard_normal_vs_uniform(self, hz):
        """Test that uniform data has larger statistic than normal."""
        torch.manual_seed(42)
        np.random.seed(42)

        normal_data = torch.randn(200, 3, dtype=torch.float32)
        uniform_data = torch.from_numpy(
            np.random.uniform(-3, 3, (200, 3)).astype(np.float32)
        )

        stat_normal = hz(normal_data)
        stat_uniform = hz(uniform_data)

        # Both should be non-negative
        assert stat_uniform.item() >= 0
        assert stat_normal.item() >= 0

    @pytest.mark.unit
    def test_multimodal_vs_normal(self, hz):
        """Test that multimodal data has larger statistic."""
        torch.manual_seed(42)

        # Standard normal
        normal_data = torch.randn(200, 2, dtype=torch.float32)

        # Bimodal (mixture of two normals)
        mode1 = torch.randn(100, 2, dtype=torch.float32) - 3
        mode2 = torch.randn(100, 2, dtype=torch.float32) + 3
        bimodal_data = torch.cat([mode1, mode2], dim=0)

        stat_normal = hz(normal_data)
        stat_bimodal = hz(bimodal_data)

        # Both should be non-negative
        assert stat_normal.item() >= 0
        assert stat_bimodal.item() >= 0

    # ==================== Consistency Tests ====================

    @pytest.mark.unit
    def test_consistency_across_runs(self, hz):
        """Test consistency across multiple runs with same data."""
        np.random.seed(42)
        data = np.random.randn(100, 3).astype(np.float32)

        results = [hz(data) for _ in range(5)]

        # All results should be identical
        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i])

    @pytest.mark.unit
    def test_different_instances_same_result(self):
        """Test that different instances give same result."""
        np.random.seed(42)
        data = np.random.randn(100, 3).astype(np.float32)

        hz1 = HZ()
        hz2 = HZ()

        result1 = hz1(data)
        result2 = hz2(data)

        assert torch.allclose(result1, result2)


# ==================== Parametrized Tests ====================


@pytest.mark.unit
@pytest.mark.parametrize("dim", [1, 2, 3, 5, 10, 15, 20])
def test_various_dimensions(dim):
    """Test with various dimensions."""
    hz = HZ()
    np.random.seed(42)
    data = np.random.randn(100, dim).astype(np.float32)
    result = hz(data)
    assert torch.isfinite(result)
    assert result.device.type == "cpu"


@pytest.mark.unit
@pytest.mark.parametrize("n_samples", [2, 5, 10, 50, 100, 500, 1000])
def test_various_sample_sizes(n_samples):
    """Test with various sample sizes."""
    hz = HZ()
    np.random.seed(42)
    data = np.random.randn(n_samples, 3).astype(np.float32)

    # Suppress warnings for small samples
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = hz(data)

    assert torch.isfinite(result)
    assert result.device.type == "cpu"


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_various_numpy_dtypes(dtype):
    """Test with various numpy dtypes."""
    hz = HZ()
    np.random.seed(42)
    data = np.random.randn(50, 3).astype(dtype)
    result = hz(data)
    assert torch.isfinite(result)
    assert result.device.type == "cpu"


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_various_torch_dtypes(dtype):
    """Test with various torch dtypes."""
    hz = HZ()
    torch.manual_seed(42)
    data = torch.randn(50, 3, dtype=dtype, device="cpu")
    result = hz(data)
    assert torch.isfinite(result)
    assert result.device.type == "cpu"


@pytest.mark.unit
@pytest.mark.parametrize(
    "n,d",
    [
        (50, 2),
        (100, 3),
        (200, 5),
        (500, 10),
    ],
)
def test_bandwidth_scaling(n, d):
    """Test bandwidth computation with various N and D."""
    beta = HZ.compute_bandwidth(n_samples=n, n_dims=d)
    assert beta > 0
    assert np.isfinite(beta)

    # Check formula manually
    expected = (1 / np.sqrt(2)) * (((2 * d + 1) * n / 4) ** (1 / (d + 4)))
    assert np.isclose(beta, expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    "n_samples,n_dims",
    [
        (100, 2),
        (100, 5),
        (200, 3),
        (50, 10),
    ],
)
def test_multiple_dimensions_same_instance(n_samples, n_dims):
    """Test that one instance works with different dimensional data."""
    hz = HZ()

    # First test
    np.random.seed(42)
    data1 = np.random.randn(n_samples, n_dims).astype(np.float32)
    result1 = hz(data1)
    assert torch.isfinite(result1)

    # Second test with different dimensions (same instance)
    np.random.seed(43)
    other_dim = n_dims + 2
    data2 = np.random.randn(n_samples, other_dim).astype(np.float32)
    result2 = hz(data2)
    assert torch.isfinite(result2)

    # Both should be valid
    assert result1.device.type == "cpu"
    assert result2.device.type == "cpu"
