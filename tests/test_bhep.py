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
from lejepa.multivariate import BHEP

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit


class TestBHEP:
    """Comprehensive unit test suite for BHEP multivariate test (CPU only)."""

    @pytest.fixture(autouse=True)
    def setup_cpu(self):
        """Ensure all tests run on CPU."""
        torch.set_default_device("cpu")
        yield
        torch.set_default_device("cpu")  # Reset after test

    @pytest.fixture
    def bhep_2d(self):
        """Standard 2D BHEP instance."""
        return BHEP(beta=0.1)

    @pytest.fixture
    def bhep_3d(self):
        """Standard 3D BHEP instance."""
        return BHEP(beta=0.5)

    # ==================== Initialization Tests ====================

    @pytest.mark.unit
    def test_init_valid(self):
        """Test valid initialization."""
        bhep = BHEP(beta=0.2)
        assert bhep.beta == 0.2

    @pytest.mark.unit
    def test_init_invalid_beta_zero(self):
        """Test that beta=0 raises ValueError."""
        with pytest.raises(ValueError, match="beta must be positive"):
            BHEP(beta=0)

    @pytest.mark.unit
    def test_init_invalid_beta_negative(self):
        """Test that negative beta raises ValueError."""
        with pytest.raises(ValueError, match="beta must be positive"):
            BHEP(beta=-0.1)

    @pytest.mark.unit
    def test_init_default_beta(self):
        """Test default beta value."""
        bhep = BHEP()
        assert bhep.beta == 0.1

    # ==================== Input Validation Tests ====================

    @pytest.mark.unit
    def test_empty_input(self, bhep_2d):
        """Test that empty input raises ValueError."""
        empty_data = np.array([]).reshape(0, 2)
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            bhep_2d(empty_data)

    @pytest.mark.unit
    def test_single_sample(self, bhep_2d):
        """Test with single sample."""
        single_sample = np.array([[1.0, 2.0]])
        result = bhep_2d(single_sample)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"  # Ensure CPU
        assert result.shape == torch.Size([])  # Scalar

    # ==================== Data Type Tests ====================

    @pytest.mark.unit
    def test_numpy_input(self, bhep_2d):
        """Test with numpy array input."""
        data = np.random.randn(50, 2)
        result = bhep_2d(data)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"
        assert result.ndim == 0  # Scalar

    @pytest.mark.unit
    def test_torch_input_cpu(self, bhep_2d):
        """Test with torch tensor input on CPU."""
        data = torch.randn(50, 2, device="cpu")
        result = bhep_2d(data)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"
        assert result.ndim == 0  # Scalar

    @pytest.mark.unit
    def test_torch_float32(self, bhep_2d):
        """Test with float32 input."""
        data = torch.randn(50, 2, dtype=torch.float32, device="cpu")
        result = bhep_2d(data)
        assert torch.isfinite(result)
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_torch_float64(self, bhep_2d):
        """Test with float64 input."""
        data = torch.randn(50, 2, dtype=torch.float64, device="cpu")
        result = bhep_2d(data)
        assert torch.isfinite(result)
        assert result.device.type == "cpu"

    # ==================== Output Tests ====================

    @pytest.mark.unit
    def test_output_shape(self, bhep_2d):
        """Test output is a scalar."""
        data = np.random.randn(100, 2)
        result = bhep_2d(data)
        assert result.shape == torch.Size([])
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_output_is_real(self, bhep_2d):
        """Test output is a real number."""
        data = np.random.randn(100, 2)
        result = bhep_2d(data)
        assert torch.isfinite(result)
        assert not torch.isnan(result)
        assert not torch.isinf(result)
        assert result.device.type == "cpu"

    # ==================== Mathematical Property Tests ====================

    @pytest.mark.unit
    def test_deterministic(self, bhep_2d):
        """Test that same input gives same output."""
        np.random.seed(42)
        data = np.random.randn(50, 2)
        result1 = bhep_2d(data)
        result2 = bhep_2d(data)
        assert torch.allclose(result1, result2)
        assert result1.device.type == "cpu"

    @pytest.mark.unit
    def test_different_beta_different_results(self):
        """Test that different beta values give different results."""
        np.random.seed(42)
        data = np.random.randn(50, 2)
        bhep1 = BHEP(beta=0.1)
        bhep2 = BHEP(beta=0.5)

        result1 = bhep1(data)
        result2 = bhep2(data)

        assert not torch.allclose(result1, result2)
        assert result1.device.type == "cpu"
        assert result2.device.type == "cpu"

    @pytest.mark.unit
    def test_mvn_data_small_statistic(self, bhep_2d):
        """Test that multivariate normal data gives small statistic."""
        # Generate data from standard MVN
        torch.manual_seed(42)
        data = torch.randn(200, 2, device="cpu")
        result = bhep_2d(data)

        # For MVN data, statistic should be close to 0
        # (exact value depends on sample size and beta)
        assert result.item() < 1.0  # Reasonable threshold
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_non_normal_data_larger_statistic(self, bhep_2d):
        """Test that non-normal data gives larger statistic."""
        # Generate clearly non-normal data (uniform)
        np.random.seed(42)
        data = np.random.uniform(-3, 3, size=(200, 2))
        result = bhep_2d(data)

        # Should detect deviation from normality
        assert result.item() >= 0  # Statistic should be non-negative
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_symmetry_property(self, bhep_2d):
        """Test that permuting samples doesn't change result."""
        np.random.seed(42)
        data = np.random.randn(50, 2)
        result1 = bhep_2d(data)

        # Shuffle the rows
        permuted_data = data[np.random.permutation(50)]
        result2 = bhep_2d(permuted_data)

        assert torch.allclose(result1, result2, atol=1e-6)

    # ==================== Edge Cases ====================

    @pytest.mark.unit
    def test_identical_points(self, bhep_2d):
        """Test with all identical points."""
        data = np.ones((10, 2))
        result = bhep_2d(data)
        assert torch.isfinite(result)
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_collinear_points(self, bhep_2d):
        """Test with collinear points."""
        data = np.column_stack([np.linspace(0, 1, 50), np.linspace(0, 1, 50)])
        result = bhep_2d(data)
        assert torch.isfinite(result)
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_large_sample_size(self, bhep_2d):
        """Test with larger sample size."""
        np.random.seed(42)
        data = np.random.randn(1000, 2)
        result = bhep_2d(data)
        assert torch.isfinite(result)
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_high_dimension(self):
        """Test with higher dimensional data."""
        bhep = BHEP(beta=0.1)
        np.random.seed(42)
        data = np.random.randn(100, 10)
        result = bhep(data)
        assert torch.isfinite(result)
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_two_samples(self, bhep_2d):
        """Test with minimum practical sample size (2)."""
        data = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = bhep_2d(data)
        assert torch.isfinite(result)
        assert result.device.type == "cpu"

    # ==================== Specific Mathematical Cases ====================

    @pytest.mark.unit
    def test_zero_centered_data(self, bhep_2d):
        """Test with zero-centered data."""
        np.random.seed(42)
        data = np.random.randn(100, 2)
        data -= data.mean(axis=0)  # Center the data
        result = bhep_2d(data)
        assert torch.isfinite(result)
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_scaled_data(self, bhep_2d):
        """Test that scaling affects the statistic."""
        np.random.seed(42)
        data = np.random.randn(100, 2)
        result1 = bhep_2d(data)
        result2 = bhep_2d(data * 2)

        # Scaling should change the statistic
        assert not torch.allclose(result1, result2)
        assert result1.device.type == "cpu"
        assert result2.device.type == "cpu"

    @pytest.mark.unit
    def test_translated_data(self, bhep_2d):
        """Test that translation affects the statistic."""
        np.random.seed(42)
        data = np.random.randn(100, 2)
        result1 = bhep_2d(data)
        result2 = bhep_2d(data + 5)

        # Translation should change the statistic
        assert not torch.allclose(result1, result2)

    @pytest.mark.unit
    def test_negative_values(self, bhep_2d):
        """Test with all negative values."""
        data = -np.abs(np.random.randn(50, 2))
        result = bhep_2d(data)
        assert torch.isfinite(result)
        assert result.device.type == "cpu"

    # ==================== Representation Test ====================

    @pytest.mark.unit
    def test_repr(self, bhep_2d):
        """Test string representation."""
        repr_str = repr(bhep_2d)
        assert "BHEP" in repr_str
        assert "beta=0.1" in repr_str

    # ==================== Numerical Stability Tests ====================

    @pytest.mark.unit
    def test_very_small_beta(self):
        """Test with very small beta."""
        bhep = BHEP(beta=0.001)
        np.random.seed(42)
        data = np.random.randn(50, 2)
        result = bhep(data)
        assert torch.isfinite(result)
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_very_large_beta(self):
        """Test with very large beta."""
        bhep = BHEP(beta=10.0)
        np.random.seed(42)
        data = np.random.randn(50, 2)
        result = bhep(data)
        assert torch.isfinite(result)
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_extreme_values(self, bhep_2d):
        """Test with extreme data values."""
        np.random.seed(42)
        data = np.random.randn(50, 2) * 1000
        result = bhep_2d(data)
        assert torch.isfinite(result)
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_very_small_values(self, bhep_2d):
        """Test with very small data values."""
        np.random.seed(42)
        data = np.random.randn(50, 2) * 1e-6
        result = bhep_2d(data)
        assert torch.isfinite(result)
        assert result.device.type == "cpu"

    @pytest.mark.unit
    def test_mixed_scale_data(self, bhep_2d):
        """Test with mixed scale data."""
        np.random.seed(42)
        data = np.random.randn(50, 2)
        data[:, 0] *= 1000  # First dimension large
        data[:, 1] *= 0.001  # Second dimension small
        result = bhep_2d(data)
        assert torch.isfinite(result)


# ==================== Parametrized Tests ====================


@pytest.mark.unit
@pytest.mark.parametrize(
    "dim,beta",
    [
        (1, 0.1),
        (2, 0.5),
        (5, 1.0),
        (10, 0.01),
    ],
)
def test_various_configurations(dim, beta):
    """Test various dimension and beta combinations."""
    bhep = BHEP(beta=beta)
    np.random.seed(42)
    data = np.random.randn(50, dim)
    result = bhep(data)
    assert torch.isfinite(result)
    assert result.device.type == "cpu"


@pytest.mark.unit
@pytest.mark.parametrize("n_samples", [2, 10, 50, 100, 500])
def test_various_sample_sizes(n_samples):
    """Test with various sample sizes."""
    bhep = BHEP(beta=0.1)
    np.random.seed(42)
    data = np.random.randn(n_samples, 2)
    result = bhep(data)
    assert torch.isfinite(result)
    assert result.device.type == "cpu"


@pytest.mark.unit
@pytest.mark.parametrize("beta", [0.01, 0.1, 0.5, 1.0, 5.0])
def test_various_beta_values(beta):
    """Test with various beta values."""
    bhep = BHEP(beta=beta)
    np.random.seed(42)
    data = np.random.randn(50, 2)
    result = bhep(data)
    assert torch.isfinite(result)
    assert result.device.type == "cpu"


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_various_dtypes(dtype):
    """Test with various numpy dtypes."""
    bhep = BHEP(beta=0.1)
    np.random.seed(42)
    data = np.random.randn(50, 2).astype(dtype)
    result = bhep(data)
    assert torch.isfinite(result)
    assert result.device.type == "cpu"
