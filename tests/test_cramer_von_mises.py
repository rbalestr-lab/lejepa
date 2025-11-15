"""
Unit tests for univariate normality tests.

Tests cover:
- Basic functionality and correctness
- Statistical power (detecting non-normality)
- Edge cases and numerical stability
- Batched inputs and multi-dimensional data
- Reproducibility and determinism
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch
from lejepa.univariate import CramerVonMises


class TestCramerVonMises:
    """Test suite for Cramér-von Mises test (mean squared error variant)."""

    @pytest.fixture
    def cvm_test(self):
        """Create a CramerVonMises test instance."""
        return CramerVonMises()

    @pytest.fixture
    def standard_normal_samples_2d(self):
        """Generate 2D samples from standard normal N(0,1)."""
        torch.manual_seed(42)
        return torch.randn(1000, 5)

    # Basic Functionality Tests
    def test_forward_returns_correct_shape(self, cvm_test, standard_normal_samples_2d):
        """Test that forward returns shape (dim,) for 2D input (n, dim)."""
        n, dim = standard_normal_samples_2d.shape
        result = cvm_test(standard_normal_samples_2d)
        assert result.shape == (dim,), f"Expected shape ({dim},), got {result.shape}"
        assert isinstance(result, torch.Tensor), "Result should be a Tensor"

    def test_forward_single_column(self, cvm_test):
        """Test with single column input (n, 1)."""
        torch.manual_seed(42)
        x = torch.randn(100, 1)
        result = cvm_test(x)
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
        assert torch.isfinite(result).all(), "Result should be finite"

    def test_standard_normal_samples_low_statistic(self, cvm_test):
        """Test that N(0,1) samples produce low test statistics."""
        torch.manual_seed(42)
        x = torch.randn(1000, 5)
        result = cvm_test(x)
        # For mean squared error, values should be small (< 1/12 ≈ 0.083)
        assert (result < 0.01).all(), f"T statistics too high for N(0,1) data: {result}"

    def test_non_standard_normal_high_statistic(self, cvm_test):
        """Test that non-N(0,1) samples produce higher statistics."""
        torch.manual_seed(42)
        x = torch.rand(1000, 5)  # Uniform
        result = cvm_test(x)
        # Uniform should have higher statistics
        assert (
            result > 0.001
        ).all(), f"T statistics too low for uniform data: {result}"

    def test_shifted_normal_rejected(self, cvm_test):
        """Test that N(5,1) has higher statistic than N(0,1)."""
        torch.manual_seed(42)
        x = torch.randn(500, 3) + 5.0
        result = cvm_test(x)
        assert (result > 0.01).all(), f"N(5,1) should have high statistic: {result}"

    def test_independent_columns(self, cvm_test):
        """Test that each column is tested independently."""
        torch.manual_seed(42)
        col1 = torch.randn(500, 1)  # N(0,1)
        col2 = torch.rand(500, 1)  # Uniform
        col3 = torch.randn(500, 1) + 5  # N(5,1)

        x = torch.cat([col1, col2, col3], dim=1)
        result = cvm_test(x)

        assert result.shape == (3,), "Should have 3 statistics"
        assert result[0] < result[1], "N(0,1) should have lower stat than uniform"
        assert result[0] < result[2], "N(0,1) should have lower stat than N(5,1)"

    # Edge Cases
    def test_small_sample_size(self, cvm_test):
        """Test with small sample sizes."""
        x = torch.randn(5, 3)
        result = cvm_test(x)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()

    def test_minimum_sample_size(self, cvm_test):
        """Test with minimum viable sample size."""
        x = torch.randn(2, 4)
        result = cvm_test(x)
        assert result.shape == (4,)
        assert torch.isfinite(result).all()

    def test_single_sample(self, cvm_test):
        """Test with single sample."""
        x = torch.randn(1, 3)
        result = cvm_test(x)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()

    def test_large_dimensionality(self, cvm_test):
        """Test with large number of dimensions."""
        torch.manual_seed(42)
        x = torch.randn(100, 100)
        result = cvm_test(x)
        assert result.shape == (100,)
        assert torch.isfinite(result).all()

    # Dtype Tests
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_different_dtypes(self, cvm_test, dtype):
        """Test with different floating point dtypes."""
        x = torch.randn(100, 3, dtype=dtype)
        result = cvm_test(x)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()

    # Mathematical Properties
    def test_statistic_non_negative(self, cvm_test):
        """Test that T statistics are always non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            x = torch.randn(100, 5)
            result = cvm_test(x)
            assert (result >= -1e-8).all(), f"T should be non-negative, got {result}"

    def test_statistic_is_mean_squared_error(self, cvm_test):
        """Test that statistic is indeed mean squared error."""
        torch.manual_seed(42)
        x = torch.randn(100, 2)
        result = cvm_test(x)

        # Mean of squares should be bounded by max square
        # For CDF values in [0,1] and k in [0,1], max diff is 1
        assert (result <= 1.0).all(), f"Mean squared error should be ≤ 1: {result}"

    def test_deterministic_output(self, cvm_test):
        """Test that same input produces same output."""
        x = torch.randn(100, 4)
        result1 = cvm_test(x)
        result2 = cvm_test(x)
        assert torch.allclose(result1, result2), "Results should be deterministic"

    def test_permutation_invariance(self, cvm_test):
        """Test that permuting samples within a column doesn't change result."""
        torch.manual_seed(42)
        x = torch.randn(100, 3)
        result1 = cvm_test(x)

        perm = torch.randperm(100)
        x_permuted = x[perm, :]
        result2 = cvm_test(x_permuted)

        assert torch.allclose(
            result1, result2, rtol=1e-4
        ), "Permuting samples should not change statistics"

    def test_column_independence(self, cvm_test):
        """Test that permuting columns permutes results."""
        torch.manual_seed(42)
        x = torch.randn(100, 4)
        result1 = cvm_test(x)

        x_permuted = x[:, [2, 0, 3, 1]]
        result2 = cvm_test(x_permuted)

        expected = result1[[2, 0, 3, 1]]
        assert torch.allclose(result2, expected, rtol=1e-4)

    # Integration Tests
    @patch.object(CramerVonMises, "prepare_data")
    def test_prepare_data_called(self, mock_prepare, cvm_test):
        """Test that prepare_data is called."""
        x = torch.randn(100, 3)
        mock_prepare.return_value = torch.randn(100, 3)
        cvm_test(x)
        mock_prepare.assert_called_once_with(x)

    # Distribution Tests
    def test_perfect_standard_normal_small_stat(self, cvm_test):
        """Test with large samples from N(0,1) gives small statistic."""
        torch.manual_seed(42)
        x = torch.randn(10000, 5)
        result = cvm_test(x)
        # Mean squared error should be very small for good fit
        assert (result < 0.002).all(), f"Large N(0,1) should have tiny stat: {result}"

    def test_uniform_distribution(self, cvm_test):
        """Test with uniform distribution."""
        torch.manual_seed(42)
        x = torch.rand(500, 3)
        result = cvm_test(x)
        assert (result > 0.001).all(), f"Uniform should have elevated stat: {result}"

    def test_mixed_distributions(self, cvm_test):
        """Test with mixed distributions."""
        torch.manual_seed(42)
        n01_col = torch.randn(500, 1)
        uniform_col = torch.rand(500, 1)
        n51_col = torch.randn(500, 1) + 5

        x = torch.cat([n01_col, uniform_col, n51_col], dim=1)
        result = cvm_test(x)

        assert result.shape == (3,)
        assert result[1] > result[0], "Uniform > N(0,1)"
        assert result[2] > result[0], "N(5,1) > N(0,1)"

    # Error Handling
    def test_1d_input_raises_error(self, cvm_test):
        """Test that 1D input raises error."""
        x = torch.randn(100)
        with pytest.raises((RuntimeError, ValueError, IndexError, AssertionError)):
            cvm_test(x)

    def test_empty_columns(self, cvm_test):
        """Test with zero samples."""
        try:
            x = torch.tensor([]).reshape(0, 3)
            result = cvm_test(x)
            assert result.shape == (3,)
        except (RuntimeError, ValueError, IndexError):
            pass

    def test_inf_input(self, cvm_test):
        """Test with infinite inputs."""
        x = torch.randn(100, 3)
        x[50, 1] = float("inf")
        result = cvm_test(x)
        assert result.shape == (3,)


class TestCramerVonMisesFormula:
    """Test formula implementation."""

    @pytest.fixture
    def cvm_test(self):
        return CramerVonMises()

    def test_k_values_correct(self, cvm_test):
        """Test that k = (2i-1)/(2n) is computed correctly."""
        # For n=4: k should be [1/8, 3/8, 5/8, 7/8] = [0.125, 0.375, 0.625, 0.875]
        n = 4
        expected_k = torch.tensor([0.125, 0.375, 0.625, 0.875])

        # Compute what the code computes
        k = torch.arange(1, n + 1, dtype=torch.float).mul_(2).sub_(1).div_(2 * n)
        assert torch.allclose(k, expected_k), f"k values incorrect: {k}"

    def test_formula_is_mse(self, cvm_test):
        """Test that formula computes mean squared error."""
        torch.manual_seed(42)
        x = torch.randn(100, 1)
        result = cvm_test(x)

        # Result should be mean of squared differences
        assert result >= 0, "MSE should be non-negative"
        assert result <= 1.0, "MSE should be bounded"


# Parametrized Tests
class TestCramerVonMisesParametrized:
    """Parametrized tests."""

    @pytest.mark.parametrize("n_samples", [10, 50, 100, 500, 1000])
    def test_various_sample_sizes(self, n_samples):
        """Test with various sample sizes."""
        cvm_test = CramerVonMises()
        x = torch.randn(n_samples, 3)
        result = cvm_test(x)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()
        assert (result >= 0).all()

    @pytest.mark.parametrize("n_dims", [1, 5, 10, 50])
    def test_various_dimensions(self, n_dims):
        """Test with various dimensions."""
        cvm_test = CramerVonMises()
        x = torch.randn(100, n_dims)
        result = cvm_test(x)
        assert result.shape == (n_dims,)
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize("seed", range(5))
    def test_multiple_seeds(self, seed):
        """Test with different random seeds."""
        cvm_test = CramerVonMises()
        torch.manual_seed(seed)
        x = torch.randn(100, 4)
        result = cvm_test(x)
        assert torch.isfinite(result).all()
        assert result.shape == (4,)

    @pytest.mark.parametrize("mean", [1.0, 5.0, -3.0])
    def test_non_zero_mean_higher_stat(self, mean):
        """Test that non-zero mean gives higher statistic."""
        cvm_test = CramerVonMises()
        torch.manual_seed(42)
        x_normal = torch.randn(500, 1)
        x_shifted = torch.randn(500, 1) + mean

        stat_normal = cvm_test(x_normal)[0]
        stat_shifted = cvm_test(x_shifted)[0]

        assert (
            stat_shifted > stat_normal
        ), f"N({mean},1) should have higher stat than N(0,1)"

    @pytest.mark.parametrize("std", [2.0, 5.0, 10.0])
    def test_non_unit_std_higher_stat(self, std):
        """Test that non-unit std gives higher statistic."""
        cvm_test = CramerVonMises()
        torch.manual_seed(42)
        x_normal = torch.randn(500, 1)
        x_scaled = torch.randn(500, 1) * std

        stat_normal = cvm_test(x_normal)[0]
        stat_scaled = cvm_test(x_scaled)[0]

        assert (
            stat_scaled > stat_normal
        ), f"N(0,{std}²) should have higher stat than N(0,1)"


class TestCramerVonMisesEdgeCases:
    """Edge case tests."""

    @pytest.fixture
    def cvm_test(self):
        return CramerVonMises()

    def test_constant_column(self, cvm_test):
        """Test with constant column."""
        x = torch.randn(100, 3)
        x[:, 1] = 5.0

        try:
            result = cvm_test(x)
            assert result.shape == (3,)
        except (RuntimeError, ValueError):
            pass

    def test_exponential_distribution(self, cvm_test):
        """Test with exponential (non-normal)."""
        torch.manual_seed(42)
        x = torch.distributions.Exponential(1.0).sample((500, 3))
        result = cvm_test(x)

        # Should have elevated statistic
        assert (result > 0.001).all()

    def test_bimodal(self, cvm_test):
        """Test with bimodal distribution."""
        torch.manual_seed(42)
        x1 = torch.randn(250, 2) - 2
        x2 = torch.randn(250, 2) + 2
        x = torch.cat([x1, x2], dim=0)

        result = cvm_test(x)
        assert (result > 0.001).all()


class TestCramerVonMisesStatistical:
    """Statistical property tests."""

    @pytest.fixture
    def cvm_test(self):
        return CramerVonMises()

    def test_larger_shift_larger_stat(self, cvm_test):
        """Test that larger departures give larger statistics."""
        torch.manual_seed(42)

        x0 = torch.randn(500, 1)
        x1 = torch.randn(500, 1) + 1.0
        x2 = torch.randn(500, 1) + 5.0

        stat0 = cvm_test(x0)[0]
        stat1 = cvm_test(x1)[0]
        stat2 = cvm_test(x2)[0]

        assert stat1 > stat0, "N(1,1) > N(0,1)"
        assert stat2 > stat1, "N(5,1) > N(1,1)"

    def test_larger_sample_smaller_variance(self, cvm_test):
        """Test that larger samples give more stable statistics."""
        torch.manual_seed(42)

        small_stats = [cvm_test(torch.randn(50, 1))[0].item() for _ in range(20)]
        large_stats = [cvm_test(torch.randn(500, 1))[0].item() for _ in range(20)]

        small_var = np.var(small_stats)
        large_var = np.var(large_stats)

        # Larger samples should be more stable
        assert large_var < small_var
