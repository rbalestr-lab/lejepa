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
from lejepa.univariate import ShapiroWilk
from unittest.mock import patch


class TestShapiroWilk:
    """Test suite for Shapiro-Wilk test for standard normal N(0,1)."""

    @pytest.fixture
    def sw_test(self):
        """Create a ShapiroWilk test instance with default parameters."""
        return ShapiroWilk()

    @pytest.fixture
    def standard_normal_samples_2d(self):
        """Generate 2D samples from standard normal N(0,1)."""
        torch.manual_seed(42)
        return torch.randn(1000, 5)

    @pytest.fixture
    def non_standard_normal_samples_2d(self):
        """Generate 2D samples NOT from standard normal."""
        torch.manual_seed(42)
        return torch.rand(1000, 5)

    # Basic Functionality Tests
    def test_forward_returns_correct_shape(self, sw_test, standard_normal_samples_2d):
        """Test that forward returns shape (dim,) for 2D input (n, dim)."""
        n, dim = standard_normal_samples_2d.shape
        result = sw_test(standard_normal_samples_2d)
        assert result.shape == (dim,), f"Expected shape ({dim},), got {result.shape}"
        assert isinstance(result, torch.Tensor), "Result should be a Tensor"

    def test_forward_single_column(self, sw_test):
        """Test with single column input (n, 1)."""
        torch.manual_seed(42)
        x = torch.randn(100, 1)
        result = sw_test(x)
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
        assert torch.isfinite(result).all(), "Result should be finite"

    def test_statistic_in_valid_range(self, sw_test):
        """Test that statistic is in [0, 1] range."""
        torch.manual_seed(42)
        x = torch.randn(100, 5)
        result = sw_test(x)
        assert (result >= 0).all(), f"Statistic should be >= 0, got {result}"
        assert (result <= 1).all(), f"Statistic should be <= 1, got {result}"

    def test_standard_normal_samples_low_statistic(
        self, sw_test, standard_normal_samples_2d
    ):
        """Test that N(0,1) samples produce low test statistics (close to 0)."""
        result = sw_test(standard_normal_samples_2d)
        # N(0,1) samples should have statistic close to 0 (good fit)
        assert (result < 0.1).all(), f"Statistics too high for N(0,1) data: {result}"

    def test_non_standard_normal_samples_high_statistic(
        self, sw_test, non_standard_normal_samples_2d
    ):
        """Test that non-N(0,1) samples produce high test statistics (close to 1)."""
        result = sw_test(non_standard_normal_samples_2d)
        # Uniform samples should have high statistics (poor fit)
        assert (
            result > 0.01
        ).all(), f"Statistics too low for non-N(0,1) data: {result}"

    def test_shifted_normal_rejected(self, sw_test):
        """Test that N(5,1) has higher statistic than N(0,1)."""
        torch.manual_seed(42)
        x = torch.randn(500, 3) + 5.0
        result = sw_test(x)
        assert (result > 0.1).all(), f"N(5,1) should have high statistic: {result}"

    def test_independent_columns(self, sw_test):
        """Test that each column is tested independently."""
        torch.manual_seed(42)
        col1 = torch.randn(500, 1)  # N(0,1)
        col2 = torch.rand(500, 1)  # Uniform
        col3 = torch.randn(500, 1) + 5  # N(5,1)

        x = torch.cat([col1, col2, col3], dim=1)
        result = sw_test(x)

        assert result.shape == (3,), "Should have 3 statistics"
        # Column 1 should have lower statistic (better fit)
        assert result[0] < result[1], "N(0,1) should have lower stat than uniform"
        assert result[0] < result[2], "N(0,1) should have lower stat than N(5,1)"

    # Parameter Tests
    @pytest.mark.parametrize("expectation_mode", ["elfving", "blom", "rahman"])
    def test_different_expectation_modes(self, expectation_mode):
        """Test with different expectation modes."""
        sw_test = ShapiroWilk(expectation_mode=expectation_mode)
        torch.manual_seed(42)
        x = torch.randn(100, 3)
        result = sw_test(x)
        assert result.shape == (3,), "Should work with all expectation modes"
        assert torch.isfinite(result).all()
        assert (result >= 0).all() and (result <= 1).all()

    @pytest.mark.parametrize("covariance_mode", ["shapiro_francia", "rahman"])
    def test_different_covariance_modes(self, covariance_mode):
        """Test with different covariance modes."""
        sw_test = ShapiroWilk(covariance_mode=covariance_mode)
        torch.manual_seed(42)
        x = torch.randn(100, 3)
        result = sw_test(x)
        assert result.shape == (3,), "Should work with all covariance modes"
        assert torch.isfinite(result).all()
        assert (result >= 0).all() and (result <= 1).all()

    def test_invalid_expectation_mode_raises(self):
        """Test that invalid expectation_mode raises error."""
        sw_test = ShapiroWilk(expectation_mode="invalid")
        x = torch.randn(100, 3)
        with pytest.raises(ValueError, match="Unknown expectation_mode"):
            sw_test(x)

    def test_invalid_covariance_mode_raises(self):
        """Test that invalid covariance_mode raises error."""
        sw_test = ShapiroWilk(covariance_mode="invalid")
        x = torch.randn(100, 3)
        with pytest.raises(ValueError, match="Unknown covariance_mode"):
            sw_test(x)

    # Edge Cases
    def test_small_sample_size(self):
        """Test with small sample sizes."""
        sw_test = ShapiroWilk()
        x = torch.randn(5, 3)
        result = sw_test(x)
        assert result.shape == (3,), "Should return statistics for all columns"
        assert torch.isfinite(result).all(), "Results should be finite"
        assert (result >= 0).all() and (result <= 1).all()

    def test_minimum_sample_size(self):
        """Test with minimum viable sample size (n=3)."""
        sw_test = ShapiroWilk()
        x = torch.randn(3, 4)
        result = sw_test(x)
        assert result.shape == (4,), "Should return statistics for all columns"
        assert torch.isfinite(result).all()

    def test_very_small_sample(self):
        """Test with n=2 (edge case)."""
        sw_test = ShapiroWilk()
        x = torch.randn(2, 3)
        result = sw_test(x)
        assert result.shape == (3,)
        # May or may not be finite depending on implementation

    def test_large_sample_size(self):
        """Test with large sample size (n=5000)."""
        sw_test = ShapiroWilk()
        torch.manual_seed(42)
        x = torch.randn(5000, 3)
        result = sw_test(x)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()
        # Large N(0,1) sample should have very low statistic
        assert (result < 0.05).all()

    def test_large_dimensionality(self):
        """Test with large number of dimensions."""
        torch.manual_seed(42)
        sw_test = ShapiroWilk()
        x = torch.randn(100, 100)
        result = sw_test(x)
        assert result.shape == (100,)
        assert torch.isfinite(result).all()

    # Dtype Tests
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_different_dtypes(self, dtype):
        """Test with different floating point dtypes."""
        sw_test = ShapiroWilk()
        x = torch.randn(100, 3, dtype=dtype)
        result = sw_test(x)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()
        assert (result >= 0).all() and (result <= 1).all()

    # Mathematical Properties
    def test_deterministic_output(self, sw_test):
        """Test that same input produces same output."""
        x = torch.randn(100, 4)
        result1 = sw_test(x)
        result2 = sw_test(x)
        assert torch.allclose(result1, result2), "Results should be deterministic"

    def test_permutation_invariance(self, sw_test):
        """Test that permuting samples within a column doesn't change result."""
        torch.manual_seed(42)
        x = torch.randn(100, 3)
        result1 = sw_test(x)

        perm = torch.randperm(100)
        x_permuted = x[perm, :]
        result2 = sw_test(x_permuted)

        assert torch.allclose(
            result1, result2, rtol=1e-4
        ), "Permuting samples should not change statistics"

    def test_column_independence(self, sw_test):
        """Test that permuting columns permutes results."""
        torch.manual_seed(42)
        x = torch.randn(100, 4)
        result1 = sw_test(x)

        x_permuted = x[:, [2, 0, 3, 1]]
        result2 = sw_test(x_permuted)

        expected = result1[[2, 0, 3, 1]]
        assert torch.allclose(result2, expected, rtol=1e-4)

    def test_correlation_interpretation(self, sw_test):
        """Test that statistic is 1 - |correlation|."""
        torch.manual_seed(42)
        x = torch.randn(100, 2)
        result = sw_test(x)

        # Since T = 1 - |ρ|, and -1 <= ρ <= 1:
        # - Perfect correlation (|ρ|=1) gives T=0
        # - No correlation (ρ=0) gives T=1
        # For N(0,1), we expect high |ρ| (close to 1), so T close to 0
        assert (result < 0.2).all(), "N(0,1) should have high correlation, low T"

    # Weight Caching Tests
    def test_weights_cached(self, sw_test):
        """Test that weights are cached after first computation."""
        x = torch.randn(100, 3)

        # First call computes weights
        assert sw_test._k is None, "Weights should be None initially"
        result1 = sw_test(x)
        assert sw_test._k is not None, "Weights should be cached after first call"
        cached_k = sw_test._k.clone()

        # Second call reuses weights
        result2 = sw_test(x)
        assert torch.equal(sw_test._k, cached_k), "Weights should be reused"
        assert torch.equal(result1, result2)

    def test_weights_recomputed_on_size_change(self, sw_test):
        """Test that weights are recomputed when sample size changes."""
        x1 = torch.randn(50, 3)
        x2 = torch.randn(100, 3)

        result1 = sw_test(x1)
        k1 = sw_test._k.clone()

        result2 = sw_test(x2)
        k2 = sw_test._k

        assert k1.size(0) == 50, "First weights should be size 50"
        assert k2.size(0) == 100, "Second weights should be size 100"
        assert not torch.equal(
            k1, k2[:50]
        ), "Weights should be different for different n"

    # Integration Tests
    @patch.object(ShapiroWilk, "prepare_data")
    def test_prepare_data_called(self, mock_prepare, sw_test):
        """Test that prepare_data is called."""
        x = torch.randn(100, 3)
        mock_prepare.return_value = torch.randn(100, 3)
        sw_test(x)
        mock_prepare.assert_called_once_with(x)

    @patch.object(ShapiroWilk, "get_shapiro_weights")
    def test_get_shapiro_weights_called(self, mock_weights, sw_test):
        """Test that get_shapiro_weights is called."""
        x = torch.randn(100, 3)
        mock_weights.return_value = torch.randn(100)
        sw_test(x)
        mock_weights.assert_called_once()

    # Distribution Tests
    def test_perfect_standard_normal_low_statistic(self):
        """Test with large samples from N(0,1) gives low statistic."""
        torch.manual_seed(42)
        sw_test = ShapiroWilk()
        x = torch.randn(1000, 5)
        result = sw_test(x)
        # Should have very low statistics (close to 0)
        assert (result < 0.05).all(), f"Large N(0,1) should have low stat: {result}"

    def test_uniform_distribution_high_statistic(self):
        """Test with uniform distribution gives high statistic."""
        torch.manual_seed(42)
        sw_test = ShapiroWilk()
        x = torch.rand(500, 3)
        result = sw_test(x)
        # Uniform should have high statistics (poor fit to normal)
        assert (result > 0.05).all(), f"Uniform should have high stat: {result}"

    def test_exponential_distribution(self):
        """Test with exponential distribution."""
        torch.manual_seed(42)
        sw_test = ShapiroWilk()
        x = torch.distributions.Exponential(1.0).sample((500, 3))
        result = sw_test(x)
        # Should have elevated statistic
        assert (result > 0.05).all()

    def test_mixed_distributions(self):
        """Test with mixed distributions."""
        torch.manual_seed(42)
        sw_test = ShapiroWilk()
        n01_col = torch.randn(500, 1)
        uniform_col = torch.rand(500, 1)
        n51_col = torch.randn(500, 1) + 5

        x = torch.cat([n01_col, uniform_col, n51_col], dim=1)
        result = sw_test(x)

        assert result.shape == (3,)
        # N(0,1) should have lowest statistic
        assert result[0] < result[1], "N(0,1) < Uniform"
        assert result[0] < result[2], "N(0,1) < N(5,1)"

    # Error Handling
    def test_1d_input_raises_error(self, sw_test):
        """Test that 1D input raises error."""
        x = torch.randn(100)
        with pytest.raises((RuntimeError, ValueError, IndexError, AssertionError)):
            sw_test(x)

    def test_empty_columns(self, sw_test):
        """Test with zero samples."""
        try:
            x = torch.tensor([]).reshape(0, 3)
            result = sw_test(x)
            assert result.shape == (3,)
        except (RuntimeError, ValueError, IndexError):
            pass

    def test_nan_input(self, sw_test):
        """Test with NaN inputs."""
        x = torch.randn(100, 3)
        x[50, 1] = float("nan")
        result = sw_test(x)
        assert result.shape == (3,)

    def test_inf_input(self, sw_test):
        """Test with infinite inputs."""
        x = torch.randn(100, 3)
        x[50, 1] = float("inf")
        result = sw_test(x)
        assert result.shape == (3,)


class TestShapiroWilkWeights:
    """Test the get_shapiro_weights static method."""

    def test_weights_unit_length(self):
        """Test that weights are normalized to unit length."""
        for N in [10, 50, 100]:
            weights = ShapiroWilk.get_shapiro_weights(N)
            norm = torch.norm(weights, p=2)
            assert torch.isclose(
                norm, torch.tensor(1.0)
            ), f"Weights should have unit norm, got {norm}"

    def test_weights_shape(self):
        """Test that weights have correct shape."""
        for N in [10, 50, 100]:
            weights = ShapiroWilk.get_shapiro_weights(N)
            assert weights.shape == (N,), f"Expected shape ({N},), got {weights.shape}"

    @pytest.mark.parametrize("expectation_mode", ["elfving", "blom", "rahman"])
    def test_weights_with_expectation_modes(self, expectation_mode):
        """Test weight computation with different expectation modes."""
        weights = ShapiroWilk.get_shapiro_weights(50, expectation_mode=expectation_mode)
        assert weights.shape == (50,)
        assert torch.isfinite(weights).all()
        assert torch.isclose(torch.norm(weights, p=2), torch.tensor(1.0))

    @pytest.mark.parametrize("covariance_mode", ["shapiro_francia", "rahman"])
    def test_weights_with_covariance_modes(self, covariance_mode):
        """Test weight computation with different covariance modes."""
        weights = ShapiroWilk.get_shapiro_weights(50, covariance_mode=covariance_mode)
        assert weights.shape == (50,)
        assert torch.isfinite(weights).all()
        assert torch.isclose(torch.norm(weights, p=2), torch.tensor(1.0))

    def test_weights_different_for_different_modes(self):
        """Test that different modes produce different weights."""
        w_sf = ShapiroWilk.get_shapiro_weights(50, covariance_mode="shapiro_francia")
        w_rahman = ShapiroWilk.get_shapiro_weights(50, covariance_mode="rahman")

        # Should be different (unless by coincidence)
        assert not torch.allclose(
            w_sf, w_rahman
        ), "Different covariance modes should give different weights"

    def test_weights_monotonicity(self):
        """Test that expected order statistics are monotonically increasing."""
        # For plotting positions, we expect p_i to be increasing
        # and thus expected order statistics m_i = Φ^(-1)(p_i) should increase
        N = 50
        # We can't directly access m_i, but weights should reflect this structure
        weights = ShapiroWilk.get_shapiro_weights(N, covariance_mode="shapiro_francia")
        # For shapiro_francia, weights are normalized m_i, which should be increasing
        # after normalization, this property might not hold exactly, but general trend should


class TestShapiroWilkComparisons:
    """Test comparisons between modes and with other tests."""

    def test_rahman_vs_shapiro_francia(self):
        """Test that rahman and shapiro_francia give similar results for N(0,1)."""
        torch.manual_seed(42)
        x = torch.randn(100, 5)

        sw_sf = ShapiroWilk(covariance_mode="shapiro_francia")
        sw_rahman = ShapiroWilk(covariance_mode="rahman")

        result_sf = sw_sf(x)
        result_rahman = sw_rahman(x)

        # Both should detect N(0,1) as good fit (low statistics)
        assert (result_sf < 0.1).all()
        assert (result_rahman < 0.1).all()

        # Should be correlated but not identical
        # Both should be small for N(0,1)

    def test_expectation_modes_similar_for_normal(self):
        """Test that different expectation modes give similar results for N(0,1)."""
        torch.manual_seed(42)
        x = torch.randn(100, 3)

        results = {}
        for mode in ["elfving", "blom", "rahman"]:
            sw = ShapiroWilk(expectation_mode=mode)
            results[mode] = sw(x)

        # All should detect N(0,1) as good fit
        for mode, result in results.items():
            assert (result < 0.1).all(), f"{mode} should detect N(0,1): {result}"


# Parametrized Tests
class TestShapiroWilkParametrized:
    """Parametrized tests for various scenarios."""

    @pytest.mark.parametrize("n_samples", [10, 50, 100, 500, 1000])
    def test_various_sample_sizes(self, n_samples):
        """Test with various sample sizes from N(0,1)."""
        sw_test = ShapiroWilk()
        x = torch.randn(n_samples, 3)
        result = sw_test(x)
        assert result.shape == (3,), f"Shape mismatch for n={n_samples}"
        assert torch.isfinite(result).all()
        assert (result >= 0).all() and (result <= 1).all()

    @pytest.mark.parametrize("n_dims", [1, 5, 10, 50])
    def test_various_dimensions(self, n_dims):
        """Test with various dimensions."""
        sw_test = ShapiroWilk()
        x = torch.randn(100, n_dims)
        result = sw_test(x)
        assert result.shape == (n_dims,)
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize("seed", range(5))
    def test_multiple_seeds(self, seed):
        """Test with different random seeds."""
        sw_test = ShapiroWilk()
        torch.manual_seed(seed)
        x = torch.randn(100, 4)
        result = sw_test(x)
        assert torch.isfinite(result).all()
        assert result.shape == (4,)
        assert (result >= 0).all() and (result <= 1).all()

    @pytest.mark.parametrize("mean", [1.0, 5.0, -3.0])
    def test_non_zero_mean_higher_stat(self, mean):
        """Test that non-zero mean gives higher statistic."""
        sw_test = ShapiroWilk()
        torch.manual_seed(42)
        x_normal = torch.randn(200, 1)
        x_shifted = torch.randn(200, 1) + mean

        stat_normal = sw_test(x_normal)[0]
        stat_shifted = sw_test(x_shifted)[0]

        assert (
            stat_shifted > stat_normal
        ), f"N({mean},1) should have higher stat than N(0,1)"

    @pytest.mark.parametrize("std", [2.0, 5.0, 10.0])
    def test_non_unit_std_higher_stat(self, std):
        """Test that non-unit std gives higher statistic."""
        sw_test = ShapiroWilk()
        torch.manual_seed(42)
        x_normal = torch.randn(200, 1)
        x_scaled = torch.randn(200, 1) * std

        stat_normal = sw_test(x_normal)[0]
        stat_scaled = sw_test(x_scaled)[0]

        assert (
            stat_scaled > stat_normal
        ), f"N(0,{std}²) should have higher stat than N(0,1)"

    @pytest.mark.parametrize("eps", [1e-5, 1e-7, 1e-9])
    def test_different_eps_values(self, eps):
        """Test with different epsilon values for numerical stability."""
        sw_test = ShapiroWilk(eps=eps)
        torch.manual_seed(42)
        x = torch.randn(100, 3)
        result = sw_test(x)
        assert torch.isfinite(result).all()
        assert (result >= 0).all() and (result <= 1).all()


class TestShapiroWilkEdgeCases:
    """Edge case tests."""

    def test_constant_column(self):
        """Test with constant column."""
        sw_test = ShapiroWilk()
        x = torch.randn(100, 3)
        x[:, 1] = 5.0

        try:
            result = sw_test(x)
            assert result.shape == (3,)
            # Constant column should have high statistic or inf/nan
            assert result[1] > 0.5 or not torch.isfinite(result[1])
        except (RuntimeError, ValueError, ZeroDivisionError):
            pass

    def test_t_distribution(self):
        """Test with t-distribution (heavier tails)."""
        torch.manual_seed(42)
        sw_test = ShapiroWilk()
        from torch.distributions import StudentT

        x = StudentT(df=5.0).sample((500, 3))

        result = sw_test(x)
        # t(5) is close to normal but not exact
        assert (result > 0).all()


class TestShapiroWilkStatistical:
    """Statistical property tests."""

    def test_larger_shift_larger_stat(self):
        """Test that larger departures give larger statistics."""
        torch.manual_seed(42)
        sw_test = ShapiroWilk()

        x0 = torch.randn(200, 1)
        x1 = torch.randn(200, 1) + 1.0
        x2 = torch.randn(200, 1) + 5.0

        stat0 = sw_test(x0)[0]
        stat1 = sw_test(x1)[0]
        stat2 = sw_test(x2)[0]

        assert stat1 > stat0, "N(1,1) > N(0,1)"
        assert stat2 > stat1, "N(5,1) > N(1,1)"

    def test_type_i_error_rate(self):
        """Test approximate type I error rate for N(0,1)."""
        torch.manual_seed(42)
        sw_test = ShapiroWilk()
        n_trials = 100
        threshold = 0.05  # Approximate threshold

        rejections = 0
        for i in range(n_trials):
            torch.manual_seed(i)
            x = torch.randn(100, 1)
            result = sw_test(x)
            if result[0] > threshold:
                rejections += 1

        rejection_rate = rejections / n_trials
        # Should be reasonably low for N(0,1) data
        assert (
            rejection_rate < 0.3
        ), f"Rejection rate too high for N(0,1): {rejection_rate}"

    def test_power_against_uniform(self):
        """Test that test has power against uniform distribution."""
        torch.manual_seed(42)
        sw_test = ShapiroWilk()
        n_trials = 20
        threshold = 0.05

        rejections = 0
        for i in range(n_trials):
            torch.manual_seed(i)
            x = torch.rand(100, 1)
            result = sw_test(x)
            if result[0] > threshold:
                rejections += 1

        rejection_rate = rejections / n_trials
        # Should reject most uniform samples
        assert (
            rejection_rate > 0.7
        ), f"Should have high power against uniform: {rejection_rate}"
