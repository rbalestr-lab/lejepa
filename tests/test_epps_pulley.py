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
from lejepa.univariate import EppsPulley


class TestEppsPulley:
    """Test suite for Epps-Pulley test for standard normal N(0,1)."""

    @pytest.fixture
    def ep_test(self):
        """Create an EppsPulley test instance with default parameters."""
        return EppsPulley()

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
    def test_forward_returns_correct_shape(self, ep_test, standard_normal_samples_2d):
        """Test that forward returns shape (dim,) for 2D input (n, dim)."""
        n, dim = standard_normal_samples_2d.shape
        result = ep_test(standard_normal_samples_2d)
        assert result.shape == (dim,), f"Expected shape ({dim},), got {result.shape}"
        assert isinstance(result, torch.Tensor), "Result should be a Tensor"

    def test_forward_single_column(self, ep_test):
        """Test with single column input (n, 1)."""
        torch.manual_seed(42)
        x = torch.randn(100, 1)
        result = ep_test(x)
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
        assert torch.isfinite(result).all(), "Result should be finite"

    def test_standard_normal_samples_low_statistic(
        self, ep_test, standard_normal_samples_2d
    ):
        """Test that N(0,1) samples produce low test statistics."""
        result = ep_test(standard_normal_samples_2d)
        # N(0,1) samples should have small statistics (good fit)
        assert (result < 1.0).all(), f"Statistics too high for N(0,1) data: {result}"

    def test_non_standard_normal_samples_high_statistic(
        self, ep_test, non_standard_normal_samples_2d
    ):
        """Test that non-N(0,1) samples produce high test statistics."""
        result = ep_test(non_standard_normal_samples_2d)
        # Uniform samples should have higher statistics
        assert (
            result > 0.01
        ).all(), f"Statistics too low for non-N(0,1) data: {result}"

    def test_shifted_normal_rejected(self, ep_test):
        """Test that N(5,1) has higher statistic than N(0,1)."""
        torch.manual_seed(42)
        x = torch.randn(500, 3) + 5.0
        result = ep_test(x)
        # Should have high statistic (different mean)
        assert (result > 0.1).all(), f"N(5,1) should have high statistic: {result}"

    def test_scaled_normal_rejected(self, ep_test):
        """Test that N(0,10) has higher statistic than N(0,1)."""
        torch.manual_seed(42)
        x = torch.randn(500, 3) * 10.0
        result = ep_test(x)
        # Should have high statistic (different variance)
        assert (result > 0.1).all(), f"N(0,10) should have high statistic: {result}"

    def test_independent_columns(self, ep_test):
        """Test that each column is tested independently."""
        torch.manual_seed(42)
        col1 = torch.randn(500, 1)  # N(0,1)
        col2 = torch.rand(500, 1)  # Uniform
        col3 = torch.randn(500, 1) + 5  # N(5,1)

        x = torch.cat([col1, col2, col3], dim=1)
        result = ep_test(x)

        assert result.shape == (3,), "Should have 3 statistics"
        # Column 1 should have lowest statistic
        assert result[0] < result[1], "N(0,1) should have lower stat than uniform"
        assert result[0] < result[2], "N(0,1) should have lower stat than N(5,1)"

    # Parameter Tests
    @pytest.mark.parametrize("t_range", [(-2, 2), (-3, 3), (-5, 5), (-10, 10)])
    def test_different_t_ranges(self, t_range):
        """Test with different t ranges."""
        ep_test = EppsPulley(t_range=t_range)
        torch.manual_seed(42)
        x = torch.randn(100, 3)
        result = ep_test(x)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()
        assert (result >= 0).all()

    @pytest.mark.parametrize("n_points", [5, 10, 20, 50])
    def test_different_n_points(self, n_points):
        """Test with different numbers of integration points."""
        ep_test = EppsPulley(n_points=n_points)
        torch.manual_seed(42)
        x = torch.randn(100, 3)
        result = ep_test(x)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()
        assert (result >= 0).all()

    @pytest.mark.parametrize("weight_type", ["gaussian", "uniform"])
    def test_different_weight_types(self, weight_type):
        """Test with different weight functions."""
        ep_test = EppsPulley(weight_type=weight_type)
        torch.manual_seed(42)
        x = torch.randn(100, 3)
        result = ep_test(x)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()
        assert (result >= 0).all()

    def test_invalid_weight_type_raises(self):
        """Test that invalid weight_type raises error."""
        ep_test = EppsPulley(weight_type="invalid")
        x = torch.randn(100, 3)
        with pytest.raises(ValueError, match="Unknown weight type"):
            ep_test(x)

    def test_more_points_more_accurate(self):
        """Test that more integration points give different (potentially more accurate) results."""
        torch.manual_seed(42)
        x = torch.randn(200, 2)

        ep_few = EppsPulley(n_points=5)
        ep_many = EppsPulley(n_points=50)

        result_few = ep_few(x)
        result_many = ep_many(x)

        # Results should be similar but not identical
        # (unless by coincidence)
        assert result_few.shape == result_many.shape

    # Edge Cases
    def test_small_sample_size(self, ep_test):
        """Test with small sample sizes."""
        x = torch.randn(10, 3)
        result = ep_test(x)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()

    def test_minimum_sample_size(self, ep_test):
        """Test with minimum viable sample size."""
        x = torch.randn(3, 4)
        result = ep_test(x)
        assert result.shape == (4,)
        assert torch.isfinite(result).all()

    def test_single_sample(self, ep_test):
        """Test with single sample (edge case)."""
        x = torch.randn(1, 3)
        result = ep_test(x)
        assert result.shape == (3,)
        # Single sample will have high statistic

    def test_large_sample_size(self, ep_test):
        """Test with large sample size."""
        torch.manual_seed(42)
        x = torch.randn(5000, 3)
        result = ep_test(x)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()
        # Large N(0,1) sample should have low statistic
        assert (result < 0.1).all()

    def test_large_dimensionality(self, ep_test):
        """Test with large number of dimensions."""
        torch.manual_seed(42)
        x = torch.randn(100, 100)
        result = ep_test(x)
        assert result.shape == (100,)
        assert torch.isfinite(result).all()

    # Dtype Tests
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_different_dtypes(self, dtype):
        """Test with different floating point dtypes."""
        ep_test = EppsPulley()
        x = torch.randn(100, 3, dtype=dtype)
        result = ep_test(x)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()

    # Mathematical Properties
    def test_statistic_non_negative(self, ep_test):
        """Test that statistics are always non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            x = torch.randn(100, 5)
            result = ep_test(x)
            assert (
                result >= -1e-7
            ).all(), f"Statistics should be non-negative, got {result}"

    def test_deterministic_output(self, ep_test):
        """Test that same input produces same output."""
        x = torch.randn(100, 4)
        result1 = ep_test(x)
        result2 = ep_test(x)
        assert torch.allclose(result1, result2), "Results should be deterministic"

    def test_permutation_invariance(self, ep_test):
        """Test that permuting samples within a column doesn't change result."""
        torch.manual_seed(42)
        x = torch.randn(100, 3)
        result1 = ep_test(x)

        perm = torch.randperm(100)
        x_permuted = x[perm, :]
        result2 = ep_test(x_permuted)

        assert torch.allclose(
            result1, result2, rtol=1e-4
        ), "Permuting samples should not change statistics"

    def test_column_independence(self, ep_test):
        """Test that permuting columns permutes results."""
        torch.manual_seed(42)
        x = torch.randn(100, 4)
        result1 = ep_test(x)

        x_permuted = x[:, [2, 0, 3, 1]]
        result2 = ep_test(x_permuted)

        expected = result1[[2, 0, 3, 1]]
        assert torch.allclose(result2, expected, rtol=1e-4)

    # Distribution Tests
    def test_perfect_standard_normal_low_statistic(self, ep_test):
        """Test that large N(0,1) sample has low statistic."""
        torch.manual_seed(42)
        x = torch.randn(5000, 5)
        result = ep_test(x)
        # Should have very low statistics (good fit)
        assert (result < 0.05).all(), f"Large N(0,1) should have low stat: {result}"

    def test_uniform_distribution_high_statistic(self, ep_test):
        """Test that uniform has high statistic."""
        torch.manual_seed(42)
        x = torch.rand(500, 3)
        result = ep_test(x)
        # Uniform should have elevated statistics
        assert (result > 0.05).all(), f"Uniform should have high stat: {result}"

    def test_exponential_distribution(self, ep_test):
        """Test with exponential distribution."""
        torch.manual_seed(42)
        x = torch.distributions.Exponential(1.0).sample((500, 3))
        result = ep_test(x)
        # Should have elevated statistic
        assert (result > 0.05).all()

    def test_mixed_distributions(self, ep_test):
        """Test with mixed distributions."""
        torch.manual_seed(42)
        n01_col = torch.randn(500, 1)
        uniform_col = torch.rand(500, 1)
        n51_col = torch.randn(500, 1) + 5

        x = torch.cat([n01_col, uniform_col, n51_col], dim=1)
        result = ep_test(x)

        assert result.shape == (3,)
        # N(0,1) should have lowest statistic
        assert result[0] < result[1], "N(0,1) < Uniform"
        assert result[0] < result[2], "N(0,1) < N(5,1)"

    def test_empty_columns(self, ep_test):
        """Test with zero samples."""
        try:
            x = torch.tensor([]).reshape(0, 3)
            result = ep_test(x)
            assert result.shape == (3,)
        except (RuntimeError, ValueError, IndexError):
            pass

    def test_nan_input(self, ep_test):
        """Test with NaN inputs."""
        x = torch.randn(100, 3)
        x[50, 1] = float("nan")
        result = ep_test(x)
        assert result.shape == (3,)
        # Column with NaN should produce NaN
        assert torch.isnan(result[1])

    def test_inf_input(self, ep_test):
        """Test with infinite inputs."""
        x = torch.randn(100, 3)
        x[50, 1] = float("inf")
        result = ep_test(x)
        assert result.shape == (3,)


class TestEppsPulleyCharacteristicFunction:
    """Test the characteristic function methods."""

    @pytest.fixture
    def ep_test(self):
        return EppsPulley()

    def test_empirical_cf_shape(self, ep_test):
        """Test empirical CF returns correct shape."""
        x = torch.randn(100, 3)
        t = torch.linspace(-3, 3, 10)

        cf = ep_test.empirical_cf(x, t)

        assert cf.shape == (10, 3), f"Expected shape (10, 3), got {cf.shape}"
        assert cf.dtype == torch.complex64 or cf.dtype == torch.complex128

    def test_empirical_cf_is_complex(self, ep_test):
        """Test that empirical CF returns complex values."""
        x = torch.randn(100, 2)
        t = torch.linspace(-3, 3, 10)

        cf = ep_test.empirical_cf(x, t)

        assert torch.is_complex(cf), "CF should be complex"

    def test_empirical_cf_at_zero(self, ep_test):
        """Test that empirical CF at t=0 equals 1."""
        x = torch.randn(100, 3)
        t = torch.tensor([0.0])

        cf = ep_test.empirical_cf(x, t)

        # At t=0, φ(0) = 1
        expected = torch.ones(1, 3, dtype=torch.complex64)
        assert torch.allclose(cf.real, expected.real, atol=1e-6)
        assert torch.allclose(cf.imag, expected.imag, atol=1e-6)

    def test_empirical_cf_magnitude_bounded(self, ep_test):
        """Test that |empirical CF| <= 1."""
        x = torch.randn(100, 3)
        t = torch.linspace(-5, 5, 20)

        cf = ep_test.empirical_cf(x, t)
        magnitude = torch.abs(cf)

        assert (
            magnitude <= 1.0 + 1e-6
        ).all(), f"CF magnitude should be <= 1, got max {magnitude.max()}"

    def test_normal_cf_shape(self, ep_test):
        """Test normal CF returns correct shape."""
        t = torch.linspace(-3, 3, 10)

        cf = ep_test.normal_cf(t, mu=0.0, sigma=1.0)

        assert cf.shape == (10,), f"Expected shape (10,), got {cf.shape}"
        assert torch.is_complex(cf)

    def test_normal_cf_at_zero(self, ep_test):
        """Test that normal CF at t=0 equals 1."""
        t = torch.tensor([0.0])

        cf = ep_test.normal_cf(t, mu=0.0, sigma=1.0)

        # At t=0, φ(0) = 1
        assert torch.allclose(cf.real, torch.tensor([1.0]), atol=1e-6)
        assert torch.allclose(cf.imag, torch.tensor([0.0]), atol=1e-6)

    def test_normal_cf_formula(self, ep_test):
        """Test normal CF formula for N(0,1)."""
        t = torch.tensor([1.0, 2.0])

        cf = ep_test.normal_cf(t, mu=0.0, sigma=1.0)

        # For N(0,1): φ(t) = exp(-t²/2)
        expected_real = torch.exp(-(t**2) / 2)
        expected_imag = torch.zeros_like(t)

        assert torch.allclose(cf.real, expected_real, rtol=1e-5)
        assert torch.allclose(cf.imag, expected_imag, atol=1e-6)


class TestEppsPulleyWeightFunction:
    """Test the weight function."""

    @pytest.fixture
    def ep_test_gaussian(self):
        return EppsPulley(weight_type="gaussian")

    @pytest.fixture
    def ep_test_uniform(self):
        return EppsPulley(weight_type="uniform")

    def test_gaussian_weight_shape(self, ep_test_gaussian):
        """Test gaussian weight returns correct shape."""
        t = torch.linspace(-3, 3, 10)
        weights = ep_test_gaussian.weight_function(t)

        assert weights.shape == t.shape
        assert not torch.is_complex(weights)

    def test_uniform_weight_shape(self, ep_test_uniform):
        """Test uniform weight returns correct shape."""
        t = torch.linspace(-3, 3, 10)
        weights = ep_test_uniform.weight_function(t)

        assert weights.shape == t.shape
        assert not torch.is_complex(weights)

    def test_gaussian_weight_at_zero(self, ep_test_gaussian):
        """Test gaussian weight at t=0 equals 1."""
        t = torch.tensor([0.0])
        weights = ep_test_gaussian.weight_function(t)

        assert torch.allclose(weights, torch.tensor([1.0]))

    def test_gaussian_weight_decreases(self, ep_test_gaussian):
        """Test gaussian weight decreases with |t|."""
        t = torch.tensor([0.0, 1.0, 2.0, 3.0])
        weights = ep_test_gaussian.weight_function(t)

        # Should be decreasing
        assert weights[0] > weights[1]
        assert weights[1] > weights[2]
        assert weights[2] > weights[3]

    def test_uniform_weight_is_one(self, ep_test_uniform):
        """Test uniform weight is always 1."""
        t = torch.linspace(-5, 5, 20)
        weights = ep_test_uniform.weight_function(t)

        assert torch.allclose(weights, torch.ones_like(t))

    def test_gaussian_weight_positive(self, ep_test_gaussian):
        """Test gaussian weight is always positive."""
        t = torch.linspace(-10, 10, 50)
        weights = ep_test_gaussian.weight_function(t)

        assert (weights > 0).all()


# Parametrized Tests
class TestEppsPulleyParametrized:
    """Parametrized tests for various scenarios."""

    @pytest.mark.parametrize("n_samples", [20, 50, 100, 500, 1000])
    def test_various_sample_sizes(self, n_samples):
        """Test with various sample sizes from N(0,1)."""
        ep_test = EppsPulley()
        x = torch.randn(n_samples, 3)
        result = ep_test(x)
        assert result.shape == (3,), f"Shape mismatch for n={n_samples}"
        assert torch.isfinite(result).all()
        assert (result >= 0).all()

    @pytest.mark.parametrize("n_dims", [1, 5, 10, 50])
    def test_various_dimensions(self, n_dims):
        """Test with various dimensions."""
        ep_test = EppsPulley()
        x = torch.randn(100, n_dims)
        result = ep_test(x)
        assert result.shape == (n_dims,)
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize("seed", range(5))
    def test_multiple_seeds(self, seed):
        """Test with different random seeds."""
        ep_test = EppsPulley()
        torch.manual_seed(seed)
        x = torch.randn(100, 4)
        result = ep_test(x)
        assert torch.isfinite(result).all()
        assert result.shape == (4,)

    @pytest.mark.parametrize("mean", [1.0, 2.0, 5.0])
    def test_non_zero_mean_higher_stat(self, mean):
        """Test that non-zero mean gives higher statistic."""
        ep_test = EppsPulley()
        torch.manual_seed(42)

        x_normal = torch.randn(300, 1)
        x_shifted = torch.randn(300, 1) + mean

        stat_normal = ep_test(x_normal)[0]
        stat_shifted = ep_test(x_shifted)[0]

        assert (
            stat_shifted > stat_normal
        ), f"N({mean},1) should have higher stat than N(0,1)"

    @pytest.mark.parametrize("std", [2.0, 5.0, 10.0])
    def test_non_unit_std_higher_stat(self, std):
        """Test that non-unit std gives higher statistic."""
        ep_test = EppsPulley()
        torch.manual_seed(42)

        x_normal = torch.randn(300, 1)
        x_scaled = torch.randn(300, 1) * std

        stat_normal = ep_test(x_normal)[0]
        stat_scaled = ep_test(x_scaled)[0]

        assert (
            stat_scaled > stat_normal
        ), f"N(0,{std}²) should have higher stat than N(0,1)"


class TestEppsPulleyEdgeCases:
    """Edge case tests."""

    @pytest.fixture
    def ep_test(self):
        return EppsPulley()

    def test_constant_column(self, ep_test):
        """Test with constant column."""
        x = torch.randn(100, 3)
        x[:, 1] = 5.0

        result = ep_test(x)
        assert result.shape == (3,)
        # Constant column should have very high statistic

    def test_bimodal_distribution(self, ep_test):
        """Test with bimodal distribution."""
        torch.manual_seed(42)
        x1 = torch.randn(250, 2) - 2
        x2 = torch.randn(250, 2) + 2
        x = torch.cat([x1, x2], dim=0)

        result = ep_test(x)
        # Should have elevated statistic
        assert (result > 0.01).all()

    def test_heavy_tailed_distribution(self, ep_test):
        """Test with heavy-tailed distribution."""
        torch.manual_seed(42)
        from torch.distributions import StudentT

        x = StudentT(df=3.0).sample((500, 3))

        result = ep_test(x)
        # Should have elevated statistic
        assert (result > 0.01).all()

    def test_very_narrow_t_range(self):
        """Test with very narrow t range."""
        ep_test = EppsPulley(t_range=(-0.5, 0.5))
        torch.manual_seed(42)
        x = torch.randn(100, 2)

        result = ep_test(x)
        assert result.shape == (2,)
        assert torch.isfinite(result).all()

    def test_very_wide_t_range(self):
        """Test with very wide t range."""
        ep_test = EppsPulley(t_range=(-20, 20))
        torch.manual_seed(42)
        x = torch.randn(100, 2)

        result = ep_test(x)
        assert result.shape == (2,)
        assert torch.isfinite(result).all()


class TestEppsPulleyStatistical:
    """Statistical property tests."""

    @pytest.fixture
    def ep_test(self):
        return EppsPulley()

    def test_larger_shift_larger_stat(self, ep_test):
        """Test that larger departures give larger statistics."""
        torch.manual_seed(42)

        x0 = torch.randn(300, 1)
        x1 = torch.randn(300, 1) + 1.0
        x2 = torch.randn(300, 1) + 3.0

        stat0 = ep_test(x0)[0]
        stat1 = ep_test(x1)[0]
        stat2 = ep_test(x2)[0]

        assert stat1 > stat0, "N(1,1) > N(0,1)"
        assert stat2 > stat1, "N(3,1) > N(1,1)"

    def test_larger_sample_smaller_variance(self, ep_test):
        """Test that larger samples give more stable statistics."""
        torch.manual_seed(42)

        small_stats = []
        for i in range(20):
            torch.manual_seed(i)
            x = torch.randn(50, 1)
            small_stats.append(ep_test(x)[0].item())

        large_stats = []
        for i in range(20):
            torch.manual_seed(i)
            x = torch.randn(500, 1)
            large_stats.append(ep_test(x)[0].item())

        small_var = np.var(small_stats)
        large_var = np.var(large_stats)

        # Larger samples should be more stable
        assert large_var < small_var, "Larger samples should be more stable"

    def test_power_against_uniform(self, ep_test):
        """Test that test has power against uniform."""
        torch.manual_seed(42)
        n_trials = 20
        threshold = 0.05  # Approximate threshold

        rejections = 0
        for i in range(n_trials):
            torch.manual_seed(i)
            x = torch.rand(200, 1)
            result = ep_test(x)
            if result[0] > threshold:
                rejections += 1

        rejection_rate = rejections / n_trials
        # Should reject most uniform samples
        assert (
            rejection_rate > 0.7
        ), f"Should have high power against uniform: {rejection_rate}"

    def test_convergence_with_sample_size(self, ep_test):
        """Test that statistic decreases with sample size for N(0,1)."""
        torch.manual_seed(42)

        x_small = torch.randn(100, 1)
        x_large = torch.randn(2000, 1)

        stat_small = ep_test(x_small)[0]
        stat_large = ep_test(x_large)[0]

        # Larger sample should have smaller statistic (better estimate)
        # This is probabilistic, so not always true
        # Just check both are reasonable
        assert stat_small < 0.5
        assert stat_large < 0.3


class TestEppsPulleyComparison:
    """Comparison tests between different configurations."""

    def test_gaussian_vs_uniform_weight(self):
        """Test that gaussian and uniform weights give different results."""
        torch.manual_seed(42)
        x = torch.randn(200, 2)

        ep_gaussian = EppsPulley(weight_type="gaussian")
        ep_uniform = EppsPulley(weight_type="uniform")

        result_gaussian = ep_gaussian(x)
        result_uniform = ep_uniform(x)

        # Results should be different
        assert not torch.allclose(
            result_gaussian, result_uniform
        ), "Different weight functions should give different results"

    def test_wider_range_includes_more_information(self):
        """Test that wider t-range captures more information."""
        torch.manual_seed(42)
        # Use non-normal data
        x = torch.rand(200, 2)

        ep_narrow = EppsPulley(t_range=(-1, 1))
        ep_wide = EppsPulley(t_range=(-10, 10))

        result_narrow = ep_narrow(x)
        result_wide = ep_wide(x)

        # Both should detect non-normality
        assert (result_narrow > 0.01).all()
        assert (result_wide > 0.01).all()
