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
from lejepa.univariate import ExtendedJarqueBera


class TestExtendedJarqueBera:
    """Test suite for Extended Jarque-Bera test for standard normal N(0,1)."""

    @pytest.fixture
    def ejb_test(self):
        """Create an ExtendedJarqueBera test instance."""
        return ExtendedJarqueBera()

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
    def test_forward_returns_correct_shape(self, ejb_test, standard_normal_samples_2d):
        """Test that forward returns shape (dim,) for 2D input (n, dim)."""
        n, dim = standard_normal_samples_2d.shape
        result = ejb_test(standard_normal_samples_2d)
        assert result.shape == (dim,), f"Expected shape ({dim},), got {result.shape}"
        assert isinstance(result, torch.Tensor), "Result should be a Tensor"

    def test_forward_single_column(self, ejb_test):
        """Test with single column input (n, 1)."""
        torch.manual_seed(42)
        x = torch.randn(1000, 1)
        result = ejb_test(x)
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
        assert torch.isfinite(result).all(), "Result should be finite"

    def test_standard_normal_samples_low_statistic(
        self, ejb_test, standard_normal_samples_2d
    ):
        """Test that N(0,1) samples produce low test statistics."""
        result = ejb_test(standard_normal_samples_2d)
        # For large N(0,1) samples, statistic should be moderate (follows χ²(4))
        # Mean of χ²(4) is 4, so values should typically be < 20
        assert (result < 30).all(), f"Statistics too high for N(0,1) data: {result}"

    def test_non_standard_normal_samples_high_statistic(
        self, ejb_test, non_standard_normal_samples_2d
    ):
        """Test that non-N(0,1) samples produce high test statistics."""
        result = ejb_test(non_standard_normal_samples_2d)
        # Uniform samples should have high statistics
        assert (result > 10).all(), f"Statistics too low for non-N(0,1) data: {result}"

    def test_shifted_normal_rejected(self, ejb_test):
        """Test that N(5,1) has high statistic (mean != 0)."""
        torch.manual_seed(42)
        x = torch.randn(500, 3) + 5.0  # N(5,1)
        result = ejb_test(x)
        # Should have very high statistic due to mean term
        assert (result > 100).all(), f"N(5,1) should have very high statistic: {result}"

    def test_scaled_normal_rejected(self, ejb_test):
        """Test that N(0,4) has high statistic (var != 1)."""
        torch.manual_seed(42)
        x = torch.randn(500, 3) * 2.0  # N(0,4)
        result = ejb_test(x)
        # Should have high statistic due to variance term
        assert (result > 10).all(), f"N(0,4) should have high statistic: {result}"

    def test_independent_columns(self, ejb_test):
        """Test that each column is tested independently."""
        torch.manual_seed(42)
        col1 = torch.randn(500, 1)  # N(0,1)
        col2 = torch.rand(500, 1)  # Uniform
        col3 = torch.randn(500, 1) + 5  # N(5,1)

        x = torch.cat([col1, col2, col3], dim=1)
        result = ejb_test(x)

        assert result.shape == (3,), "Should have 3 statistics"
        # Column 1 should have lowest statistic
        assert result[0] < result[1], "N(0,1) should have lower stat than uniform"
        assert result[0] < result[2], "N(0,1) should have lower stat than N(5,1)"

    # Edge Cases
    def test_small_sample_size(self, ejb_test):
        """Test with small sample sizes."""
        x = torch.randn(10, 3)
        result = ejb_test(x)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()

    def test_minimum_sample_size(self, ejb_test):
        """Test with minimum viable sample size (n > 1 for unbiased variance)."""
        x = torch.randn(5, 4)
        result = ejb_test(x)
        assert result.shape == (4,)
        assert torch.isfinite(result).all()

    def test_very_small_sample(self, ejb_test):
        """Test with n=2 (minimum for unbiased variance)."""
        x = torch.randn(2, 3)
        result = ejb_test(x)
        assert result.shape == (3,)
        # May have very high statistics due to small n

    def test_large_sample_size(self, ejb_test):
        """Test with large sample size."""
        torch.manual_seed(42)
        x = torch.randn(10000, 3)
        result = ejb_test(x)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()
        # Large N(0,1) sample should have reasonable statistics
        assert (result < 20).all()

    def test_large_dimensionality(self, ejb_test):
        """Test with large number of dimensions."""
        torch.manual_seed(42)
        x = torch.randn(100, 100)
        result = ejb_test(x)
        assert result.shape == (100,)
        assert torch.isfinite(result).all()

    # Dtype Tests
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_different_dtypes(self, ejb_test, dtype):
        """Test with different floating point dtypes."""
        x = torch.randn(100, 3, dtype=dtype)
        result = ejb_test(x)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()

    # Mathematical Properties
    def test_statistic_non_negative(self, ejb_test):
        """Test that statistics are always non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            x = torch.randn(100, 5)
            result = ejb_test(x)
            assert (
                result >= 0
            ).all(), f"Statistics should be non-negative, got {result}"

    def test_deterministic_output(self, ejb_test):
        """Test that same input produces same output."""
        x = torch.randn(100, 4)
        result1 = ejb_test(x)
        result2 = ejb_test(x)
        assert torch.allclose(result1, result2), "Results should be deterministic"

    def test_permutation_invariance(self, ejb_test):
        """Test that permuting samples within a column doesn't change result."""
        torch.manual_seed(42)
        x = torch.randn(100, 3)
        result1 = ejb_test(x)

        perm = torch.randperm(100)
        x_permuted = x[perm, :]
        result2 = ejb_test(x_permuted)

        assert torch.allclose(
            result1, result2, rtol=1e-4
        ), "Permuting samples should not change statistics"

    def test_column_independence(self, ejb_test):
        """Test that permuting columns permutes results."""
        torch.manual_seed(42)
        x = torch.randn(100, 4)
        result1 = ejb_test(x)

        x_permuted = x[:, [2, 0, 3, 1]]
        result2 = ejb_test(x_permuted)

        expected = result1[[2, 0, 3, 1]]
        assert torch.allclose(result2, expected, rtol=1e-4)

    # Component Tests
    def test_mean_component_dominates_for_shifted_normal(self, ejb_test):
        """Test that mean component dominates for N(μ,1) with μ != 0."""
        torch.manual_seed(42)
        n = 500

        # N(0,1) - baseline
        x_normal = torch.randn(n, 1)
        stat_normal = ejb_test(x_normal)[0]

        # N(3,1) - shifted
        x_shifted = torch.randn(n, 1) + 3.0
        stat_shifted = ejb_test(x_shifted)[0]

        # Shifted should have much higher statistic
        assert (
            stat_shifted > stat_normal * 10
        ), f"Shifted normal should have much higher stat due to mean component"

    def test_variance_component_dominates_for_scaled_normal(self, ejb_test):
        """Test that variance component contributes for N(0,σ²) with σ != 1."""
        torch.manual_seed(42)
        n = 500

        # N(0,1) - baseline
        x_normal = torch.randn(n, 1)
        stat_normal = ejb_test(x_normal)[0]

        # N(0,9) - scaled
        x_scaled = torch.randn(n, 1) * 3.0
        stat_scaled = ejb_test(x_scaled)[0]

        # Scaled should have higher statistic
        assert (
            stat_scaled > stat_normal
        ), f"Scaled normal should have higher stat due to variance component"

    def test_skewness_component_for_skewed_distribution(self, ejb_test):
        """Test that skewness component contributes for skewed distributions."""
        torch.manual_seed(42)
        n = 500

        # N(0,1) - baseline
        x_normal = torch.randn(n, 1)
        stat_normal = ejb_test(x_normal)[0]

        # Exponential - highly skewed
        x_skewed = torch.distributions.Exponential(1.0).sample((n, 1))
        # Need to standardize to isolate skewness effect
        x_skewed = (x_skewed - x_skewed.mean()) / x_skewed.std()
        stat_skewed = ejb_test(x_skewed)[0]

        # Skewed should have higher statistic
        assert stat_skewed > stat_normal

    # Distribution Tests
    def test_perfect_standard_normal_reasonable_statistic(self, ejb_test):
        """Test that large N(0,1) sample has reasonable statistic."""
        torch.manual_seed(42)
        x = torch.randn(10000, 5)
        result = ejb_test(x)

        # For χ²(4), mean=4, variance=8, so 95% should be < ~12
        # With large sample, most should pass
        assert (result < 15).sum() >= 3, f"Most N(0,1) should have low stats: {result}"

    def test_uniform_distribution_high_statistic(self, ejb_test):
        """Test that uniform has high statistic."""
        torch.manual_seed(42)
        x = torch.rand(500, 3)
        result = ejb_test(x)
        # Uniform has mean=0.5, var=1/12, different from N(0,1)
        assert (result > 50).all()

    def test_exponential_distribution(self, ejb_test):
        """Test with exponential distribution."""
        torch.manual_seed(42)
        x = torch.distributions.Exponential(1.0).sample((500, 3))
        result = ejb_test(x)
        # Exponential has mean=1, var=1, high skewness and kurtosis
        assert (result > 10).all()

    def test_mixed_distributions(self, ejb_test):
        """Test with mixed distributions."""
        torch.manual_seed(42)
        n01_col = torch.randn(500, 1)
        uniform_col = torch.rand(500, 1)
        n51_col = torch.randn(500, 1) + 5

        x = torch.cat([n01_col, uniform_col, n51_col], dim=1)
        result = ejb_test(x)

        assert result.shape == (3,)
        # N(0,1) should have lowest statistic
        assert result[0] < result[1], "N(0,1) < Uniform"
        assert result[0] < result[2], "N(0,1) < N(5,1)"

    def test_empty_columns(self, ejb_test):
        """Test with zero samples."""
        try:
            x = torch.tensor([]).reshape(0, 3)
            result = ejb_test(x)
            assert result.shape == (3,)
        except (RuntimeError, ValueError, IndexError, ZeroDivisionError):
            pass

    def test_nan_input(self, ejb_test):
        """Test with NaN inputs."""
        x = torch.randn(100, 3)
        x[50, 1] = float("nan")
        result = ejb_test(x)
        assert result.shape == (3,)
        # Column with NaN should produce NaN
        assert torch.isnan(result[1])

    def test_inf_input(self, ejb_test):
        """Test with infinite inputs."""
        x = torch.randn(100, 3)
        x[50, 1] = float("inf")
        result = ejb_test(x)
        assert result.shape == (3,)


class TestExtendedJarqueBeraComponents:
    """Test individual components of the test statistic."""

    @pytest.fixture
    def ejb_test(self):
        return ExtendedJarqueBera()

    def test_mean_zero_for_n01(self, ejb_test):
        """Test that N(0,1) has sample mean close to 0."""
        torch.manual_seed(42)
        x = torch.randn(10000, 1)

        # Compute mean component manually
        n = x.shape[0]
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=True)
        stat_mean = (mean**2) / (var / n)

        # For large N(0,1), mean component should be small (~ χ²(1))
        assert (
            stat_mean[0] < 10
        ), f"Mean component should be small for N(0,1): {stat_mean}"

    def test_variance_one_for_n01(self, ejb_test):
        """Test that N(0,1) has sample variance close to 1."""
        torch.manual_seed(42)
        x = torch.randn(10000, 1)

        # Compute variance component manually
        n = x.shape[0]
        var = x.var(dim=0, unbiased=True)
        stat_var = ((var - 1) ** 2) / (2 / (n - 1))

        # For large N(0,1), variance component should be small
        assert (
            stat_var[0] < 10
        ), f"Variance component should be small for N(0,1): {stat_var}"

    def test_skewness_zero_for_n01(self, ejb_test):
        """Test that N(0,1) has sample skewness close to 0."""
        torch.manual_seed(42)
        x = torch.randn(10000, 1)

        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=True)
        std = var.sqrt()
        skewness = ((x - mean) / std).pow(3).mean(dim=0)

        # For large N(0,1), skewness should be close to 0
        assert (
            abs(skewness[0]) < 0.1
        ), f"Skewness should be close to 0 for N(0,1): {skewness}"

    def test_kurtosis_three_for_n01(self, ejb_test):
        """Test that N(0,1) has sample kurtosis close to 3."""
        torch.manual_seed(42)
        x = torch.randn(10000, 1)

        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=True)
        std = var.sqrt()
        kurtosis = ((x - mean) / std).pow(4).mean(dim=0)

        # For large N(0,1), kurtosis should be close to 3
        assert (
            abs(kurtosis[0] - 3) < 0.2
        ), f"Kurtosis should be close to 3 for N(0,1): {kurtosis}"

    def test_mean_component_for_shifted_data(self, ejb_test):
        """Test mean component increases for shifted data."""
        torch.manual_seed(42)
        n = 500

        x_normal = torch.randn(n, 1)
        x_shifted = torch.randn(n, 1) + 2.0

        # Mean component for normal
        mean_n = x_normal.mean(dim=0)
        var_n = x_normal.var(dim=0, unbiased=True)
        stat_mean_n = (mean_n**2) / (var_n / n)

        # Mean component for shifted
        mean_s = x_shifted.mean(dim=0)
        var_s = x_shifted.var(dim=0, unbiased=True)
        stat_mean_s = (mean_s**2) / (var_s / n)

        assert (
            stat_mean_s > stat_mean_n * 100
        ), f"Mean component should be much larger for shifted data"

    def test_variance_component_for_scaled_data(self, ejb_test):
        """Test variance component increases for scaled data."""
        torch.manual_seed(42)
        n = 500

        x_normal = torch.randn(n, 1)
        x_scaled = torch.randn(n, 1) * 3.0

        # Variance component for normal
        var_n = x_normal.var(dim=0, unbiased=True)
        stat_var_n = ((var_n - 1) ** 2) / (2 / (n - 1))

        # Variance component for scaled
        var_s = x_scaled.var(dim=0, unbiased=True)
        stat_var_s = ((var_s - 1) ** 2) / (2 / (n - 1))

        assert (
            stat_var_s > stat_var_n * 10
        ), f"Variance component should be much larger for scaled data"

    def test_total_is_sum_of_components(self, ejb_test):
        """Test that total statistic equals sum of components."""
        torch.manual_seed(42)
        x = torch.randn(100, 2)

        n = x.shape[0]
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=True)
        std = var.sqrt().clamp(min=1e-8)
        skewness = ((x - mean) / std).pow(3).mean(dim=0)
        kurtosis = ((x - mean) / std).pow(4).mean(dim=0)

        stat_mean = (mean**2) / (var / n)
        stat_var = ((var - 1) ** 2) / (2 / (n - 1))
        stat_skew_kurt = n / 6 * (skewness**2 + 0.25 * (kurtosis - 3) ** 2)

        expected_total = stat_mean + stat_var + stat_skew_kurt
        actual_total = ejb_test(x)

        assert torch.allclose(
            actual_total, expected_total, rtol=1e-5
        ), f"Total should equal sum of components"


# Parametrized Tests
class TestExtendedJarqueBeraParametrized:
    """Parametrized tests for various scenarios."""

    @pytest.mark.parametrize("n_samples", [10, 50, 100, 500, 1000])
    def test_various_sample_sizes(self, n_samples):
        """Test with various sample sizes from N(0,1)."""
        ejb_test = ExtendedJarqueBera()
        x = torch.randn(n_samples, 3)
        result = ejb_test(x)
        assert result.shape == (3,), f"Shape mismatch for n={n_samples}"
        assert torch.isfinite(result).all()
        assert (result >= 0).all()

    @pytest.mark.parametrize("n_dims", [1, 5, 10, 50])
    def test_various_dimensions(self, n_dims):
        """Test with various dimensions."""
        ejb_test = ExtendedJarqueBera()
        x = torch.randn(100, n_dims)
        result = ejb_test(x)
        assert result.shape == (n_dims,)
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize("seed", range(5))
    def test_multiple_seeds(self, seed):
        """Test with different random seeds."""
        ejb_test = ExtendedJarqueBera()
        torch.manual_seed(seed)
        x = torch.randn(100, 4)
        result = ejb_test(x)
        assert torch.isfinite(result).all()
        assert result.shape == (4,)

    @pytest.mark.parametrize("mean", [0.5, 1.0, 2.0, 5.0])
    def test_non_zero_mean_higher_stat(self, mean):
        """Test that non-zero mean gives higher statistic."""
        ejb_test = ExtendedJarqueBera()
        torch.manual_seed(42)
        n = 500

        x_normal = torch.randn(n, 1)
        x_shifted = torch.randn(n, 1) + mean

        stat_normal = ejb_test(x_normal)[0]
        stat_shifted = ejb_test(x_shifted)[0]

        assert (
            stat_shifted > stat_normal
        ), f"N({mean},1) should have higher stat than N(0,1)"

    @pytest.mark.parametrize("std", [0.5, 2.0, 5.0])
    def test_non_unit_variance_higher_stat(self, std):
        """Test that non-unit variance gives higher statistic."""
        ejb_test = ExtendedJarqueBera()
        torch.manual_seed(42)
        n = 500

        x_normal = torch.randn(n, 1)
        x_scaled = torch.randn(n, 1) * std

        stat_normal = ejb_test(x_normal)[0]
        stat_scaled = ejb_test(x_scaled)[0]

        assert (
            stat_scaled > stat_normal
        ), f"N(0,{std}²) should have higher stat than N(0,1)"


class TestExtendedJarqueBeraEdgeCases:
    """Edge case tests."""

    @pytest.fixture
    def ejb_test(self):
        return ExtendedJarqueBera()

    def test_constant_column(self, ejb_test):
        """Test with constant column (zero variance)."""
        x = torch.randn(100, 3)
        x[:, 1] = 5.0

        result = ejb_test(x)
        assert result.shape == (3,)
        # Constant column should have very high or infinite statistic
        # due to variance component and division by std

    def test_near_constant_column(self, ejb_test):
        """Test with near-constant column."""
        x = torch.randn(100, 3)
        x[:, 1] = 5.0 + torch.randn(100) * 1e-6

        result = ejb_test(x)
        assert result.shape == (3,)
        # Near-constant should have very high statistic

    def test_bimodal_distribution(self, ejb_test):
        """Test with bimodal distribution."""
        torch.manual_seed(42)
        x1 = torch.randn(250, 2) - 2
        x2 = torch.randn(250, 2) + 2
        x = torch.cat([x1, x2], dim=0)

        result = ejb_test(x)
        # Bimodal might have near-zero mean and near-unit variance
        # but different higher moments
        assert (result > 0).all()

    def test_heavy_tailed_distribution(self, ejb_test):
        """Test with heavy-tailed distribution (t-distribution)."""
        torch.manual_seed(42)
        from torch.distributions import StudentT

        x = StudentT(df=3.0).sample((500, 3))

        result = ejb_test(x)
        # t(3) has infinite kurtosis, should have high statistic
        assert (result > 5).all()

    def test_laplace_distribution(self, ejb_test):
        """Test with Laplace distribution (zero skewness, high kurtosis)."""
        torch.manual_seed(42)
        from torch.distributions import Laplace

        x = Laplace(0, 1).sample((500, 3))

        result = ejb_test(x)
        # Laplace has kurtosis = 6 (excess kurtosis = 3)
        # Should have elevated statistic
        assert (result > 5).all()


class TestExtendedJarqueBeraStatistical:
    """Statistical property tests."""

    @pytest.fixture
    def ejb_test(self):
        return ExtendedJarqueBera()

    def test_chi_squared_distribution_under_null(self, ejb_test):
        """Test that statistic roughly follows χ²(4) under null hypothesis."""
        torch.manual_seed(42)
        n_trials = 100
        statistics = []

        for i in range(n_trials):
            torch.manual_seed(i)
            x = torch.randn(200, 1)  # N(0,1)
            stat = ejb_test(x)[0].item()
            statistics.append(stat)

        statistics = np.array(statistics)

        # Mean of χ²(4) is 4
        mean_stat = np.mean(statistics)
        assert 2 < mean_stat < 8, f"Mean statistic should be around 4: {mean_stat}"

        # Median of χ²(4) is around 3.36
        median_stat = np.median(statistics)
        assert (
            1 < median_stat < 6
        ), f"Median statistic should be around 3.36: {median_stat}"

    def test_type_i_error_rate(self, ejb_test):
        """Test approximate Type I error rate for N(0,1)."""
        torch.manual_seed(42)
        n_trials = 100
        alpha = 0.05
        # χ²(4) critical value at 5% is approximately 9.488
        critical_value = 9.488

        rejections = 0
        for i in range(n_trials):
            torch.manual_seed(i)
            x = torch.randn(200, 1)
            result = ejb_test(x)
            if result[0] > critical_value:
                rejections += 1

        rejection_rate = rejections / n_trials
        # Should be close to alpha
        assert (
            0.01 < rejection_rate < 0.15
        ), f"Type I error rate {rejection_rate} not close to {alpha}"

    def test_power_against_shifted_normal(self, ejb_test):
        """Test that test has high power against N(1,1)."""
        torch.manual_seed(42)
        n_trials = 20
        critical_value = 9.488

        rejections = 0
        for i in range(n_trials):
            torch.manual_seed(i)
            x = torch.randn(200, 1) + 1.0  # N(1,1)
            result = ejb_test(x)
            if result[0] > critical_value:
                rejections += 1

        rejection_rate = rejections / n_trials
        # Should reject most of the time
        assert (
            rejection_rate > 0.9
        ), f"Should have very high power against N(1,1): {rejection_rate}"

    def test_power_against_uniform(self, ejb_test):
        """Test that test has high power against uniform."""
        torch.manual_seed(42)
        n_trials = 20
        critical_value = 9.488

        rejections = 0
        for i in range(n_trials):
            torch.manual_seed(i)
            x = torch.rand(200, 1)
            result = ejb_test(x)
            if result[0] > critical_value:
                rejections += 1

        rejection_rate = rejections / n_trials
        # Should reject all
        assert (
            rejection_rate > 0.95
        ), f"Should have very high power against uniform: {rejection_rate}"

    def test_larger_departure_larger_stat(self, ejb_test):
        """Test that larger departures give larger statistics."""
        torch.manual_seed(42)
        n = 500

        x0 = torch.randn(n, 1)  # N(0,1)
        x1 = torch.randn(n, 1) + 1.0  # N(1,1)
        x2 = torch.randn(n, 1) + 3.0  # N(3,1)

        stat0 = ejb_test(x0)[0]
        stat1 = ejb_test(x1)[0]
        stat2 = ejb_test(x2)[0]

        assert stat1 > stat0, "N(1,1) > N(0,1)"
        assert stat2 > stat1, "N(3,1) > N(1,1)"

    def test_variance_departure_ordering(self, ejb_test):
        """Test that larger variance departures give larger statistics."""
        torch.manual_seed(42)
        n = 500

        x0 = torch.randn(n, 1)  # N(0,1)
        x1 = torch.randn(n, 1) * 1.5  # N(0,2.25)
        x2 = torch.randn(n, 1) * 3.0  # N(0,9)

        stat0 = ejb_test(x0)[0]
        stat1 = ejb_test(x1)[0]
        stat2 = ejb_test(x2)[0]

        assert stat1 > stat0, "N(0,2.25) > N(0,1)"
        assert stat2 > stat1, "N(0,9) > N(0,2.25)"

    def test_sample_size_effect_on_power(self, ejb_test):
        """Test that larger samples have more power to detect departures."""
        torch.manual_seed(42)

        # Small sample
        x_small = torch.randn(50, 1) + 0.5
        stat_small = ejb_test(x_small)[0]

        # Large sample with same departure
        torch.manual_seed(42)
        x_large = torch.randn(500, 1) + 0.5
        stat_large = ejb_test(x_large)[0]

        # Larger sample should detect the departure better (higher statistic)
        assert (
            stat_large > stat_small
        ), "Larger sample should have higher statistic for same departure"


class TestExtendedJarqueBeraComparison:
    """Comparison with other tests."""

    def test_all_four_moments_tested(self):
        """Test that all four moments contribute to the statistic."""
        ejb_test = ExtendedJarqueBera()
        torch.manual_seed(42)
        n = 500

        # Create data that violates each moment
        x_mean = torch.randn(n, 1) + 2.0  # Wrong mean
        x_var = torch.randn(n, 1) * 2.0  # Wrong variance
        x_skew = torch.distributions.Exponential(1.0).sample((n, 1))  # Wrong skewness
        # All should have high statistics

        stat_mean = ejb_test(x_mean)[0]
        stat_var = ejb_test(x_var)[0]
        stat_skew = ejb_test(x_skew)[0]

        # All should be elevated
        assert stat_mean > 10, "Wrong mean should be detected"
        assert stat_var > 5, "Wrong variance should be detected"
        assert stat_skew > 5, "Wrong skewness should be detected"
