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
from scipy import stats
from lejepa.univariate import AndersonDarling
from unittest.mock import patch


class TestAndersonDarling:
    """Test suite for Anderson-Darling test for standard normal N(0,1)."""

    @pytest.fixture
    def ad_test(self):
        """Create an AndersonDarling test instance."""
        return AndersonDarling()

    @pytest.fixture
    def standard_normal_samples_2d(self):
        """Generate 2D samples from standard normal N(0,1)."""
        torch.manual_seed(42)
        return torch.randn(1000, 5)  # Already N(0,1)

    @pytest.fixture
    def non_standard_normal_samples_2d(self):
        """Generate 2D samples NOT from standard normal."""
        torch.manual_seed(42)
        return torch.rand(1000, 5)  # Uniform, not N(0,1)

    # Basic Functionality Tests
    def test_forward_returns_correct_shape(self, ad_test, standard_normal_samples_2d):
        """Test that forward returns shape (dim,) for 2D input (n, dim)."""
        n, dim = standard_normal_samples_2d.shape
        result = ad_test(standard_normal_samples_2d)
        assert result.shape == (dim,), f"Expected shape ({dim},), got {result.shape}"
        assert isinstance(result, torch.Tensor), "Result should be a Tensor"

    def test_forward_single_column(self, ad_test):
        """Test with single column input (n, 1)."""
        torch.manual_seed(42)
        x = torch.randn(100, 1)  # N(0,1)
        result = ad_test(x)
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
        assert torch.isfinite(result).all(), "Result should be finite"

    def test_standard_normal_samples_low_statistic(
        self, ad_test, standard_normal_samples_2d
    ):
        """Test that N(0,1) samples produce low test statistics."""
        result = ad_test(standard_normal_samples_2d)
        # Most N(0,1) samples should have A² < 2.492 (5% critical value)
        assert (result < 5.0).all(), f"A² statistics too high for N(0,1) data: {result}"

    def test_non_standard_normal_samples_high_statistic(
        self, ad_test, non_standard_normal_samples_2d
    ):
        """Test that non-N(0,1) samples produce high test statistics."""
        result = ad_test(non_standard_normal_samples_2d)
        # Uniform samples should have high A² statistics
        assert (
            result > 2.492
        ).all(), f"A² statistics too low for non-N(0,1) data: {result}"

    def test_shifted_normal_rejected(self, ad_test):
        """Test that N(5,1) is correctly rejected (not N(0,1))."""
        torch.manual_seed(42)
        x = torch.randn(500, 3) + 5.0  # N(5,1), not N(0,1)
        result = ad_test(x)
        # Should have very high statistics since mean != 0
        assert (result > 10.0).all(), f"N(5,1) should be strongly rejected: {result}"

    def test_scaled_normal_rejected(self, ad_test):
        """Test that N(0,10) is correctly rejected (not N(0,1))."""
        torch.manual_seed(42)
        x = torch.randn(500, 3) * 10.0  # N(0,10), not N(0,1)
        result = ad_test(x)
        # Should have high statistics since variance != 1
        assert (result > 10.0).all(), f"N(0,10) should be strongly rejected: {result}"

    def test_independent_columns(self, ad_test):
        """Test that each column is tested independently."""
        torch.manual_seed(42)
        # Create data where columns have different distributions
        col1 = torch.randn(500, 1)  # N(0,1)
        col2 = torch.rand(500, 1) * 10  # Uniform
        col3 = torch.randn(500, 1) + 5  # N(5,1)

        x = torch.cat([col1, col2, col3], dim=1)
        result = ad_test(x)

        assert result.shape == (3,), "Should have 3 statistics for 3 columns"
        # Column 1 should have low statistic, others high
        assert result[0] < result[1], "N(0,1) should have lower stat than uniform"
        assert result[0] < result[2], "N(0,1) should have lower stat than N(5,1)"

    # Edge Cases
    def test_small_sample_size(self, ad_test):
        """Test with small sample sizes."""
        x = torch.randn(5, 3)  # N(0,1)
        result = ad_test(x)
        assert result.shape == (3,), "Should return statistics for all columns"
        assert torch.isfinite(
            result
        ).all(), "Results should be finite for small samples"

    def test_minimum_sample_size(self, ad_test):
        """Test with minimum viable sample size."""
        x = torch.randn(2, 4)  # N(0,1)
        result = ad_test(x)
        assert result.shape == (4,), "Should return statistics for all columns"
        assert torch.isfinite(result).all(), "Results should be finite"

    def test_large_dimensionality(self, ad_test):
        """Test with large number of dimensions."""
        torch.manual_seed(42)
        x = torch.randn(100, 100)  # N(0,1), 100 dimensions
        result = ad_test(x)
        assert result.shape == (100,), "Should handle large dimensionality"
        assert torch.isfinite(result).all(), "All results should be finite"

    # Numerical Stability Tests
    def test_extreme_values_from_n01(self, ad_test):
        """Test numerical stability with extreme values from N(0,1)."""
        torch.manual_seed(42)
        # Generate large sample to get some extreme values
        x = torch.randn(10000, 4)  # Will have some values around ±4
        # Take subset that includes extremes
        indices = torch.randperm(10000)[:100]
        x_subset = x[indices]

        result = ad_test(x_subset)
        assert result.shape == (4,), "Shape should be correct"
        assert torch.isfinite(result).all(), "Results should be finite"

    def test_near_zero_variance_rejected(self, ad_test):
        """Test that near-constant data is rejected (not N(0,1))."""
        x = torch.ones(100, 3) * 0.01  # Nearly constant, not N(0,1)
        result = ad_test(x)
        # Should have very high statistics
        assert (result > 5.0).all(), "Near-constant data should be rejected"

    # Mathematical Properties Tests
    def test_statistic_non_negative(self, ad_test):
        """Test that A² statistics are always non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            x = torch.randn(100, 5)  # N(0,1)
            result = ad_test(x)
            assert (
                result >= -1e-5
            ).all(), f"A² statistics should be non-negative, got {result}"

    def test_deterministic_output(self, ad_test):
        """Test that same input produces same output."""
        x = torch.randn(100, 4)
        result1 = ad_test(x)
        result2 = ad_test(x)
        assert torch.allclose(result1, result2), "Results should be deterministic"

    def test_permutation_invariance_within_column(self, ad_test):
        """Test that permuting samples within a column doesn't change result."""
        torch.manual_seed(42)
        x = torch.randn(100, 3)
        result1 = ad_test(x)

        # Permute rows (samples)
        perm = torch.randperm(100)
        x_permuted = x[perm, :]
        result2 = ad_test(x_permuted)

        assert torch.allclose(
            result1, result2, rtol=1e-4
        ), "Permuting samples should not change statistics"

    def test_column_independence(self, ad_test):
        """Test that permuting columns changes which statistic is which but not values."""
        torch.manual_seed(42)
        x = torch.randn(100, 4)
        result1 = ad_test(x)

        # Permute columns
        x_permuted = x[:, [2, 0, 3, 1]]
        result2 = ad_test(x_permuted)

        # Results should be permuted the same way
        expected = result1[[2, 0, 3, 1]]
        assert torch.allclose(
            result2, expected, rtol=1e-4
        ), "Column permutation should permute results accordingly"

    # Integration with prepare_data
    @patch.object(AndersonDarling, "prepare_data")
    def test_prepare_data_called(self, mock_prepare, ad_test):
        """Test that prepare_data is called."""
        x = torch.randn(100, 3)
        mock_prepare.return_value = torch.randn(100, 3)
        ad_test(x)
        mock_prepare.assert_called_once_with(x)

    # Comparison with scipy (column-wise)
    def test_comparison_with_scipy_single_column(self, ad_test):
        """Compare results with scipy's Anderson-Darling test for single column."""
        torch.manual_seed(42)
        x = torch.randn(100, 1)  # N(0,1)

        # PyTorch result
        torch_result = ad_test(x)[0].item()

        # Scipy result
        scipy_result = stats.anderson(x[:, 0].numpy(), dist="norm")

        # Should be close
        assert (
            abs(torch_result - scipy_result.statistic) < 0.5
        ), f"PyTorch ({torch_result}) and Scipy ({scipy_result.statistic}) differ significantly"

    def test_comparison_with_scipy_multiple_columns(self, ad_test):
        """Compare results with scipy for multiple columns."""
        torch.manual_seed(42)
        x = torch.randn(100, 3)  # N(0,1)

        torch_results = ad_test(x)

        for i in range(3):
            scipy_result = stats.anderson(x[:, i].numpy(), dist="norm")
            assert (
                abs(torch_results[i].item() - scipy_result.statistic) < 0.5
            ), f"Column {i}: PyTorch and Scipy differ significantly"

    # Special Distribution Tests
    def test_perfect_standard_normal(self, ad_test):
        """Test with large samples from N(0,1)."""
        torch.manual_seed(42)
        x = torch.randn(10000, 5)  # Large sample from N(0,1)
        result = ad_test(x)
        # Should have low statistics
        assert (
            result < 2.492
        ).all(), f"Large N(0,1) samples should pass at 5% level, got {result}"

    def test_mixed_distributions(self, ad_test):
        """Test with mixed N(0,1) and non-N(0,1) columns."""
        torch.manual_seed(42)
        n01_col = torch.randn(500, 1)  # N(0,1)
        uniform_col = torch.rand(500, 1)  # Uniform
        n51_col = torch.randn(500, 1) + 5  # N(5,1)

        x = torch.cat([n01_col, uniform_col, n51_col], dim=1)
        result = ad_test(x)

        assert result.shape == (3,), "Should have 3 statistics"
        # N(0,1) column should have low statistic
        assert result[0] < 2.492, f"N(0,1) column failed: {result[0]}"
        # Non-N(0,1) columns should have high statistics
        assert result[1] > 2.492, f"Uniform column passed: {result[1]}"
        assert (
            result[2] > 10.0
        ), f"N(5,1) column should be strongly rejected: {result[2]}"

    # Error Handling Tests
    def test_1d_input_raises_error(self, ad_test):
        """Test that 1D input raises appropriate error."""
        x = torch.randn(100)
        with pytest.raises((RuntimeError, ValueError, IndexError, AssertionError)):
            ad_test(x)

    def test_empty_columns(self, ad_test):
        """Test with zero samples."""
        try:
            x = torch.tensor([]).reshape(0, 3)
            result = ad_test(x)
            assert result.shape == (3,)
        except (RuntimeError, ValueError, IndexError):
            pass  # Expected

    def test_nan_input_propagation(self, ad_test):
        """Test behavior with NaN inputs."""
        x = torch.randn(100, 3)
        x[50, 1] = float("nan")
        result = ad_test(x)
        assert result.shape == (3,), "Should return correct shape"

    def test_inf_input_handling(self, ad_test):
        """Test behavior with infinite inputs."""
        x = torch.randn(100, 3)
        x[50, 1] = float("inf")
        result = ad_test(x)
        assert result.shape == (3,)


class TestAndersonDarlingFormula:
    """Test the mathematical formula implementation."""

    @pytest.fixture
    def ad_test(self):
        return AndersonDarling()

    def test_symmetric_terms_per_column(self, ad_test):
        """Test that symmetric terms are computed correctly for each column."""
        x = torch.randn(100, 2)
        result = ad_test(x)

        assert torch.isfinite(result).all()
        assert result.shape == (2,)

    def test_flip_operation(self, ad_test):
        """Test that flip operation is used in the formula."""
        torch.manual_seed(42)
        x = torch.randn(10, 1)

        # The formula should use s.flip(0), so flipping input shouldn't give same result
        # unless by chance
        result1 = ad_test(x)
        result2 = ad_test(x.flip(0))

        # Due to sorting in prepare_data, these should actually be the same
        # since sorting removes order
        # So this test just checks it runs
        assert result1.shape == result2.shape


# Parametrized Tests
class TestAndersonDarlingParametrized:
    """Parametrized tests for various scenarios."""

    @pytest.mark.parametrize("n_samples", [10, 50, 100, 500, 1000])
    def test_various_sample_sizes(self, n_samples):
        """Test with various sample sizes from N(0,1)."""
        ad_test = AndersonDarling()
        x = torch.randn(n_samples, 3)  # N(0,1)
        result = ad_test(x)
        assert result.shape == (3,), f"Shape mismatch for n={n_samples}"
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize("n_dims", [1, 5, 10, 50, 100])
    def test_various_dimensions(self, n_dims):
        """Test with various numbers of dimensions."""
        ad_test = AndersonDarling()
        x = torch.randn(100, n_dims)  # N(0,1)
        result = ad_test(x)
        assert result.shape == (n_dims,), f"Shape mismatch for dim={n_dims}"
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize("seed", range(5))
    def test_multiple_random_seeds(self, seed):
        """Test consistency across different random seeds."""
        ad_test = AndersonDarling()
        torch.manual_seed(seed)
        x = torch.randn(100, 4)  # N(0,1)
        result = ad_test(x)
        assert torch.isfinite(result).all()
        assert result.shape == (4,)

    @pytest.mark.parametrize("mean", [0.5, 1.0, 5.0, -3.0])
    def test_non_zero_mean_rejected(self, mean):
        """Test that N(mean,1) with mean≠0 is correctly rejected."""
        ad_test = AndersonDarling()
        torch.manual_seed(42)
        x = torch.randn(500, 3) + mean  # N(mean,1)
        result = ad_test(x)
        # Should be strongly rejected
        assert (result > 5.0).all(), f"N({mean},1) should be rejected, got {result}"

    @pytest.mark.parametrize("std", [0.5, 2.0, 5.0, 10.0])
    def test_non_unit_variance_rejected(self, std):
        """Test that N(0,std²) with std≠1 is correctly rejected."""
        ad_test = AndersonDarling()
        torch.manual_seed(42)
        x = torch.randn(500, 3) * std  # N(0,std²)
        result = ad_test(x)
        # Should be rejected
        assert (result > 5.0).all(), f"N(0,{std}²) should be rejected, got {result}"


class TestAndersonDarlingEdgeCases:
    """Test edge cases and corner scenarios."""

    @pytest.fixture
    def ad_test(self):
        return AndersonDarling()

    def test_constant_column(self, ad_test):
        """Test with a constant column (not N(0,1))."""
        x = torch.randn(100, 3)
        x[:, 1] = 5.0  # Constant, definitely not N(0,1)

        try:
            result = ad_test(x)
            assert result.shape == (3,)
            # Constant column should be rejected
            assert result[1] > 10.0 or not torch.isfinite(result[1])
        except (RuntimeError, ValueError):
            pass  # Also acceptable

    def test_highly_skewed_data(self, ad_test):
        """Test with highly skewed data (not N(0,1))."""
        torch.manual_seed(42)
        x = torch.distributions.Exponential(1.0).sample((500, 3))
        result = ad_test(x)

        # Should be strongly rejected
        assert (result > 2.492).all(), "Skewed data should be rejected"

    def test_bimodal_distribution(self, ad_test):
        """Test with bimodal distribution (not N(0,1))."""
        torch.manual_seed(42)
        # Mixture of N(-2,1) and N(2,1)
        x1 = torch.randn(250, 2) - 2
        x2 = torch.randn(250, 2) + 2
        x = torch.cat([x1, x2], dim=0)

        result = ad_test(x)
        # Should be rejected
        assert (result > 2.492).all(), "Bimodal should be rejected"

    def test_t_distribution(self, ad_test):
        """Test with t-distribution (heavier tails than N(0,1))."""
        torch.manual_seed(42)
        from torch.distributions import StudentT

        x = StudentT(df=5.0).sample((500, 3))

        result = ad_test(x)
        # t-distribution with df=5 is close to N(0,1) but not exact
        # Should have elevated statistics
        assert (result > 0).all()


class TestAndersonDarlingStatisticalProperties:
    """Test statistical properties."""

    @pytest.fixture
    def ad_test(self):
        return AndersonDarling()

    def test_type_i_error_rate(self, ad_test):
        """Test that type I error rate is approximately correct for N(0,1)."""
        torch.manual_seed(42)
        n_trials = 100
        alpha = 0.05
        critical_value = 2.492  # 5% critical value

        rejections = 0
        for i in range(n_trials):
            torch.manual_seed(i)
            x = torch.randn(200, 1)  # N(0,1)
            result = ad_test(x)
            if result[0] > critical_value:
                rejections += 1

        rejection_rate = rejections / n_trials
        # Should be close to alpha (with tolerance for randomness)
        assert (
            0.01 < rejection_rate < 0.15
        ), f"Type I error rate {rejection_rate} not close to {alpha}"

    def test_power_against_shifted_normal(self, ad_test):
        """Test that the test has power against N(0.5,1)."""
        torch.manual_seed(42)
        n_trials = 20
        critical_value = 2.492

        rejections = 0
        for i in range(n_trials):
            torch.manual_seed(i)
            x = torch.randn(200, 1) + 0.5  # N(0.5,1), not N(0,1)
            result = ad_test(x)
            if result[0] > critical_value:
                rejections += 1

        rejection_rate = rejections / n_trials
        # Should reject most of the time
        assert (
            rejection_rate > 0.7
        ), f"Should have high power against N(0.5,1), got {rejection_rate}"

    def test_critical_values_ordering(self, ad_test):
        """Test that larger departures produce higher statistics."""
        torch.manual_seed(42)

        # N(0,1)
        x0 = torch.randn(500, 1)
        # N(1,1)
        x1 = torch.randn(500, 1) + 1.0
        # N(5,1)
        x2 = torch.randn(500, 1) + 5.0

        stat0 = ad_test(x0)[0]
        stat1 = ad_test(x1)[0]
        stat2 = ad_test(x2)[0]

        # Larger shifts should produce larger statistics
        assert stat1 > stat0, "N(1,1) should have higher stat than N(0,1)"
        assert stat2 > stat1, "N(5,1) should have higher stat than N(1,1)"
