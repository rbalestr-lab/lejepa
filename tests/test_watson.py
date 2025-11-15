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
from lejepa.univariate import (
    CramerVonMises,
    Watson,
)


class TestWatson:
    """Test suite for Watson test for standard normal N(0,1)."""

    @pytest.fixture
    def watson_test(self):
        """Create a Watson test instance."""
        return Watson()

    @pytest.fixture
    def cvm_test(self):
        """Create a CramerVonMises test instance for comparison."""
        return CramerVonMises()

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
    def test_forward_returns_correct_shape(
        self, watson_test, standard_normal_samples_2d
    ):
        """Test that forward returns shape (dim,) for 2D input (n, dim)."""
        n, dim = standard_normal_samples_2d.shape
        result = watson_test(standard_normal_samples_2d)
        assert result.shape == (dim,), f"Expected shape ({dim},), got {result.shape}"
        assert isinstance(result, torch.Tensor), "Result should be a Tensor"

    def test_forward_single_column(self, watson_test):
        """Test with single column input (n, 1)."""
        torch.manual_seed(42)
        x = torch.randn(100, 1)
        result = watson_test(x)
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
        assert torch.isfinite(result).all(), "Result should be finite"

    def test_inherits_from_cramer_von_mises(self, watson_test):
        """Test that Watson inherits from CramerVonMises."""
        assert isinstance(
            watson_test, CramerVonMises
        ), "Watson should inherit from CramerVonMises"

    def test_standard_normal_samples_low_statistic(
        self, watson_test, standard_normal_samples_2d
    ):
        """Test that N(0,1) samples produce low test statistics."""
        result = watson_test(standard_normal_samples_2d)
        # N(0,1) samples should have small statistics
        assert (result < 0.5).all(), f"Statistics too high for N(0,1) data: {result}"

    def test_non_standard_normal_samples_high_statistic(
        self, watson_test, non_standard_normal_samples_2d
    ):
        """Test that non-N(0,1) samples produce high test statistics."""
        result = watson_test(non_standard_normal_samples_2d)
        # Uniform samples should have higher statistics
        assert (
            result > 0.001
        ).all(), f"Statistics too low for non-N(0,1) data: {result}"

    def test_independent_columns(self, watson_test):
        """Test that each column is tested independently."""
        torch.manual_seed(42)
        col1 = torch.randn(500, 1)  # N(0,1)
        col2 = torch.rand(500, 1)  # Uniform
        col3 = torch.randn(500, 1) + 5  # N(5,1)

        x = torch.cat([col1, col2, col3], dim=1)
        result = watson_test(x)

        assert result.shape == (3,), "Should have 3 statistics"
        # Column 1 should have lowest statistic
        assert result[0] < result[1], "N(0,1) should have lower stat than uniform"
        assert result[0] < result[2], "N(0,1) should have lower stat than N(5,1)"

    # Comparison with CVM Tests
    def test_watson_equals_cvm_minus_correction(self, watson_test, cvm_test):
        """Test that Watson = CVM - (m - 0.5)²."""
        torch.manual_seed(42)
        x = torch.randn(200, 3)

        watson_stat = watson_test(x)
        cvm_stat = cvm_test(x)

        # Compute correction term manually
        # Need to access the distribution - assuming it's stored in watson_test.g
        m = watson_test.g.cdf(watson_test.prepare_data(x)).mean(0)
        correction = (m - 0.5).square()

        expected_watson = cvm_stat - correction

        assert torch.allclose(
            watson_stat, expected_watson, rtol=1e-5
        ), f"Watson should equal CVM minus correction"

    def test_watson_less_than_or_equal_cvm_for_n01(self, watson_test, cvm_test):
        """Test that Watson ≤ CVM for N(0,1) data (correction is non-negative)."""
        torch.manual_seed(42)
        x = torch.randn(500, 5)

        watson_stat = watson_test(x)
        cvm_stat = cvm_test(x)

        # Watson = CVM - (m - 0.5)², and (m - 0.5)² ≥ 0
        # So Watson ≤ CVM
        assert (
            watson_stat <= cvm_stat + 1e-6
        ).all(), f"Watson should be ≤ CVM (correction is non-negative)"

    def test_watson_correction_small_for_n01(self, watson_test, cvm_test):
        """Test that correction term is small for N(0,1) data."""
        torch.manual_seed(42)
        x = torch.randn(1000, 3)  # Large sample

        watson_stat = watson_test(x)
        cvm_stat = cvm_test(x)
        correction = cvm_stat - watson_stat

        # For N(0,1), mean CDF should be close to 0.5, so correction should be small
        assert (
            correction < 0.01
        ).all(), f"Correction should be small for N(0,1): {correction}"

    def test_watson_correction_large_for_shifted_normal(self, watson_test, cvm_test):
        """Test that correction term is larger for shifted normal."""
        torch.manual_seed(42)
        x = torch.randn(500, 2) + 2.0  # N(2,1)

        watson_stat = watson_test(x)
        cvm_stat = cvm_test(x)
        correction = cvm_stat - watson_stat

        # For N(2,1), mean CDF will be > 0.5, so correction should be larger
        assert (
            correction > 0.01
        ).all(), f"Correction should be larger for shifted normal: {correction}"

    # Edge Cases
    def test_small_sample_size(self, watson_test):
        """Test with small sample sizes."""
        x = torch.randn(10, 3)
        result = watson_test(x)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()

    def test_minimum_sample_size(self, watson_test):
        """Test with minimum viable sample size."""
        x = torch.randn(3, 4)
        result = watson_test(x)
        assert result.shape == (4,)
        assert torch.isfinite(result).all()

    def test_single_sample(self, watson_test):
        """Test with single sample."""
        x = torch.randn(1, 3)
        result = watson_test(x)
        assert result.shape == (3,)

    def test_large_sample_size(self, watson_test):
        """Test with large sample size."""
        torch.manual_seed(42)
        x = torch.randn(5000, 3)
        result = watson_test(x)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()
        # Large N(0,1) sample should have low statistic
        assert (result < 0.05).all()

    def test_large_dimensionality(self, watson_test):
        """Test with large number of dimensions."""
        torch.manual_seed(42)
        x = torch.randn(100, 100)
        result = watson_test(x)
        assert result.shape
