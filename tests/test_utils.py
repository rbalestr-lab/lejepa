import pytest
import torch
import numpy as np
from scipy import stats
from lejepa.univariate.utils import log_norm_cdf, log_norm_cdf_helper, norm_cdf


@pytest.mark.unit
class TestNormCDF:
    """Test standard normal CDF implementation."""

    def test_basic_values(self):
        """Test known values of the normal CDF."""
        x = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        result = norm_cdf(x)
        scipy_result = stats.norm.cdf(x.numpy())

        np.testing.assert_allclose(result.numpy(), scipy_result, rtol=1e-6, atol=1e-8)

    def test_zero(self):
        """Test that Φ(0) = 0.5 exactly."""
        result = norm_cdf(torch.tensor(0.0, dtype=torch.float64)).item()
        assert abs(result - 0.5) < 1e-10

    def test_symmetry(self):
        """Test that Φ(-x) = 1 - Φ(x)."""
        x = torch.linspace(-5, 5, 50, dtype=torch.float64)
        left = norm_cdf(-x)
        right = 1 - norm_cdf(x)
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)

    def test_against_scipy(self):
        """Test against scipy reference implementation."""
        x = torch.linspace(-10, 10, 1000, dtype=torch.float64)
        torch_result = norm_cdf(x).numpy()
        scipy_result = stats.norm.cdf(x.numpy())

        np.testing.assert_allclose(torch_result, scipy_result, rtol=1e-6, atol=1e-8)

    def test_bounds(self):
        """Test that output is always in [0, 1]."""
        x = torch.randn(1000, dtype=torch.float64) * 10
        result = norm_cdf(x)

        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_monotonic(self):
        """Test that CDF is monotonically increasing."""
        x = torch.linspace(-10, 10, 1000, dtype=torch.float64)
        result = norm_cdf(x)

        diffs = result[1:] - result[:-1]
        assert (diffs >= -1e-10).all()

    def test_dtypes(self):
        """Test different dtypes."""
        for dtype in [torch.float32, torch.float64]:
            x = torch.tensor([0.0, 1.0, -1.0], dtype=dtype)
            result = norm_cdf(x)
            assert result.dtype == dtype


# ============================================================================
# Test log_norm_cdf_helper
# ============================================================================


@pytest.mark.unit
class TestLogNormCDFHelper:
    """Test the helper function for tail approximation."""

    def test_positive_output(self):
        """Test that output is positive for x > 0."""
        x = torch.linspace(0.1, 10, 100, dtype=torch.float64)
        result = log_norm_cdf_helper(x)
        assert (result > 0).all()

    def test_no_nan(self):
        """Test that no NaN values are produced for valid inputs."""
        x = torch.linspace(3, 10, 100, dtype=torch.float64)
        result = log_norm_cdf_helper(x)
        assert not torch.isnan(result).any()

    def test_increasing(self):
        """Test that helper is monotonically increasing for x > 0."""
        x = torch.linspace(3, 10, 100, dtype=torch.float64)
        result = log_norm_cdf_helper(x)
        diffs = result[1:] - result[:-1]
        assert (diffs >= -1e-10).all()


# ============================================================================
# Test log_norm_cdf
# ============================================================================


@pytest.mark.unit
class TestLogNormCDF:
    """Test log normal CDF implementation."""

    def test_basic_functionality(self):
        """Test that function runs without errors."""
        x = torch.linspace(-10, 10, 100, dtype=torch.float64)
        result = log_norm_cdf(x)

        assert result.shape == x.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_middle_region_accuracy(self):
        """Test accuracy in middle region where erf is used directly."""
        # In [-3, 3] range, the function uses norm_cdf directly
        x = torch.linspace(-2.5, 2.5, 100, dtype=torch.float64)
        torch_result = log_norm_cdf(x).numpy()
        scipy_result = stats.norm.logcdf(x.numpy())

        # Should be very accurate in the middle region
        np.testing.assert_allclose(torch_result, scipy_result, rtol=1e-5, atol=1e-7)

    def test_zero(self):
        """Test that log(Φ(0)) = log(0.5)."""
        result = log_norm_cdf(torch.tensor(0.0, dtype=torch.float64))
        expected = np.log(0.5)
        np.testing.assert_allclose(result.item(), expected, rtol=1e-6)

    def test_output_is_negative(self):
        """Test that output is always negative or zero (since Φ(x) ≤ 1)."""
        x = torch.linspace(-10, 10, 1000, dtype=torch.float64)
        result = log_norm_cdf(x)

        # log(Φ(x)) ≤ log(1) = 0
        # Allow tiny positive values due to numerical errors
        assert (result <= 1e-6).all()

    def test_consistency_with_norm_cdf_middle_region(self):
        """Test that exp(log_norm_cdf(x)) ≈ norm_cdf(x) in middle region."""
        # Only test middle region where both should be accurate
        x = torch.linspace(-2.5, 2.5, 100, dtype=torch.float64)
        log_result = log_norm_cdf(x)
        direct_result = norm_cdf(x)

        torch.testing.assert_close(
            log_result.exp(), direct_result, rtol=1e-5, atol=1e-7
        )

    def test_monotonic(self):
        """Test that log CDF is monotonically increasing."""
        x = torch.linspace(-10, 10, 1000, dtype=torch.float64)
        result = log_norm_cdf(x)

        # Check differences are non-negative (with tolerance)
        diffs = result[1:] - result[:-1]
        assert (
            diffs >= -1e-6
        ).all(), f"Non-monotonic at indices: {torch.where(diffs < -1e-6)[0]}"

    def test_batch_input(self):
        """Test with batch/multidimensional input."""
        x = torch.randn(10, 20, 30, dtype=torch.float64)
        result = log_norm_cdf(x)

        assert result.shape == x.shape
        assert not torch.isnan(result).any()

    def test_float32_dtype(self):
        """Test with float32."""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
        result = log_norm_cdf(x)

        assert result.dtype == torch.float32
        assert not torch.isnan(result).any()

    def test_float64_dtype(self):
        """Test with float64."""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
        result = log_norm_cdf(x)

        assert result.dtype == torch.float64
        assert not torch.isnan(result).any()

    def test_gradient_computation(self):
        """Test that gradients can be computed correctly."""
        x = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64, requires_grad=True)
        result = log_norm_cdf(x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

        # Gradient should be positive (CDF is increasing)
        assert (x.grad > 0).all()

    def test_gradient_at_zero(self):
        """Test gradient at x=0 matches theoretical value."""
        x = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        result = log_norm_cdf(x)
        result.backward()

        # Gradient of log(Φ(x)) at x=0 is φ(0)/Φ(0) = (1/√(2π)) / 0.5 = √(2/π)
        expected_grad = np.sqrt(2 / np.pi)
        actual_grad = x.grad.item()

        np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-5)

    def test_symmetric_behavior(self):
        """Test that behavior is roughly symmetric around zero."""
        x_pos = torch.linspace(0.1, 2.5, 50, dtype=torch.float64)
        x_neg = -x_pos

        result_pos = log_norm_cdf(x_pos)
        result_neg = log_norm_cdf(x_neg)

        # Both should be negative
        assert (result_pos < 0).all()
        assert (result_neg < 0).all()

        # Positive x should give values closer to 0
        assert (result_pos > result_neg).all()


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_tensor(self):
        """Test with empty tensor."""
        x = torch.tensor([], dtype=torch.float64)
        result = log_norm_cdf(x)
        assert result.shape == (0,)

    def test_single_value(self):
        """Test with single scalar value."""
        x = torch.tensor(1.5, dtype=torch.float64)
        result = log_norm_cdf(x)
        assert result.shape == ()
        assert not torch.isnan(result)

    def test_very_large_positive_values(self):
        """Test numerical stability for very large positive values."""
        x = torch.tensor([10.0, 50.0, 100.0], dtype=torch.float64)
        result = log_norm_cdf(x)

        # Should be close to 0 (log(1) = 0) and not produce inf/nan
        assert (result <= 0).all()
        assert not torch.isinf(result).any()
        assert not torch.isnan(result).any()

    def test_very_large_negative_values(self):
        """Test numerical stability for very large negative values."""
        x = torch.tensor([-10.0, -50.0, -100.0], dtype=torch.float64)
        result = log_norm_cdf(x)

        # Should be very negative but not inf
        assert (result < -10).all()
        assert not torch.isinf(result).any()
        assert not torch.isnan(result).any()

    def test_mixed_dimensions(self):
        """Test with different tensor shapes."""
        shapes = [
            (10,),
            (5, 10),
            (2, 3, 4),
            (2, 3, 4, 5),
        ]

        for shape in shapes:
            x = torch.randn(shape, dtype=torch.float64)
            result = log_norm_cdf(x)
            assert result.shape == shape
            assert not torch.isnan(result).any()


# ============================================================================
# Functional Properties
# ============================================================================


@pytest.mark.unit
class TestFunctionalProperties:
    """Test mathematical properties of the function."""

    def test_satisfies_cdf_properties(self):
        """Test that the function satisfies basic CDF properties."""
        x = torch.linspace(-10, 10, 1000, dtype=torch.float64)
        result = log_norm_cdf(x)

        # 1. Should be monotonically increasing
        assert (result[1:] >= result[:-1] - 1e-6).all()

        # 2. Should approach 0 as x → ∞ (log scale)
        assert result[-10:].mean() > result[10:20].mean()

        # 3. Should be very negative for small x
        assert result[0] < result[-1]

    def test_comparison_with_direct_log(self):
        """Compare with direct log(norm_cdf(x)) where it's numerically stable."""
        # In middle range, both methods should work
        x = torch.linspace(-2, 2, 50, dtype=torch.float64)

        result_optimized = log_norm_cdf(x)
        result_direct = torch.log(norm_cdf(x))

        # Should match closely in this range
        torch.testing.assert_close(
            result_optimized, result_direct, rtol=1e-5, atol=1e-7
        )


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.unit
class TestIntegration:
    """Integration tests combining multiple features."""

    def test_batch_gradient_computation(self):
        """Test gradient computation on batch input."""
        x = torch.randn(10, 20, dtype=torch.float64, requires_grad=True)
        result = log_norm_cdf(x)
        loss = result.sum()
        loss.backward()

        assert x.grad.shape == x.shape
        assert not torch.isnan(x.grad).any()
        assert (x.grad > 0).all()  # Should be positive

    def test_mixed_regions_in_batch(self):
        """Test batch containing values from all regions."""
        x = torch.tensor(
            [
                -10.0,  # Far left tail
                -5.0,  # Left tail
                -3.0,  # Threshold
                0.0,  # Middle
                3.0,  # Threshold
                5.0,  # Right tail
                10.0,  # Far right tail
            ],
            dtype=torch.float64,
        )

        result = log_norm_cdf(x)

        # Basic sanity checks
        assert result.shape == x.shape
        assert not torch.isnan(result).any()
        assert (result <= 0).all()

        # Should be monotonically increasing
        assert (result[1:] >= result[:-1] - 1e-6).all()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit", "--tb=short"])
