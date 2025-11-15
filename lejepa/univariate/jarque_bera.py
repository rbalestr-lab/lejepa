from .base import UnivariateTest


class VCReg(UnivariateTest):
    """
    VCReg (Variance-Covariance Regularization) test for standard normality N(0,1).

    This test checks only the first two moments (mean=0, var=1) to determine
    if data follows a standard normal distribution. Unlike the full Jarque-Bera
    test which includes skewness and kurtosis, VCReg focuses solely on location
    and scale parameters.

    The test statistic is the sum of two components:
        VCReg = S_mean + S_var

    where:
        - S_mean = n·μ̂² / σ̂² tests H₀: μ = 0
        - S_var = (n-1)·(σ̂² - 1)² / 2 tests H₀: σ² = 1

    Under H₀: X ~ N(0,1), the statistic follows approximately χ²(2).

    Returns
    -------
    torch.Tensor
        Test statistic (scalar or shape matching input features). Higher values
        indicate stronger evidence against N(0,1).

    Notes
    -----
    This is a simplified normality test that does NOT check skewness or kurtosis.
    For a complete four-moment test, use ExtendedJarqueBera instead.
    """

    def forward(self, x):
        """
        Compute VCReg test statistic for mean and variance components only.

        Args:
            x (torch.Tensor): Tensor of shape (N, ...), where the test is performed over dim=0.

        Returns:
            torch.Tensor: Test statistic, shape (...). Under H₀: X ~ N(0,1), 
                         follows approximately χ²(2) distribution.
        """
        n = x.shape[0]
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=True)
        # Test statistics for each moment
        # 1. Mean: (mean^2) / (var / n) ~ chi2(1)
        stat_mean = (mean**2) / (var / n)
        # 2. Variance: ((var - 1)^2) / (2 / (n-1)) ~ chi2(1)
        stat_var = ((var - 1) ** 2) / (2 / (n - 1))
        # Total statistic: sum of the two
        stat = stat_mean + stat_var

        # p-value for chi-squared with 4 degrees of freedom
        # CDF: 1 - gammainc(2, stat/2)
        # For 4 dof: 1 - (1 + stat/2 + (stat/2)**2/2) * exp(-stat/2)
        # p_value = 1 - (1 + stat / 2 + (stat / 2) ** 2 / 2) * torch.exp(-stat / 2)
        # moments = {"mean": mean, "var": var, "skewness": skewness, "kurtosis": kurtosis}
        return stat


class ExtendedJarqueBera(UnivariateTest):
    """
    Extended Jarque-Bera test for goodness-of-fit to standard normal N(0,1).
    This test extends the classical Jarque-Bera test by examining all four
    moments of the standard normal distribution: mean (μ=0), variance (σ²=1),
    skewness (γ₁=0), and kurtosis (γ₂=3). Unlike the standard Jarque-Bera test
    which only checks the third and fourth moments, this version provides a
    comprehensive omnibus test for the complete specification of N(0,1).
    The test statistic is computed as the sum of four components:
        JB_ext = S_mean + S_var + S_skew + S_kurt
    where each component tests a specific moment:
        1. Mean:     S_mean = n·μ̂² / σ̂²           ~ χ²(1)
        2. Variance: S_var = (n-1)·(σ̂² - 1)² / 2   ~ χ²(1)
        3. Skewness: S_skew = n·γ̂₁² / 6            ~ χ²(1)
        4. Kurtosis: S_kurt = n·(γ̂₂ - 3)² / 24     ~ χ²(1)
    Under H₀: X ~ N(0,1), the total statistic JB_ext ~ χ²(4).
    Components
    ----------
    Mean component:
        Tests H₀: μ = 0 using the standardized squared mean.
        For N(μ,σ²), √n(μ̂-μ)/σ ~ N(0,1), so n·μ̂²/σ̂² ~ χ²(1).
    Variance component:
        Tests H₀: σ² = 1 using the squared deviation of sample variance.
        For N(0,σ²), (n-1)·σ̂²/σ² ~ χ²(n-1), leading to the approximation.
    Skewness and Kurtosis components:
        These follow the classical Jarque-Bera formulation.
        For normal data, γ̂₁ ~ N(0, 6/n) and γ̂₂ ~ N(3, 24/n) asymptotically.
    Parameters
    ----------
    x : torch.Tensor
        Input samples to test for N(0,1). Should be shape (n, d) where the
        test is applied along dimension 0 for each of d features independently.
    Returns
    -------
    stat : torch.Tensor
        The extended Jarque-Bera test statistic. Shape matches the feature
        dimensions (d,). Higher values indicate stronger evidence against
        N(0,1). Under the null hypothesis, follows χ²(4) distribution.
    Notes
    -----
    - The test statistic is the sum of four independent χ²(1) components,
      yielding a χ²(4) distribution under H₀: X ~ N(0,1)
    - Critical values at common significance levels (α):
        * α = 0.10: 7.779
        * α = 0.05: 9.488
        * α = 0.01: 13.277
    - This test has high power against:
        * Location shifts (detects μ ≠ 0)
        * Scale changes (detects σ² ≠ 1)
        * Skewed distributions (detects γ₁ ≠ 0)
        * Heavy or light tails (detects γ₂ ≠ 3)
    - The test requires n ≥ 4 for meaningful results (preferably n ≥ 20)
    - Each component can be examined separately to diagnose which moment
      deviates from N(0,1)
    Advantages over Standard Jarque-Bera
    -------------------------------------
    - Tests the complete specification of N(0,1), not just shape
    - Can detect location and scale departures
    - More powerful for detecting multi-modal or shifted distributions
    - Provides 4 degrees of freedom for the omnibus test
    Disadvantages
    -------------
    - Less powerful than specialized tests for individual moments
    - Requires larger sample sizes for accurate χ² approximation
    - Not invariant to location-scale transformations (tests N(0,1) specifically)
    References
    ----------
    .. [1] Jarque, C. M., & Bera, A. K. (1980). "Efficient tests for normality,
           homoscedasticity and serial independence of regression residuals".
           Economics Letters, 6(3), 255-259.
    .. [2] Jarque, C. M., & Bera, A. K. (1987). "A test for normality of
           observations and regression residuals". International Statistical
           Review, 55(2), 163-172.
    .. [3] Thadewald, T., & Büning, H. (2007). "Jarque-Bera test and its
           competitors for testing normality–a power comparison". Journal of
           Applied Statistics, 34(1), 87-105.
    .. [4] https://en.wikipedia.org/wiki/Jarque–Bera_test
    See Also
    --------
    AndersonDarling : EDF-based test for N(0,1)
    ShapiroWilk : Correlation-based test for N(0,1)
    CramerVonMises : Another EDF-based test for N(0,1)
    Examples
    --------
    >>> # Test if samples follow N(0,1)
    >>> x = torch.randn(1000, 5)
    >>> test = ExtendedJarqueBera()
    >>> statistic = test(x)
    >>> # If statistic > 9.488, reject N(0,1) at 5% significance level
    >>> (statistic < 9.488).all()
    True

    >>> # Test with shifted data (mean ≠ 0)
    >>> x_shifted = torch.randn(1000, 3) + 2.0
    >>> stat_shifted = test(x_shifted)
    >>> # Should have very high statistic due to mean component
    >>> (stat_shifted > 100).all()
    True

    >>> # Test with scaled data (variance ≠ 1)
    >>> x_scaled = torch.randn(1000, 3) * 3.0
    >>> stat_scaled = test(x_scaled)
    >>> # Should have high statistic due to variance component
    >>> (stat_scaled > 20).all()
    True

    >>> # Test with skewed data
    >>> x_skewed = torch.distributions.Exponential(1.0).sample((1000, 3))
    >>> stat_skewed = test(x_skewed)
    >>> # Should have high statistic due to all components
    >>> (stat_skewed > 50).all()
    True
    """

    def forward(self, x):
        """
        Computes an extended Jarque-Bera test statistic and p-value for normality,
        testing all four moments (mean=0, var=1, skew=0, kurt=3) along the first dimension.

        Args:
            x (torch.Tensor): Tensor of shape (N, ...), where the test is performed over dim=0.

        Returns:
            stat (torch.Tensor): Extended test statistic, shape (...).
            p_value (torch.Tensor): p-value for the test statistic, shape (...).
            moments (dict): Dictionary with mean, var, skewness, kurtosis tensors.
        """
        n = x.shape[0]
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=True)
        std = var.sqrt().clamp(min=1e-8)
        skewness = ((x - mean) / std).pow(3).mean(dim=0)
        kurtosis = ((x - mean) / std).pow(4).mean(dim=0)

        # Test statistics for each moment
        # 1. Mean: (mean^2) / (var / n) ~ chi2(1)
        stat_mean = (mean**2) / (var / n)
        # 2. Variance: ((var - 1)^2) / (2 / (n-1)) ~ chi2(1)
        stat_var = ((var - 1) ** 2) / (2 / (n - 1))
        # 3. Skewness and 4. Kurtosis: Jarque-Bera part
        stat_skew_kurt = n / 6 * (skewness**2 + 0.25 * (kurtosis - 3) ** 2)

        # Total statistic: sum of all four
        stat = stat_mean + stat_var + stat_skew_kurt

        # p-value for chi-squared with 4 degrees of freedom
        # CDF: 1 - gammainc(2, stat/2)
        # For 4 dof: 1 - (1 + stat/2 + (stat/2)**2/2) * exp(-stat/2)
        # p_value = 1 - (1 + stat / 2 + (stat / 2) ** 2 / 2) * torch.exp(-stat / 2)
        # moments = {"mean": mean, "var": var, "skewness": skewness, "kurtosis": kurtosis}
        return stat
