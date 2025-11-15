import numpy as np
import torch
from .utils import log_norm_cdf
from .base import UnivariateTest


class NLL(UnivariateTest):
    def __init__(self, alpha: float = 0.5, k: int = None, N: int = None):
        super().__init__()
        assert 0 <= alpha <= 0.5
        self.alpha = alpha
        assert k is None or type(k) is int
        assert N is None or type(N) is int
        self.k = k
        self.N = N
        self.g = torch.distributions.Normal(0, 1)
        self._cached = (-1, -1)

    @torch.no_grad()
    def get_cutoffs(self, device, ndim):
        assert self.N is not None
        if self.alpha == 0.5:
            return
        elif self._cached == (self.k, self.N):
            return self.cutoffs

        original_alpha = self.alpha
        original_k = self.k
        self.alpha = 0.5

        samples = torch.linspace(-7, 7, 1000, device=device)
        if self.k is None:
            candidates = range(1, self.N + 1)
        else:
            assert type(self.k) is int and self.k >= 1 and self.k <= self.N
            candidates = [self.k]

        density = torch.empty((len(candidates), len(samples)), device=device)
        for i, k in enumerate(candidates):
            self.k = k
            density[i] = self.forward(samples)

        density.negative_().exp_()
        # cumulative area
        density = torch.cumsum(
            (density[:, 1:] + density[:, :-1]).mul_((samples[1] - samples[0]) / 2), 1
        )
        cutoff_low = (density > original_alpha).float().argmax(1)
        cutoff_high = (density > 1 - original_alpha).float().argmax(1)
        cutoffs = torch.stack([samples[cutoff_low], samples[cutoff_high]], 0)
        self.cutoffs = cutoffs.view((2, len(candidates)) + (1,) * (ndim - 1))

        self.k = original_k
        self.alpha = original_alpha
        self._cached = (self.k, self.N)
        return self.cutoffs

    @torch.no_grad()
    def get_constants(self, device, ndim: int):
        assert self.N is not None
        top = torch.arange(1, self.N + 1).log().sum()
        # case of a single order statistic
        # in this case:
        # - cst is a scalar
        # - k_factors and N_m_k_factors are tensors with 1 element
        if type(self.k) is int:
            assert self.k >= 1 and self.k <= self.N
            k_factors = torch.full((1,), self.k - 1, device=device, dtype=torch.float)
            bottom_left = torch.arange(self.k).log()
            bottom_left[0] = 0
            bottom_left = bottom_left.sum()
            bottom_right = torch.arange(1, self.N - self.k + 1, device=device).log()
            bottom_right = bottom_right.sum()
            N_m_k_factors = torch.full(
                (1,), self.N - self.k, device=device, dtype=torch.float
            )
        # case where each sample is its own order statistic
        # in this case:
        # - cst, k_factors and N_m_k_factors are N-long vectors
        else:
            k_factors = torch.arange(self.N, device=device, dtype=torch.float)
            bottom_left = k_factors.log()
            bottom_left[0] = 0
            torch.cumsum(bottom_left, dim=0, out=bottom_left)
            bottom_right = bottom_left.flip(0)
            N_m_k_factors = k_factors.flip(0)
        cst = top - bottom_left - bottom_right

        extra_dims = ndim - 1
        cst = cst.view([-1] + [1] * extra_dims)
        k_factors = k_factors.view([-1] + [1] * extra_dims)
        N_m_k_factors = N_m_k_factors.view([-1] + [1] * extra_dims)
        return k_factors, N_m_k_factors, cst

    def forward(self, x):
        if torch.isnan(x).any():
            raise ValueError("Given input to the loss contains NaN!")
        if x.ndim < 1:
            raise ValueError(f"input should have at least one dim, got {x.ndim}")
        if self.N is None:
            N_was_None = True
            self.N = x.size(0)
        else:
            N_was_None = False

        # in case we use bounds
        cutoffs = self.get_cutoffs(x.device, x.ndim)
        # get constants
        k_factors, N_m_k_factors, cst = self.get_constants(x.device, x.ndim)

        if self.k is None:
            s, indices = torch.sort(x, dim=0)
        else:
            s = x

        logcdf = log_norm_cdf(s)  # standard_gaussian_survival(-s).log()
        one_m_logcdf = log_norm_cdf(-s)  # standard_gaussian_survival(s).log()

        sample_loss = -(
            cst
            + logcdf.mul(k_factors)
            + one_m_logcdf.mul(N_m_k_factors)
            + self.g.log_prob(s)
        )
        if self.alpha < 0.5:
            assert cutoffs is not None
            mask = s.gt(cutoffs[0]).logical_and_(s.lt(cutoffs[1]))
            sample_loss[mask] = torch.nan
        if N_was_None:
            self.N = None

        if self.k is None:
            # we reorder
            return torch.gather(sample_loss, dim=0, index=indices)
        return sample_loss


if __name__ == "__main__":
    import torch

    loss = NLL()
    NLL()(torch.randn(10, 10))
    NLL()(torch.randn(10, 1))
    NLL()(torch.randn(10, 10, 10, 10))
    loss = NLL(k=5)
    NLL()(torch.randn(10, 10))
    NLL()(torch.randn(10, 1))
    NLL()(torch.randn(10, 10, 10, 10))

    x = torch.linspace(-5, 5, 400).double()
    import matplotlib.pyplot as plt
    from scipy.integrate import simpson

    fig, axs = plt.subplots(2, 3, sharex="all")
    for k in [1, 50, 200, 400]:
        L = NLL(k=k)
        ll = L(x)
        axs[1, 0].plot(x, ll)
        density = (-ll).exp()
        volume = simpson(density, x)
        axs[0, 0].plot(x, density, label=np.round(volume, 4))

        L = NLL(k=k, alpha=0.1)
        ll = L(x)
        axs[1, 1].plot(x, ll)
        density = (-ll).exp()
        (line,) = axs[0, 1].plot(x, density)
    for alpha in [0.5, 0.3, 0.1, 0.01]:
        L = NLL(k=k, alpha=alpha)
        ll = L(x)
        axs[1, 2].plot(x, ll, label=alpha)
        density = (-ll).exp()
        axs[0, 2].plot(x, density)
    axs[1, 2].legend(title="alpha")

    axs[0, 0].legend()

    plt.savefig("test.png")
    plt.close()
