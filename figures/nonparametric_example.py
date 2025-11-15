import matplotlib.pyplot as plt
from tqdm.rich import trange
import torch
import lejepa as ds

# Data generation parameters
N_SAMPLES = 100
DEVICE = "cpu"
RANDOM_SEED = 0

# Optimization parameters
LEARNING_RATE = 0.1
NUM_OPTIMIZATION_STEPS = 1520

# Slicing test parameters
NUM_SLICES_OPTIONS = [10, 100]
DIMENSION_OPTIONS = [128, 1024]

# Plot parameters
SCATTER_ALPHA = 0.5
SCATTER_LINEWIDTH = 1
SCATTER_ZORDER = 1000
FIGURE_HEIGHT = 6
COLUMN_WIDTH = 2


def get_X(N, dim):
    torch.manual_seed(RANDOM_SEED)
    X = torch.randn(N, dim, device=DEVICE)
    X[:, 1] = X[:, 0] * ((torch.rand(X.size(0), device=DEVICE) > 0.5).float() * 2 - 1)
    X[:, :2] = X[:, :2] + 1
    return X


def create_univariate_tests():
    """Create test instances once to avoid repeated instantiation."""
    return [
        ds.univariate.VCReg(),
        ds.univariate.ExtendedJarqueBera(),
        ds.univariate.CramerVonMises(),
        ds.univariate.Watson(),
        ds.univariate.AndersonDarling(),
        ds.univariate.EppsPulley(),
    ]


tests = create_univariate_tests()
for num_slices in NUM_SLICES_OPTIONS:
    for i, dim in enumerate(DIMENSION_OPTIONS):
        fig, axs = plt.subplots(
            2,
            len(tests) + 1,
            sharex="row",
            sharey="row",
            figsize=(COLUMN_WIDTH * (len(tests) + 1), FIGURE_HEIGHT),
        )
        X = get_X(N_SAMPLES, dim)
        with torch.no_grad():
            axs[0, 0].scatter(
                X[:, 0].cpu(),
                X[:, 1].cpu(),
                c="grey",
                linewidth=SCATTER_LINEWIDTH,
                edgecolor="k",
                alpha=SCATTER_ALPHA,
                zorder=SCATTER_ZORDER,
            )
            axs[1, 0].scatter(
                X[:, 2].cpu(),
                X[:, 3].cpu(),
                c="grey",
                linewidth=SCATTER_LINEWIDTH,
                edgecolor="k",
                alpha=SCATTER_ALPHA,
                zorder=SCATTER_ZORDER,
            )
        axs[0, 0].set_title("original data")
        for j, test in enumerate(tests):
            print(test)
            torch.manual_seed(RANDOM_SEED)
            if isinstance(test, ds.univariate.UnivariateTest):
                g_loss = ds.multivariate.SlicingUnivariateTest(
                    dim=1, univariate_test=test, num_slices=num_slices
                )
            else:
                g_loss = test
            Xp = X.clone().detach().requires_grad_(True)
            optim = torch.optim.Adam([Xp], lr=LEARNING_RATE)
            losses = []
            for step in trange(NUM_OPTIMIZATION_STEPS):
                optim.zero_grad()
                loss = g_loss(Xp)
                losses.append(loss.item())
                loss.backward()
                optim.step()
            # axs[0,j].plot(losses, label=stat.__name__)
            with torch.no_grad():
                axs[0, j + 1].scatter(
                    Xp[:, 0].cpu(),
                    Xp[:, 1].cpu(),
                    c="green",
                    linewidth=SCATTER_LINEWIDTH,
                    edgecolor="k",
                    alpha=SCATTER_ALPHA,
                    zorder=SCATTER_ZORDER,
                )
                axs[1, j + 1].scatter(
                    Xp[:, 2].cpu(),
                    Xp[:, 3].cpu(),
                    c="green",
                    linewidth=SCATTER_LINEWIDTH,
                    edgecolor="k",
                    alpha=SCATTER_ALPHA,
                    zorder=SCATTER_ZORDER,
                )
            axs[0, j + 1].set_title(str(test)[:-2])
        for i in range(axs.shape[1]):
            axs[0, i].set_xlabel("dim 1")
            axs[1, i].set_xlabel("dim 3")
        axs[0, 0].set_ylabel("dim 2")
        axs[1, 0].set_ylabel("dim 4")
        plt.tight_layout()
        plt.savefig(f"2d_slicing_dim_{dim}_N_{N_SAMPLES}_slices_{num_slices}.pdf")
        plt.close()
