import seaborn as sns
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import FormatStrFormatter

import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import lejepa as ds

torch.manual_seed(10)
np.random.seed(10)

# Configuration constants
PROJECTION_OFFSET = 150  # Vertical spacing between projections
NUM_PROJECTIONS = 10  # Number of random projection angles
PROJECTION_RADIUS = 2  # Length of projection arrows
HISTOGRAM_BINS = 50  # Number of bins for projection histograms
ARROW_WIDTH = 0.1  # Width of projection arrows in plot
KDE_LEVELS = 30  # Number of contour levels in KDE plot
FIGURE_WIDTH = 11  # Figure width in inches
FIGURE_HEIGHT = 2.5  # Figure height in inches
PLOT_LINEWIDTH = 1  # Line width for histogram outlines
EDGE_LINEWIDTH = 2  # Line width for scatter plot edges
MARKER_SIZE = 7  # Size of legend markers


def create_univariate_tests():
    """
    Create instances of univariate normality tests.
    
    Instantiating tests outside the loop avoids repeated object creation.
    
    Returns:
        dict: Mapping of test names to test instances
    """
    return {
        'cramer_von_mises': ds.univariate.CramerVonMises(),
        'vcreg': ds.univariate.VCReg(),
        'watson': ds.univariate.Watson(),
        'anderson_darling': ds.univariate.AndersonDarling(),
        'ext_jarque_beta': ds.univariate.ExtendedJarqueBera(),
        'epps_pulley': ds.univariate.EppsPulley(),
    }


def generate_figure(X1, X2):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Define marker styles for each test
    markers = {
        "vcreg": "o",
        "anderson_darling": "^",
        "watson": "v",
        "cramer_von_mises": "s",
        "epps_pulley": "*",
        "ext_jarque_beta": "D",
    }
    
    # Create test instances once
    tests = create_univariate_tests()
    
    # Initialize statistics DataFrame
    stats = pd.DataFrame(
        index=range(NUM_PROJECTIONS),
        columns=list(tests.keys()),
    )
    
    # Generate projection angles and colors
    As = []
    cmap = plt.get_cmap("coolwarm")
    for theta in np.linspace(0, np.pi, NUM_PROJECTIONS):
        x = (0, np.cos(theta) * PROJECTION_RADIUS)
        y = (0, np.sin(theta) * PROJECTION_RADIUS)
        As.append([x, y, cmap(theta / np.pi)])

    fig, axs = plt.subplots(1, 4, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    sns.histplot(x=X1, color="blue", element="step", ax=axs[0])
    sns.histplot(x=X2, color="red", element="step", ax=axs[0], alpha=0.4)
    axs[0].set_xlabel(r"$x_1$ (blue) | $x_2$ (red)")

    sns.kdeplot(x=X1, y=X2, cmap="magma", fill=True, levels=KDE_LEVELS, ax=axs[1])
    df = pd.DataFrame(columns=["data", "proj"], index=range(NUM_PROJECTIONS * X1.size(0)))
    axs[2].set_yticks(np.arange(len(As)) * PROJECTION_OFFSET)
    axs[3].set_xticks(range(len(As)))
    df = pd.DataFrame(columns=range(NUM_PROJECTIONS))
    
    for i, (x, y, color) in enumerate(As):
        axs[1].arrow(x[0], y[0], x[1], y[1], color=color, width=ARROW_WIDTH)
        rescalor = np.sqrt(x[1] ** 2 + y[1] ** 2)
        proj = (X1 * x[1] + X2 * y[1]) / rescalor
        heights, edges = np.histogram(proj, bins=HISTOGRAM_BINS)
        axs[2].stairs(
            heights + i * PROJECTION_OFFSET,
            edges,
            baseline=i * PROJECTION_OFFSET,
            linewidth=PLOT_LINEWIDTH,
            edgecolor="k",
            facecolor=color,
            zorder=1000 - i,
            fill=True,
        )
        axs[2].axhline(i * PROJECTION_OFFSET, c="k")
        
        # Compute test statistics using pre-instantiated tests
        proj_data = proj.unsqueeze(1)
        stats.loc[i, "cramer_von_mises"] = tests['cramer_von_mises'](proj_data).mean()
        stats.loc[i, "vcreg"] = tests['vcreg'](proj_data).mean()
        stats.loc[i, "watson"] = tests['watson'](proj_data)
        stats.loc[i, "anderson_darling"] = tests['anderson_darling'](proj_data).mean()
        stats.loc[i, "ext_jarque_beta"] = tests['ext_jarque_beta'](proj_data).mean()
        stats.loc[i, "epps_pulley"] = tests['epps_pulley'](proj_data).mean()
        df[i] = proj.tolist()
    stats -= stats.min(0)
    stats /= stats.max(0)
    for i in range(len(As)):
        color = As[i][2]
        for name in stats:
            axs[3].scatter(
                [i],
                [stats.loc[i, name]],
                color=color,
                marker=markers[name],
                edgecolor="k",
                label="l2",
                zorder=100000,
                linewidth=2,
            )
            if i > 0:
                axs[3].plot(
                    [i - 1, i],
                    [stats.loc[i - 1, name], stats.loc[i, name]],
                    c=color,
                    zorder=10,
                )
        axs[3].get_xticklabels()[i].set_color(color)
        axs[2].get_yticklabels()[i].set_color(color)
    axs[3].set_ylabel(r"$\ell_1$ and $\ell_2$")
    axs[3].legend()
    handles = [
        Line2D(
            [0],
            [0],
            label=name,
            marker=markers[name],
            markersize=7,
            markeredgecolor="k",
            markerfacecolor="white",
            linestyle="",
            markeredgewidth=2,
        )
        for name in stats.columns
    ]
    labs = [l.get_label() for l in handles]
    axs[3].legend(
        handles,
        labs,
        loc="upper left",
        ncol=7,
        bbox_to_anchor=(-2.9, 1.12),
        labelspacing=1,
        columnspacing=0.8,
    )
    # extra_axs2.set_ylabel(r"$1-r^2$ (%)")
    axs[1].set_xlim(-3, 3)
    axs[1].set_ylim(-3, 3)
    axs[2].set_yticks([])
    axs[1].set_aspect("equal")
    axs[2].set_aspect(0.009)
    axs[1].set_xlabel(r"$x_1$")
    axs[1].set_ylabel(r"$x_2$")
    axs[2].set_xlabel(r"$\langle x, a_i \rangle$")
    axs[2].set_ylabel(r"$p(\langle x, a_i \rangle)$")
    axs[2].set_yticks(np.arange(len(As)) * PROJECTION_OFFSET, [f"i:{i}" for i in range(len(As))])
    axs[3].set_xticks(np.arange(len(As)), [f"i:{i}" for i in range(len(As))])
    axs[3].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axs[3].set_yticks(axs[3].get_yticks()[::2])
    axs[3].set_ylim(-0.1, 1.1)

    plt.subplots_adjust(0.06, 0.22, 0.97, 0.95, 0.1, 0.1)


Z1 = torch.randn(2000)
Z2 = torch.randn(2000)

X1 = Z1
X2 = Z1.sign() * Z2.abs()
generate_figure(X1, X2)
plt.savefig("2d_slicing_example_1.pdf")
plt.close()
Z1 = torch.randn(2000)
B = (torch.rand(2000) > 0.5).float() * 2 - 1

X1 = Z1
X2 = Z1 * B
generate_figure(X1, X2)
plt.savefig("2d_slicing_example_2.pdf")
plt.close()
