"""
2D Euclidean Blobby Density with Projections
--------------------------------------------

This script visualizes a blobby isotropic GMM density over the square [-2,2] x [-2,2]
in Euclidean coordinates. It draws three colored arrows (length 1) from the origin,
overlays a unit circle, and shows projected point densities along each arrow direction.

Outputs:
    - 'teaser_manifold_2d.png' (transparent, 300dpi)

Requirements:
    - numpy
    - matplotlib
    - scipy

Usage:
    python teaser_manifold_2d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm as gnorm


# --- Parameters ---
Nx, Ny = 512, 512  # grid resolution for Euclidean density image
K = 14  # number of Gaussian blobs
sigma_min, sigma_max = 0.04, 0.10
epsilon_baseline = 0.01
n_samples = 20000  # samples used for projection histograms
extent = (-2.0, 2.0, -2.0, 2.0)  # image extent in Cartesian coordinates

# Arrow directions (in radians) and colors
arrow_thetas = np.array([np.deg2rad(25), np.deg2rad(155), np.deg2rad(285)])
arrow_colors = ["#b58900", "tab:orange", "tab:green"]  # dark yellow, orange, green
arrow_labels = ["Arrow A", "Arrow B", "Arrow C"]


# --- Helper functions ---
def isotropic_gaussian_2d(x, y, mu, sigma):
    """
    2D isotropic Gaussian PDF.
    
    Args:
        x, y: Coordinate arrays
        mu: Mean [x, y]
        sigma: Standard deviation (scalar)
        
    Returns:
        Gaussian density values
    """
    dx = x - mu[0]
    dy = y - mu[1]
    return np.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))


def sample_gmm_isotropic_in_disk(n, means, sigmas, weights, r_max=2.0):
    """
    Sample points from GMM, keeping only those within radius r_max.
    
    Args:
        n: Number of samples to generate
        means: Array of means (K, 2)
        sigmas: Array of standard deviations (K,)
        weights: Array of mixture weights (K,)
        r_max: Maximum radius to keep
        
    Returns:
        Array of sampled points (n_inside, 2)
    """
    K = len(means)
    comp_idx = np.random.choice(K, size=n, p=weights)
    pts = np.zeros((n, 2))
    for k in range(K):
        mask = comp_idx == k
        m = mask.sum()
        if m == 0:
            continue
        pts[mask] = means[k] + sigmas[k] * np.random.randn(m, 2)
    radii = np.linalg.norm(pts, axis=1)
    inside = radii <= r_max
    return pts[inside]


def random_means_in_disk(K, r_min=0.3, r_max=1.6):
    """
    Generate random means uniformly distributed in annular region.
    
    Args:
        K: Number of means
        r_min: Minimum radius
        r_max: Maximum radius
        
    Returns:
        Array of means (K, 2)
    """
    angles = np.random.uniform(0, 2 * np.pi, K)
    radii = np.random.uniform(r_min, r_max, K)
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)


def render_euclidean_density():
    """
    Render 2D Euclidean density with projections and save figure.
    """
    # Create mixture parameters in extended disk (Cartesian)
    means = random_means_in_disk(K) * 0.9
    means[-1] += 0.4
    sigmas = np.random.uniform(sigma_min, sigma_max, K)
    weights = np.random.uniform(0.8, 1.2, K)
    weights /= weights.sum()

    # Euclidean grid (cell centers)
    x_centers = np.linspace(extent[0], extent[1], Nx)
    y_centers = np.linspace(extent[2], extent[3], Ny)
    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="xy")

    # Evaluate density at centers
    DENS = np.zeros_like(Xc)
    for k in range(K):
        DENS += weights[k] * isotropic_gaussian_2d(Xc, Yc, means[k], sigmas[k])
    DENS += epsilon_baseline

    # Normalize and mild gamma
    DENS_min = DENS.min()
    DENS_max = DENS.max()
    DENS = (DENS - DENS_min) / (DENS_max - DENS_min)
    DENS = DENS**0.5

    # Prepare colormap
    cmap = cm.coolwarm

    # Sample points for projection histograms
    points = sample_gmm_isotropic_in_disk(n_samples, means, sigmas, weights, r_max=2.0)

    # Projections onto the arrow unit vectors
    arrow_unit_vecs = np.stack([np.cos(arrow_thetas), np.sin(arrow_thetas)], axis=1)
    projections = [points @ uvec for uvec in arrow_unit_vecs]

    # Smoothed histogram densities with vertical offsets
    bins = np.linspace(-4.0, 4.0, 200)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    hist_densities = []
    for proj in projections:
        h, _ = np.histogram(proj, bins=bins, density=True)
        # Smooth via simple Gaussian-like kernel convolution
        kernel = np.exp(-0.5 * (np.linspace(-2, 2, 21) ** 2))
        kernel /= kernel.sum()
        h_smooth = np.convolve(h, kernel, mode="same")
        hist_densities.append(h_smooth)

    offsets = np.array([0.0, 0.8, 1.6])

    # Rendering
    plt.rcParams.update(
        {
            "figure.figsize": (14, 6),
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "savefig.dpi": 300,
            "savefig.transparent": True,
            "mathtext.fontset": "stix",
        }
    )

    fig = plt.figure(figsize=(12, 6))

    # Left subplot: Euclidean density with arrows and unit circle
    ax_cart = fig.add_subplot(1, 2, 1)
    norm = plt.Normalize(vmin=DENS.min(), vmax=DENS.max())
    ax_cart.imshow(
        DENS, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="bilinear"
    )
    ax_cart.set_aspect("equal", adjustable="box")

    # Draw unit circle (thin black line)
    circle = plt.Circle(
        (0, 0), 1.0, edgecolor="black", facecolor="none", linewidth=1.0, zorder=2
    )
    ax_cart.add_patch(circle)

    # Draw colorful arrows from center to unit circle
    for th, col in zip(arrow_thetas, arrow_colors):
        end = (np.cos(th), np.sin(th))
        ax_cart.annotate(
            "",
            xy=end,
            xytext=(0.0, 0.0),
            arrowprops=dict(
                arrowstyle="-|>", color=col, lw=7.0, mutation_scale=50
            ),
        )

    # Show axes crossing at origin
    for spine in ["left", "bottom"]:
        ax_cart.spines[spine].set_position("zero")
    for spine in ["top", "right"]:
        ax_cart.spines[spine].set_visible(False)
    ax_cart.set_xticks([])
    ax_cart.set_yticks([])
    ax_cart.tick_params(axis="both", which="both", length=5, size=26)
    ax_cart.set_xlim(-1.8, 1.8)
    ax_cart.set_ylim(-1.8, 1.8)
    ax_cart.set_title("Embedding distribution", fontsize=30)

    # Right subplot: projected densities along arrows
    ax_proj = fig.add_subplot(1, 2, 2)
    gauss = gnorm.pdf(bin_centers, loc=0, scale=1)
    for dens, col, off in zip(hist_densities, arrow_colors, offsets):
        ax_proj.plot(bin_centers, dens + off, color=col, linewidth=2.5, label=None)
        ax_proj.plot(bin_centers, gauss + off, color="k", linewidth=1.5, label=None)
        ax_proj.fill_between(
            bin_centers,
            gauss + off,
            dens + off,
            color="gray",
            alpha=0.3,
            hatch="/",
            label="error" if off == offsets[0] else None,
        )

    ax_proj.set_title("Projected point densities", fontsize=30)
    ax_proj.set_xlabel("Projection coordinate (along direction)", fontsize=26)

    # Clean up spines and ticks
    ax_proj.xaxis.set_ticks_position("bottom")
    ax_proj.spines["bottom"].set_position(("outward", 0))
    ax_proj.spines["top"].set_visible(False)
    ax_proj.spines["right"].set_visible(False)
    ax_proj.spines["left"].set_visible(False)
    ax_proj.set_yticks([])
    ax_proj.set_xticks([-4, -2, 0, 2, 4], labels=[-4, -2, 0, 2, 4], fontsize=20)
    ax_proj.set_xlim(-4.0, 4.0)
    ymax = offsets[-1] + max(h.max() for h in hist_densities) + 0.2
    ax_proj.set_ylim(-0.1, ymax)
    ax_proj.legend(fontsize=18)

    plt.subplots_adjust(0.0, 0.12, 0.98, 0.92, 0.05, 0.05)
    plt.savefig("teaser_manifold_2d.png", transparent=True)
    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Render and save
    render_euclidean_density()
    print("✓ 2D Euclidean density visualization saved as 'teaser_manifold_2d.png'")
