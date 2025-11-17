"""
Teaser Manifold Figure - 3D Swiss Roll and 2D Embedding Visualization
---------------------------------------------------------------------

This script generates a self-contained figure showing:
1. A 3D Swiss roll manifold with density overlay (left/first figure)
2. A 2D Euclidean embedding with projections (right/second figure)

The visualizations are designed to show the relationship between the 3D manifold
structure and its 2D embeddings.

Outputs:
    - 'teaser_manifold_0.png' - 3D Swiss roll with density
    - 'teaser_manifold_1.png' - 2D embedding with projections

Requirements:
    - numpy
    - matplotlib
    - scipy

Usage:
    python teaser_manifold.py

Author: Metamate AI
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm as gnorm
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# 3D MANIFOLD FUNCTIONS
# =============================================================================

def swiss_roll(u, v):
    """
    Map (u, v) in parameter space to (x, y, z) in 3D Swiss roll with undulations.
    """
    x = u * np.cos(u)
    y = v + 2.0 * np.sin(0.5 * u) * np.cos(0.5 * v)  # undulate in y
    z = u * np.sin(u) + 1.5 * np.cos(0.3 * u + 0.2 * v)  # undulate in z
    return x, y, z


def isotropic_gaussian_2d(u, v, mu, sigma):
    """
    2D isotropic Gaussian PDF.
    """
    diff_u = u - mu[0]
    diff_v = v - mu[1]
    exponent = -0.5 * ((diff_u**2 + diff_v**2) / sigma**2)
    return np.exp(exponent)


def build_isotropic_blobs_density(u, v, K=14, epsilon=0.01):
    """
    Build a density as a GMM with K compact, isotropic, circular blobs in normalized (u,v) space.
    """
    # Normalize u, v to [0,1]
    u_norm = (u - u_min) / (u_max - u_min)
    v_norm = (v - v_min) / (v_max - v_min)
    # Sample means in [0.1, 0.9]^2 to avoid edge effects
    means = np.random.uniform(0.1, 0.9, size=(K, 2))
    # Isotropic sigmas in [0.02, 0.05]
    sigmas = np.random.uniform(0.02, 0.05, K)
    # Positive random weights, normalized
    weights = np.random.uniform(0.7, 1.3, K)
    weights /= weights.sum()
    # Build GMM density
    density = np.zeros_like(u)
    for k in range(K):
        density += weights[k] * isotropic_gaussian_2d(
            u_norm, v_norm, means[k], sigmas[k]
        )
    # Add a tiny baseline to avoid full black background
    density += epsilon
    # Normalize to [0, 1] and apply mild gamma for contrast
    density -= density.min()
    density /= density.max()
    density = np.clip(density, 0, 1)
    density = density**0.98  # mild gamma for punchy but natural look
    return density


def render_manifold(X, Y, Z, DENS, filename_base="teaser_manifold_0"):
    """
    Render the manifold with density overlay and save as PNG and SVG.
    """
    # Style
    plt.rcParams.update(
        {
            "figure.figsize": (12, 9),
            "font.size": 18,
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "savefig.dpi": 300,
            "savefig.transparent": True,
            "mathtext.fontset": "stix",
            "axes.edgecolor": "none",
            "axes.facecolor": "none",
            "axes.grid": False,
        }
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Normalize for colormap
    norm = plt.Normalize(vmin=DENS.min(), vmax=DENS.max())
    facecolors = cm.coolwarm(norm(DENS))

    # Surface with explicit facecolors so we truly color by DENS (not Z)
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=1.0,
    )

    # Associate the same norm/cmap to the colorbar via a ScalarMappable
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)
    mappable.set_array(DENS)
    # cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, aspect=30, pad=0.03)
    # cbar.set_label("Density", fontsize=18)
    # cbar.ax.tick_params(labelsize=14)

    # Remove chartjunk
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-12, 12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([np.ptp(X), np.ptp(Y), np.ptp(Z)])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_axis_off()

    # Camera view
    ax.view_init(elev=32, azim=128)

    # Save
    plt.subplots_adjust(0, 0, 1, 1)
    output_path = os.path.join(SCRIPT_DIR, f"{filename_base}.png")
    plt.savefig(output_path, transparent=True, dpi=300)
    # plt.savefig(f'{filename_base}.svg', transparent=True)
    plt.show()


# =============================================================================
# 2D EMBEDDING FUNCTIONS
# =============================================================================

def isotropic_gaussian_2d_euclidean(x, y, mu, sigma):
    """2D isotropic Gaussian PDF for Euclidean space."""
    dx = x - mu[0]
    dy = y - mu[1]
    return np.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))


def sample_gmm_isotropic_in_disk(n, means, sigmas, weights, r_max=2.0):
    """Sample points from GMM within a disk."""
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
    """Generate random means within a disk."""
    angles = np.random.uniform(0, 2 * np.pi, K)
    radii = np.random.uniform(r_min, r_max, K)
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)


def render_2d_embedding():
    """
    Render 2D Euclidean embedding with density, arrows, and projections.
    
    Creates a two-panel figure:
    - Left: 2D density map with directional arrows
    - Right: Projected densities along arrow directions
    """
    # Set reproducibility
    np.random.seed(42)
    
    # Parameters
    Nx, Ny = 512, 512  # grid resolution
    K = 14  # number of Gaussian blobs
    sigma_min, sigma_max = 0.04, 0.10
    epsilon_baseline = 0.01
    n_samples = 20000
    extent = (-2.0, 2.0, -2.0, 2.0)
    
    # Arrow directions and colors
    arrow_thetas = np.array([np.deg2rad(25), np.deg2rad(155), np.deg2rad(285)])
    arrow_colors = ["#b58900", "tab:orange", "tab:green"]
    arrow_labels = ["Arrow A", "Arrow B", "Arrow C"]
    
    # Create mixture parameters
    means = random_means_in_disk(K) * 0.9
    means[-1] += 0.4
    sigmas = np.random.uniform(sigma_min, sigma_max, K)
    weights = np.random.uniform(0.8, 1.2, K)
    weights /= weights.sum()
    
    # Euclidean grid
    x_centers = np.linspace(extent[0], extent[1], Nx)
    y_centers = np.linspace(extent[2], extent[3], Ny)
    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="xy")
    
    # Evaluate density
    DENS = np.zeros_like(Xc)
    for k in range(K):
        DENS += weights[k] * isotropic_gaussian_2d_euclidean(Xc, Yc, means[k], sigmas[k])
    DENS += epsilon_baseline
    
    # Normalize
    DENS_min = DENS.min()
    DENS_max = DENS.max()
    DENS = (DENS - DENS_min) / (DENS_max - DENS_min)
    DENS = DENS**0.5
    
    # Colormap
    cmap = cm.coolwarm
    
    # Sample points for projections
    points = sample_gmm_isotropic_in_disk(n_samples, means, sigmas, weights, r_max=2.0)
    
    # Project onto arrow directions
    arrow_unit_vecs = np.stack([np.cos(arrow_thetas), np.sin(arrow_thetas)], axis=1)
    projections = [points @ uvec for uvec in arrow_unit_vecs]
    
    # Compute smoothed histograms
    bins = np.linspace(-4.0, 4.0, 200)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    hist_densities = []
    for proj in projections:
        h, _ = np.histogram(proj, bins=bins, density=True)
        # Smooth with Gaussian kernel
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
    
    # Left subplot: Euclidean density
    ax_cart = fig.add_subplot(1, 2, 1)
    norm = plt.Normalize(vmin=DENS.min(), vmax=DENS.max())
    img = ax_cart.imshow(
        DENS, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="bilinear"
    )
    ax_cart.set_aspect("equal", adjustable="box")
    
    # Draw unit circle
    circle = plt.Circle(
        (0, 0), 1.0, edgecolor="black", facecolor="none", linewidth=1.0, zorder=2
    )
    # Draw unit circle
    circle = plt.Circle(
        (0, 0), 1.0, edgecolor="black", facecolor="none", linewidth=1.0, zorder=2
    )
    ax_cart.add_patch(circle)
    
    # Draw colorful arrows from center to unit circle
    for th, col, lbl in zip(arrow_thetas, arrow_colors, arrow_labels):
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
    
    # Right subplot: projected densities
    ax_proj = fig.add_subplot(1, 2, 2)
    gauss = gnorm.pdf(bin_centers, loc=0, scale=1)
    for dens, col, lbl, off in zip(hist_densities, arrow_colors, arrow_labels, offsets):
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
    
    # Clean up axes
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
    output_path = os.path.join(SCRIPT_DIR, "teaser_manifold_1.png")
    plt.savefig(output_path, transparent=True)
    # plt.savefig('euclidean_blobs_density_custom.svg', transparent=True)
    # plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Set reproducibility
    np.random.seed(42)
    
    print("Generating 3D manifold visualization...")
    
    # 3D Manifold parameters
    N_u, N_v = 200, 80
    u_min, u_max = 1.5 * np.pi, 4.5 * np.pi
    v_min, v_max = -12, 12
    
    # Generate 3D manifold
    u = np.linspace(u_min, u_max, N_u)
    v = np.linspace(v_min, v_max, N_v)
    U, V = np.meshgrid(u, v, indexing="ij")
    X, Y, Z = swiss_roll(U, V)
    DENS = build_isotropic_blobs_density(U, V, K=14, epsilon=0.01)
    
    # Render 3D manifold
    render_manifold(X, Y, Z, DENS)
    print(f"✓ 3D manifold saved: {os.path.join(SCRIPT_DIR, 'teaser_manifold_0.png')}")
    
    print("\nGenerating 2D embedding visualization...")
    
    # Render 2D embedding
    render_2d_embedding()
    print(f"✓ 2D embedding saved: {os.path.join(SCRIPT_DIR, 'teaser_manifold_1.png')}")
    
    print("\nComplete! Both visualizations generated successfully.")