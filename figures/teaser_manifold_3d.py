"""
3D Swiss Roll Manifold with Localized Isotropic Blobs
-----------------------------------------------------

This script generates a high-quality 3D Swiss roll manifold with a density overlay
composed of highly localized, isotropic Gaussian blobs. The density is constructed
in normalized parameter space to ensure circular blobs, and the 'coolwarm' colormap
is used for visualization.

Outputs:
    - 'teaser_manifold_3d.png' (transparent, 300dpi)

Requirements:
    - numpy
    - matplotlib

Usage:
    python teaser_manifold_3d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# --- Manifold: Swiss Roll with Undulations ---
def swiss_roll(u, v):
    """
    Map (u, v) in parameter space to (x, y, z) in 3D Swiss roll with undulations.
    
    Args:
        u: Parameter in u direction
        v: Parameter in v direction
        
    Returns:
        x, y, z: 3D coordinates
    """
    x = u * np.cos(u)
    y = v + 2.0 * np.sin(0.5 * u) * np.cos(0.5 * v)  # undulate in y
    z = u * np.sin(u) + 1.5 * np.cos(0.3 * u + 0.2 * v)  # undulate in z
    return x, y, z


def isotropic_gaussian_2d(u, v, mu, sigma):
    """
    2D isotropic Gaussian PDF.
    
    Args:
        u, v: Coordinate arrays
        mu: Mean [u, v]
        sigma: Standard deviation (scalar)
        
    Returns:
        Gaussian density values
    """
    diff_u = u - mu[0]
    diff_v = v - mu[1]
    exponent = -0.5 * ((diff_u**2 + diff_v**2) / sigma**2)
    return np.exp(exponent)


def build_isotropic_blobs_density(u, v, K=14, epsilon=0.01):
    """
    Build a density as a GMM with K compact, isotropic, circular blobs in normalized (u,v) space.
    
    Args:
        u, v: Parameter coordinate arrays
        K: Number of Gaussian blobs
        epsilon: Baseline offset to avoid zero density
        
    Returns:
        Normalized density array
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


def render_manifold(X, Y, Z, DENS, filename_base="teaser_manifold_3d"):
    """
    Render the manifold with density overlay and save as PNG.
    
    Args:
        X, Y, Z: 3D coordinate arrays
        DENS: Density values for coloring
        filename_base: Output filename (without extension)
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

    # Surface with explicit facecolors (colored by DENS, not Z)
    ax.plot_surface(
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

    # Remove axes and ticks
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
    plt.savefig(f"{filename_base}.png", transparent=True, dpi=300)
    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameter ranges for Swiss roll
    N_u, N_v = 200, 80
    u_min, u_max = 1.5 * np.pi, 4.5 * np.pi
    v_min, v_max = -12, 12
    
    # Generate parameter grid
    u = np.linspace(u_min, u_max, N_u)
    v = np.linspace(v_min, v_max, N_v)
    U, V = np.meshgrid(u, v, indexing='ij')
    
    # Build manifold
    X, Y, Z = swiss_roll(U, V)
    
    # Build density overlay
    DENS = build_isotropic_blobs_density(U, V, K=14, epsilon=0.01)
    
    # Render and save
    render_manifold(X, Y, Z, DENS)
    print("✓ 3D Swiss roll visualization saved as 'teaser_manifold_3d.png'")
