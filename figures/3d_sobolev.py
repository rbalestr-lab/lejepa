import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import sph_harm

# Grid resolution parameters
N_PHI = 400  # Latitude resolution
N_THETA = 700  # Longitude resolution
SPHERE_RADIUS = 1.0

# Sobolev smoothness parameters
SOBOLEV_ALPHAS = [1, 10, 100]  # Lower = more detail, higher = smoother

# Visualization parameters
FIGURE_SIZE = (5, 5.3)
TITLE_FONTSIZE = 28
OUTPUT_DPI = 300
AXIS_LIMITS = [-0.65, 0.65]
CAMERA_DISTANCE = 1

# Spherical harmonic parameters
MAX_DEGREE_CLIP = (2, 60)  # Min and max degree for spherical harmonics
DEGREE_ALPHA_EXPONENT = 0.7  # Controls alpha-to-degree mapping

# Create meshgrid
phi = np.linspace(0, np.pi, N_PHI)
theta = np.linspace(0, 2 * np.pi, N_THETA)
phi, theta = np.meshgrid(phi, theta)
x = SPHERE_RADIUS * np.sin(phi) * np.cos(theta)
y = SPHERE_RADIUS * np.sin(phi) * np.sin(theta)
z = SPHERE_RADIUS * np.cos(phi)


def random_spherical_harmonic_density(phi, theta, max_degree):
    density = np.zeros_like(phi)
    for l in range(max_degree + 1):
        for m in range(-l, l + 1):
            coeff = np.random.randn() + 1j * np.random.randn()
            # No Sobolev weighting here: sharper difference
            density += np.real(coeff * sph_harm(m, l, theta, phi))
    # Nonlinear transformation for contrast
    density = np.abs(density)
    density = np.log1p(density)  # log scale for more contrast
    density -= density.min()
    density /= density.max()
    return density


# Map alpha to max_degree more aggressively
def alpha_to_degree(alpha):
    # Lower alpha = higher degree (more chaos), higher alpha = lower degree (smoother)
    return int(np.clip(60 / (alpha**DEGREE_ALPHA_EXPONENT), *MAX_DEGREE_CLIP))


for i, alpha in enumerate(SOBOLEV_ALPHAS):
    max_degree = alpha_to_degree(alpha)
    density = random_spherical_harmonic_density(phi, theta, max_degree)
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    colors = cm.inferno(density)
    surf = ax.plot_surface(
        x,
        y,
        z,
        facecolors=colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=True,
    )
    ax.set_title(r"Sobolev $\alpha$=" + str(alpha), fontsize=TITLE_FONTSIZE)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.set_facecolor("white")
    ax.set_xlim(AXIS_LIMITS)
    ax.set_ylim(AXIS_LIMITS)
    ax.set_zlim(AXIS_LIMITS)
    ax.dist = CAMERA_DISTANCE
    plt.subplots_adjust(0, 0, 1, 0.94)
    plt.savefig(f"3d_sphere_{i}.png", dpi=OUTPUT_DPI)
    plt.close()
