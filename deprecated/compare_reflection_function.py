import numpy as np
import matplotlib.pyplot as plt
import utils.numpy_vector as np_vec
import time

def plot_2d_crosssection(sample_func, label: str, n_samples: int = 1000):
    rng = np.random.default_rng(42)
    normal = np.array([0, 1, 0], dtype=np.float32)  # vertical normal
    normals = np.tile(normal, (n_samples, 1))
    directions = sample_func(normals, rng)

    # Project directions onto 2D X-Y plane
    x = directions[:, 0]
    y = directions[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=2, alpha=0.6)
    plt.quiver(0, 0, normal[0], normal[1], color='red', scale=5)
    plt.axhline(0, color='gray', linestyle='--')  # reflection plane
    plt.title(f"{label} - 2D Cross-Section (X-Y)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# 2D visualizations
plot_2d_crosssection(
    np_vec.heuristic_sample_batch,
    "Heuristic (normal + random)"
)

plot_2d_crosssection(
    np_vec.uniform_hemisphere_sample_batch,
    "Uniform Hemisphere"
)

plot_2d_crosssection(
    np_vec.cosine_weighted_sample_batch,
    "Cosine-Weighted Hemisphere (Lambertian)"
)

def plot_angle_distribution_with_pdf(sample_func, label: str, pdf_type: str, n_samples: int = 50_000):
    rng = np.random.default_rng(42)
    normal = np.array([0, 1, 0], dtype=np.float32)
    normals = np.tile(normal, (n_samples, 1))
    start_time = time.time()
    directions = sample_func(normals, rng)
    print(f"sampled {n_samples} points from {label} distribution in {time.time() - start_time:.6f} seconds")
    # cos_theta = np_vec.dot_batch(directions, normals)
    cos_theta = directions[:,1]
    angles = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * (180 / np.pi)
    print(angles)

    # Histogram
    bins = np.linspace(0, 90, 90)
    _, bin_edges = np.histogram(angles, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.figure(figsize=(7, 4))
    plt.hist(angles, bins=60, density=True, alpha=0.75, color='skyblue', edgecolor='black', label="Sampled")

    # Overlay theoretical PDF
    theta_rad = np.radians(bin_centers)
    if pdf_type == "uniform":
        pdf = np.sin(theta_rad)
    elif pdf_type == "cosine":
        pdf = np.sin(theta_rad) * np.cos(theta_rad)
    else:
        pdf = np.ones_like(theta_rad)
    pdf /= np.trapezoid(pdf)  # Normalize PDF

    # Rescale for matching histogram height
    plt.plot(bin_centers, pdf, label="Expected PDF", color="red", linestyle="--")

    plt.title(f"Angle Distribution vs PDF — {label}")
    plt.xlabel("Angle to Normal (degrees)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot comparisons
plot_angle_distribution_with_pdf(np_vec.heuristic_sample_batch, "Heuristic (Normal + Random)", pdf_type="cosine")
plot_angle_distribution_with_pdf(np_vec.uniform_hemisphere_sample_batch, "Uniform Hemisphere", pdf_type="uniform")
plot_angle_distribution_with_pdf(np_vec.cosine_weighted_sample_batch, "Cosine-Weighted Hemisphere", pdf_type="cosine")