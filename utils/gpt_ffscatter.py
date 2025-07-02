import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from line_profiler import profile

@profile
def generate_ff_phase_function(n_ff=1.1, M=18000):
    """
    Generate the Fournier-Forand phase function with given parameters.
    """
    # Scattering angle from 0 to pi
    theta = np.linspace(0, np.pi, M)
    mu = np.cos(theta)  # Cosine of scattering angle

    # Simple Fournier-Forand phase function approximation (single parameter version)
    # p(theta) = 1 / (1 + (n_ff * theta)^2)^(1.5)
    ff_phase = 1 / (1 + (n_ff * theta)**2)**1.5

    # Normalize so integral over sphere is 1
    sin_theta = np.sin(theta)
    normalization = np.trapezoid(ff_phase * sin_theta * 2 * np.pi, theta)
    ff_phase /= normalization

    return theta, ff_phase

def generate_parametric_ff_phase_function(n=1.05, mu=3.5, M=18000):
    """
    Generate the normalized Fournier-Forand phase function.
    
    Parameters:
        n (float): Refractive index of particles (e.g., 1.05–1.20)
        mu (float): Junge slope (typical: 3–5)
        M (int): Number of angular samples
    
    Returns:
        theta (ndarray): Scattering angles [radians]
        p_theta (ndarray): Normalized phase function p(theta)
    """
    theta = np.linspace(0, np.pi, M)
    cos_theta = np.cos(theta)

    # Compute delta and v from n and mu
    delta = 1 / (1 + 0.5 * (n - 1)**2)
    v = (3 - mu) / 2

    # FF phase function (simplified normalized form)
    p_theta = (1 - delta**2) / (4 * np.pi * (1 + delta**2 - 2 * delta * cos_theta)**v)

    # Optional: Normalize numerically to correct any discretization error
    sin_theta = np.sin(theta)
    integral = np.trapezoid(p_theta * sin_theta * 2 * np.pi, theta)
    p_theta /= integral

    return theta, p_theta


def plot_phase_function(theta, ff_phase):
    """
    Visualize the phase function on a semilog scale.
    """
    plt.figure(figsize=(8, 4))
    plt.semilogy(np.degrees(theta), ff_phase)
    plt.xlabel('Scattering Angle (degrees)')
    plt.ylabel('Phase Function p(theta)')
    plt.title('Fournier-Forand Phase Function')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

@profile
def sample_scattering_direction(ff_phase, theta):
    """
    Sample a new scattering direction from the phase function.
    """
    cdf = np.cumsum(ff_phase * np.sin(theta))
    cdf /= cdf[-1]
    inverse_cdf = interp1d(cdf, theta, kind='linear', bounds_error=False, fill_value=(theta[0], theta[-1]))
    
    rnd = np.random.rand()
    scatter_angle = inverse_cdf(rnd)

    # Generate random azimuthal angle
    phi = np.random.uniform(0, 2 * np.pi)
    return scatter_angle, phi

def spherical_to_cartesian(theta, phi):
    """
    Convert spherical angles to cartesian vector.
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

@profile
def scatter_energy(ff_phase, theta, a_dir, b_dir):
    """
    Given two directions (unit vectors), compute scattering probability.
    """
    # Compute angle between a_dir and b_dir
    cos_theta = np.clip(np.dot(a_dir, b_dir), -1.0, 1.0)
    scatter_angle = np.arccos(cos_theta)

    # Interpolate the phase function
    phase_interp = interp1d(theta, ff_phase, kind='linear', bounds_error=False, fill_value=0)
    return phase_interp(scatter_angle)

# def sample_scattering_directions_batch(ff_phase, theta, incoming_dirs):
#     """
#     Vectorized batch sampling of scattering directions for arbitrary incoming directions.
    
#     Parameters:
#         ff_phase: Phase function values
#         theta: Angle array (same length as ff_phase)
#         incoming_dirs: (N, 3) array of unit vectors representing incoming photon directions
        
#     Returns:
#         directions: (N, 3) numpy array of scattered unit direction vectors
#     """
#     N = incoming_dirs.shape[0]

#     # CDF for inverse transform sampling
#     cdf = np.cumsum(ff_phase * np.sin(theta))
#     cdf /= cdf[-1]
#     inverse_cdf = interp1d(cdf, theta, kind='linear', bounds_error=False, fill_value=(theta[0], theta[-1]))

#     # Sample scattering angles
#     random_vals = np.random.rand(N)
#     scatter_theta = inverse_cdf(random_vals)
#     scatter_phi = np.random.uniform(0, 2 * np.pi, size=N)

#     # Local frame scattering vectors (z-aligned frame)
#     x_local = np.sin(scatter_theta) * np.cos(scatter_phi)
#     y_local = np.sin(scatter_theta) * np.sin(scatter_phi)
#     z_local = np.cos(scatter_theta)
#     local_dirs = np.stack((x_local, y_local, z_local), axis=1)

#     # Rotate local_dir into incoming_dir frame
#     def compute_rotation_matrix(v):
#         """Construct orthonormal basis (u, v, w) where w = v (z'), u is arbitrary orthogonal."""
#         w = v / np.linalg.norm(v)
#         up = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
#         u = np.cross(up, w)
#         u /= np.linalg.norm(u)
#         v_ = np.cross(w, u)
#         return np.stack((u, v_, w), axis=1)  # 3x3 rotation matrix

#     # Build rotation matrices per photon
#     rotation_matrices = np.array([compute_rotation_matrix(v) for v in incoming_dirs])  # (N, 3, 3)

#     # Apply rotations: batched matrix-vector multiplication
#     directions = np.einsum('nij,nj->ni', rotation_matrices, local_dirs)
#     return directions

@profile
def sample_scattering_directions_batch(ff_phase, theta, incoming_dirs, rng: np.random.Generator):
    """
    Vectorized batch sampling of scattering directions for arbitrary incoming directions.
    
    Parameters:
        ff_phase: Phase function values
        theta: Angle array (same length as ff_phase)
        incoming_dirs: (N, 3) array of unit vectors representing incoming photon directions
        
    Returns:
        directions: (N, 3) numpy array of scattered unit direction vectors
    """
    N = incoming_dirs.shape[0]

    # CDF for inverse transform sampling
    cdf = np.cumsum(ff_phase * np.sin(theta))
    cdf /= cdf[-1]
    inverse_cdf = interp1d(cdf, theta, kind='linear', bounds_error=False, fill_value=(theta[0], theta[-1]))

    # Sample scattering angles
    random_vals = rng.random(N)
    scatter_theta = inverse_cdf(random_vals)
    scatter_phi = rng.uniform(0, 2 * np.pi, size=N)

    # Local frame scattering vectors (z-aligned frame)
    x_local = np.sin(scatter_theta) * np.cos(scatter_phi)
    y_local = np.sin(scatter_theta) * np.sin(scatter_phi)
    z_local = np.cos(scatter_theta)
    local_dirs = np.stack((x_local, y_local, z_local), axis=1)

    # Build rotation matrices per photon
    # rotation_matrices = np.array([compute_rotation_matrix(v) for v in incoming_dirs])  # (N, 3, 3)
    rotation_matrices = build_rotation_matrices(incoming_dirs)

    # Apply rotations: batched matrix-vector multiplication
    directions = np.einsum('nij,nj->ni', rotation_matrices, local_dirs)
    return directions

# Rotate local_dir into incoming_dir frame
@profile
def compute_rotation_matrix(v):
    """Construct orthonormal basis (u, v, w) where w = v (z'), u is arbitrary orthogonal."""
    w = v / np.linalg.norm(v)
    up = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    u = np.cross(up, w)
    u /= np.linalg.norm(u)
    v_ = np.cross(w, u)
    return np.stack((u, v_, w), axis=1)  # 3x3 rotation matrix

@profile
def build_rotation_matrices(incoming_dirs):
    """
    Vectorized construction of rotation matrices to align local +z with incoming_dirs.
    
    Parameters:
        incoming_dirs: (N, 3) array of incoming unit direction vectors.
        
    Returns:
        rotation_matrices: (N, 3, 3) array of rotation matrices.
    """
    N = incoming_dirs.shape[0]
    w = incoming_dirs / np.linalg.norm(incoming_dirs, axis=1, keepdims=True)  # (N, 3)

    # Choose "up" vector adaptively to avoid numerical instability
    use_z_up = np.abs(w[:, 2]) < 0.99
    up = np.zeros_like(w)
    up[use_z_up] = np.array([0, 0, 1])
    up[~use_z_up] = np.array([1, 0, 0])

    u = np.cross(up, w)
    u /= np.linalg.norm(u, axis=1, keepdims=True)

    v = np.cross(w, u)

    # Stack u, v, w as rotation matrix columns
    rotation_matrices = np.stack((u, v, w), axis=2)  # shape (N, 3, 3)
    return rotation_matrices


# Example usage
if __name__ == "__main__":
    n_ff = 1.1
    M = 18000

    theta, ff_phase = generate_ff_phase_function(n_ff=n_ff, M=M)
    cdf = np.cumsum(ff_phase * np.sin(theta))
    cdf /= cdf[-1]
    theta2, parametric_ff_phase = generate_parametric_ff_phase_function(n_ff, 3.5, M)
    # plot_phase_function(theta, ff_phase)
    plt.figure(figsize=(8, 4))
    plt.semilogx(np.degrees(theta), cdf)
    plt.xlabel('Scattering Angle (degrees)')
    plt.ylabel('Phase Function p(theta)')
    plt.title('Fournier-Forand Phase Function')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Sample a scattering direction
    # scatter_theta, scatter_phi = sample_scattering_direction(ff_phase, theta)
    # scatter_dir = spherical_to_cartesian(scatter_theta, scatter_phi)

    # # Compute energy between two directions
    # a_dir = np.array([0, 0, 1])  # forward
    # b_dir = scatter_dir         # sampled
    # energy = scatter_energy(ff_phase, theta, a_dir, b_dir)

    # print(f"Sampled scattering direction: theta = {np.degrees(scatter_theta):.2f}°, phi = {np.degrees(scatter_phi):.2f}°")
    # print(f"Scattering energy (probability density): {energy:.4e}")
