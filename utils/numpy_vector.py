import numpy as np
from numpy.typing import NDArray
from typing import Union, List

# Type aliases
Vector = Union[List[float], NDArray[np.float32]]         # Single vector
Matrix = Union[List[List[float]], NDArray[np.float32]]   # Matrix of vectors (N, 3)

# -----------------------------
# Vector utilities
# -----------------------------

def to_matrix(vectors: List[Vector]) -> NDArray[np.float32]:
    """Convert list of vectors to a (N, 3) NumPy matrix."""
    return np.array(vectors, dtype=np.float32)

def from_matrix(matrix: NDArray[np.float32]) -> List[List[float]]:
    """Convert a (N, 3) matrix to a list of vectors."""
    return matrix.tolist()

def is_batch(v: Union[Vector, Matrix]) -> bool:
    """Check if input is a batch of vectors (2D array)."""
    return isinstance(v, np.ndarray) and v.ndim == 2

def ensure_array(v: Union[Vector, Matrix]) -> NDArray[np.float32]:
    """Ensure the input is a NumPy array of shape (N, 3) or (3,)."""
    return np.atleast_2d(np.array(v, dtype=np.float32))

# -----------------------------
# Vector math functions
# -----------------------------

def add(a: Union[Vector, Matrix], b: Union[Vector, Matrix]) -> NDArray[np.float32]:
    return np.array(a, dtype=np.float32) + np.array(b, dtype=np.float32)

def subtract(a: Union[Vector, Matrix], b: Union[Vector, Matrix]) -> NDArray[np.float32]:
    return np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)

def scalar_mul(v: Union[Vector, Matrix], scalar: float) -> NDArray[np.float32]:
    return np.array(v, dtype=np.float32) * scalar

def length(v: Union[Vector, Matrix]) -> Union[float, NDArray[np.float32]]:
    v_arr = ensure_array(v)
    axis = 1 if is_batch(v) else 0
    return np.linalg.norm(v_arr, axis=axis)

def normalize_vector(v: Vector) -> Vector:
    """Normalize a single 3D vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def normalize_batch(v_batch: Matrix) -> Matrix:
    """Normalize a batch of 3D vectors (shape: (N, 3))."""
    norms = np.linalg.norm(v_batch, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v_batch / norms


def dot_vector(a: Vector, b: Vector) -> float:
    return float(np.dot(a, b))

def dot_batch(a: Matrix, b: Matrix) -> NDArray[np.float32]:
    return np.sum(a * b, axis=1)

def cross(a: Union[Vector, Matrix], b: Union[Vector, Matrix]) -> NDArray[np.float32]:
    return np.cross(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32))

def reflect_vector(ray: Vector, normal: Vector) -> Vector:
    """Reflect a single ray around a normal."""
    ray = np.asarray(ray, dtype=np.float32)
    normal = normalize_vector(normal)
    return ray - 2 * dot_vector(ray, normal) * normal

def reflect_batch(rays: Matrix, normals: Matrix) -> Matrix:
    """Reflect a batch of rays around corresponding normals."""
    rays = np.asarray(rays, dtype=np.float32)
    normals = normalize_batch(normals)
    dot_products = dot_batch(rays, normals)[:, np.newaxis]  # shape (N, 1)
    return rays - 2 * dot_products * normals

def sample_directions_in_cone(
    laser_dir: NDArray[np.float32],
    divergence_angle: float,
    num_samples: int,
    rng: np.random.Generator
) -> NDArray[np.float32]:
    """
    Generate num_samples unit direction vectors within a cone defined by laser_dir and divergence_angle (in radians).
    """
    # Ensure laser_dir is normalized
    laser_dir = normalize_vector(laser_dir)

    # https://math.stackexchange.com/a/205589
    # Create random directions in local cone coordinates
    z = rng.uniform(np.cos(divergence_angle), 1, size=num_samples)
    sin_theta = np.sqrt(1 - z**2)
    phi = rng.uniform(0, 2 * np.pi, size=num_samples)

    # Local direction vectors (cone along +Z)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    local_dirs = np.stack([x, y, z], axis=1)  # shape (N, 3)

    # Build rotation matrix from [0, 0, 1] to laser_dir
    def rotation_matrix_from_vectors(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
        """Returns rotation matrix that rotates vector a to vector b."""
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        v = np.cross(a, b)
        
        c = np.dot(a, b)
        if np.isclose(c, 1.0):
            return np.eye(3)
        elif np.isclose(c, -1.0):
            # 180-degree rotation
            orthogonal = np.array([1, 0, 0]) if not np.allclose(a, [1, 0, 0]) else np.array([0, 1, 0])
            v = np.cross(a, orthogonal)
            v /= np.linalg.norm(v)
            H = np.eye(3) - 2 * np.outer(v, v)
            return H
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
        R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
        return R

    R = rotation_matrix_from_vectors(np.array([0, 0, 1], dtype=np.float32), laser_dir.astype(np.float32))
    directions = local_dirs @ R.T  # Rotate each local direction to align with laser_dir

    return directions.astype(np.float32)


def random_unit_vector(rng: np.random.Generator) -> Vector:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)

def random_unit_vector_batch(rng: np.random.Generator, n: int) -> Matrix:
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v.astype(np.float32)

# -----------------------------
# Reflection direction sampling functions
# -----------------------------
def heuristic_sample(normal: Vector, rng: np.random.Generator) -> Vector:
    return normalize_vector(normal + random_unit_vector(rng))

def heuristic_sample_batch(normals: Matrix, rng: np.random.Generator) -> Matrix:
    return normalize_batch(normals + random_unit_vector_batch(rng, normals.shape[0]))

def uniform_hemisphere_sample(normal: Vector, rng: np.random.Generator) -> Vector:
    while True:
        direction = random_unit_vector(rng)
        if np.dot(direction, normal) > 0:
            return direction

def uniform_hemisphere_sample_batch(normals: Matrix, rng: np.random.Generator) -> Matrix:
    n = normals.shape[0]
    dirs = random_unit_vector_batch(rng, n)

    # Dot product to check which side each vector is on
    # Einstein summation does not build a temporary array, and is much more memory optimized
    # https://stackoverflow.com/a/33641428/8990620
    dots = np.einsum('ij,ij->i', dirs, normals)

    # Flip directions that are in the wrong hemisphere
    dirs[dots < 0] *= -1

    return normalize_batch(dirs)

def cosine_weighted_sample(normal: Vector, rng: np.random.Generator) -> Vector:
    u1 = rng.random()
    u2 = rng.random()
    r = np.sqrt(u1)
    theta = 2 * np.pi * u2

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.sqrt(1 - u1)
    local = np.array([x, y, z], dtype=np.float32)

    up = np.array([0, 0, 1], dtype=np.float32)
    if np.abs(normal[2]) > 0.999:
        up = np.array([1, 0, 0], dtype=np.float32)

    tangent = np.cross(up, normal)
    tangent = normalize_vector(tangent)
    bitangent = np.cross(normal, tangent)

    return normalize_vector(
        tangent * local[0] + bitangent * local[1] + normal * local[2]
    )

def cosine_weighted_sample_batch(normals: Matrix, rng: np.random.Generator) -> Matrix:
    n = normals.shape[0]
    u1 = rng.random(n)
    u2 = rng.random(n)
    r = np.sqrt(u1)
    theta = 2 * np.pi * u2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.sqrt(1 - u1)
    local_dirs = np.stack((x, y, z), axis=1)

    results = np.empty_like(local_dirs)
    up = np.array([0, 0, 1], dtype=np.float32)
    for i in range(n):
        normal = normals[i]
        if np.abs(normal[2]) > 0.999:
            t_up = np.array([1, 0, 0], dtype=np.float32)
        else:
            t_up = up
        tangent = normalize_vector(np.cross(t_up, normal))
        bitangent = np.cross(normal, tangent)
        local = local_dirs[i]
        world = (
            tangent * local[0] +
            bitangent * local[1] +
            normal * local[2]
        )
        results[i] = normalize_vector(world)
    return results

