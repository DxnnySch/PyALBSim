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


def dot(a: Union[Vector, Matrix], b: Union[Vector, Matrix]) -> Union[float, NDArray[np.float32]]:
    a_arr = ensure_array(a)
    b_arr = ensure_array(b)
    axis = 1 if is_batch(a) else 0
    return np.sum(a_arr * b_arr, axis=axis)

def cross(a: Union[Vector, Matrix], b: Union[Vector, Matrix]) -> NDArray[np.float32]:
    return np.cross(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32))

def reflect(ray: NDArray[np.float32], normal: NDArray[np.float32]) -> NDArray[np.float32]:
    """Reflect a ray around a normal."""
    ray = np.array(ray, dtype=np.float32)
    normal = normalize_batch(normal) if is_batch(ray) else normalize_vector(normal)
    return ray - 2 * dot(ray, normal)[:, None] * normal if is_batch(ray) else ray - 2 * dot(ray, normal) * normal

def sample_directions_in_cone(
    laser_dir: NDArray[np.float32],
    divergence_angle: float,
    N: int
) -> NDArray[np.float32]:
    """
    Generate N unit direction vectors within a cone defined by laser_dir and divergence_angle (in radians).
    """
    # Ensure laser_dir is normalized
    print(laser_dir)
    laser_dir = normalize_vector(laser_dir)
    print(laser_dir)

    # https://math.stackexchange.com/a/205589
    # Create random directions in local cone coordinates
    z = np.random.uniform(np.cos(divergence_angle), 1, size=N)
    sin_theta = np.sqrt(1 - z**2)
    phi = np.random.uniform(0, 2 * np.pi, size=N)

    # Local direction vectors (cone along +Z)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = z
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
