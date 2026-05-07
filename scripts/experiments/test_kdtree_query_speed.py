import time

import numpy as np
from scipy.spatial import KDTree

# -----------------------------
# Parameters
# -----------------------------
N_PHOTONS = 200_000  # photon map size
N_SAMPLES = 10_000  # number of shading points
K = 1000
DIM = 3
DTYPE = np.float32

# -----------------------------
# Generate fake data
# -----------------------------
rng = np.random.default_rng(0)

photon_positions = rng.random((N_PHOTONS, DIM)).astype(DTYPE)
sample_positions = rng.random((N_SAMPLES, DIM)).astype(DTYPE)

tree = KDTree(photon_positions)

# Warm-up (important for fair timing)
tree.query(sample_positions[:10], k=K)

# -----------------------------
# 1) Looped query
# -----------------------------
t0 = time.perf_counter()

for pos in sample_positions:
    dist, idx = tree.query(pos, k=K)

t1 = time.perf_counter()
loop_time = t1 - t0

print(f"Looped query time: {loop_time:.3f} s")

# -----------------------------
# 2) Batched query
# -----------------------------
t0 = time.perf_counter()

dist_batch, idx_batch = tree.query(sample_positions, k=K)

t1 = time.perf_counter()
batch_time = t1 - t0

print(f"Batched query time: {batch_time:.3f} s")

# -----------------------------
# Speedup
# -----------------------------
print(f"Speedup: {loop_time / batch_time:.2f}×")
