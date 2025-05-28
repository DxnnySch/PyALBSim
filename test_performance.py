import numpy as np
import time

# Number of vectors
N = 1_000_000

# Generate dummy data
positions = np.random.rand(N, 3).astype(np.float32)
velocities = np.random.rand(N, 3).astype(np.float32)

# ----------------------------
# Vectorized using NumPy
# ----------------------------
start = time.time()
new_positions_vectorized = positions + velocities
elapsed_vectorized = time.time() - start
print(f"Vectorized addition time: {elapsed_vectorized:.6f} seconds")

# ----------------------------
# Naive loop
# ----------------------------
start = time.time()
new_positions_loop = np.zeros_like(positions)
for i in range(N):
    new_positions_loop[i] = positions[i] + velocities[i]
elapsed_loop = time.time() - start
print(f"Loop addition time:      {elapsed_loop:.6f} seconds")

# ----------------------------
# Validate results match
# ----------------------------
if np.allclose(new_positions_vectorized, new_positions_loop):
    print("Results match")
else:
    print("Results differ")