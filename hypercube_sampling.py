from scipy.stats import qmc
import numpy as np
import pandas as pd
import math
from simulation_multiprocess_v2 import *

# Create Latin Hypercube for 3 variables
sampler = qmc.LatinHypercube(d=10)
sample = sampler.random(n=1)

# Define lower and upper bounds for each variable
l_bounds = [5,	0.1, 	1e9, 	25e-9, 	0.05, 	1, 	0, 	 0.01, 	0.01, 	0.01]
u_bounds = [50,	10,		10e9, 	100e-9, 0.5, 	10, 100, 0.5, 	0.1, 	0.5	]

# Scale all dimensions at once
scaled = qmc.scale(sample, l_bounds, u_bounds)

# Assign columns
flying_height = scaled[:, 0]
water_depth = scaled[:, 1]
sample_rate = scaled[:, 2]
t_max = scaled[:, 3]
absorption_coefficient = scaled[:, 4]
total_scattering_coefficient = scaled[:, 5]
salinity_unit = scaled[:, 6]
seafloor_albedo = scaled[:, 7]
water_surface_roughness = scaled[:, 8]
water_surface_albedo = scaled[:, 9]

simulations = pd.DataFrame({
    'flying_height': flying_height,
    'water_depth': water_depth,
    'sample_rate': sample_rate,
    't_max': t_max,
    'absorption_coefficient': absorption_coefficient,
    'total_scattering_coefficient': total_scattering_coefficient,
    'salinity_unit': salinity_unit,
    'seafloor_albedo': seafloor_albedo,
    'water_surface_roughness': water_surface_roughness,
    'water_surface_albedo': water_surface_albedo,
})

if __name__ == "__main__":
    for index, row in simulations.iterrows():
        distance = ((row["flying_height"] +  row["water_depth"]) / math.cos(30))
        steps = int(1.5 * (distance * row["sample_rate"]) / 2.998e8)
        print(steps)
        options = {
            'flying_height': row["flying_height"],
            'water_depth': row["water_depth"],
            'sample_rate': row["sample_rate"],
            't_max': row["t_max"],
            'absorption_coefficient': row["absorption_coefficient"],
            'total_scattering_coefficient': row["total_scattering_coefficient"],
            'salinity_unit': row["salinity_unit"],
            'seafloor_albedo': row["seafloor_albedo"],
            'water_surface_roughness': row["water_surface_roughness"],
            'water_surface_albedo': row["water_surface_albedo"],
        }
        print(options)

        start = time.time()
        nproc = 24

        # ------------------------------
        # Forward pass (parallel + progress)
        # ------------------------------
        photons_per_batch = 5_000
        forward_batches = 24
        forward_args = [(photons_per_batch, steps, secrets.randbits(64), options)
                        for _ in range(forward_batches)]

        print("Starting forward pass...")
        with mp.Pool(processes=nproc) as pool:
            forward_results = run_with_progress(pool, forward_worker,
                                                forward_args, "Forward", forward_batches)

        photon_np_array = np.concatenate([res for res in forward_results if res is not None])
        print(f"Photon map has {len(photon_np_array):,} entries, "
            f"size {(sys.getsizeof(photon_np_array)/1024/1024):.2f} MiB")

        # ------------------------------
        # Backward pass (parallel + persistent KDTree + progress)
        # ------------------------------
        photons_per_batch = 5_000
        backward_batches = 24

        # Build args list for backward batches (seeds only)
        backward_args = [(photons_per_batch, steps, secrets.randbits(64), options)
                        for _ in range(backward_batches)]

        print("Starting backward pass...")
        start_time = time.time()
        with mp.Pool(processes=nproc,
                    initializer=backward_worker_init,
                    initargs=(photon_np_array,)) as pool:
            print(f"initialized workers in {(time.time()-start_time):.2f} s ({(time.time()-start_time)/60:.2f} min)")
            backward_results = run_with_progress(pool, backward_worker_batch,
                                                backward_args, "Backward", backward_batches)

        total_waveform = np.sum(backward_results, axis=0)
        print("Simulation finished.")
        print(f"Total time {(time.time()-start):.2f} s ({(time.time()-start)/60:.2f} min)")
        plot_2d(total_waveform, title="waveform", ylabel="Intensity", xlabel="Sample")
