from scipy.stats import qmc
import numpy as np
import pandas as pd
import math
from simulation_multiprocess_v2 import *
from alb_sim.plotting.plot_waveform import plot_2d_better
import utils.numpy_vector as np_vec
import matplotlib
matplotlib.use("TkAgg")
from deprecated.hypercube_xlim_regression import predict_xlim

# Create Latin Hypercube for 10 variables
sampler = qmc.LatinHypercube(d=10)
sample = sampler.random(n=50)

# Define lower and upper bounds for each variable
l_bounds = [5,	0.1, 	15e9 - 1, 	25e-9, 	0.05, 	0.1, 	0, 	 0.01, 	0.01, 	0.01]
u_bounds = [50,	10,		15e9, 	    100e-9, 0.5,    5,      40,   0.5, 	0.1, 	0.5	]

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
        distance = ((row["flying_height"] + row["water_depth"]) / math.cos(math.radians(15)))
        steps = int(1.5 * (distance * round(row["sample_rate"])) / 2.998e8)
        print(steps)
        print(f"{steps} steps, this will simulate {steps * 2.998e8 / round(row['sample_rate'])} m")
        print(f"distance laser - seafloor is {round(np.dot(np.array([0, -(row['flying_height'] + row['water_depth']), 0]), np.array([0, 1, 0]))/np.dot(np_vec.normalize_vector(np.array([math.sin(math.radians(30 / 2)), -math.cos(math.radians(30 / 2)), 0])), np.array([0, 1, 0])), 2)} m")
        options = {
            'flying_height': row["flying_height"],
            'water_depth': row["water_depth"],
            'sample_rate': round(15e9),
            'sample_multiplier': 50,
            't_max': row["t_max"],
            'absorption_coefficient': row["absorption_coefficient"],
            'total_scattering_coefficient': row["total_scattering_coefficient"],
            'salinity_unit': row["salinity_unit"],
            'seafloor_albedo': row["seafloor_albedo"],
            'water_surface_roughness': row["water_surface_roughness"],
            'water_surface_albedo': row["water_surface_albedo"],
        }
        print(options)
        print([float(options["flying_height"]), float(options["water_depth"]), float(options["sample_rate"])])
        start = time.time()
        nproc = 16
        options["num_workers"] = nproc

        # ------------------------------
        # Forward pass (parallel + progress)
        # ------------------------------
        photons_per_batch = 10_000
        forward_batches = 16
        
        options["photons_per_batch_forward"] = photons_per_batch
        options["batches_forward"] = forward_batches
        
        forward_args = [(photons_per_batch, steps, secrets.randbits(64), options)
                        for _ in range(forward_batches)]

        print("Starting forward pass...")
        with mp.Pool(processes=nproc) as pool:
            forward_results = run_with_progress(pool, forward_worker,
                                                forward_args, "Forward", forward_batches)

        photon_np_array = np.concatenate([res for res in forward_results if res is not None])
        print(f"Photon map has {len(photon_np_array):,} entries, "
            f"size {(sys.getsizeof(photon_np_array)/1024/1024):.2f} MiB")
        
        options["photon_map_size"] = len(photon_np_array)

        # ------------------------------
        # Backward pass (parallel + persistent KDTree + progress)
        # ------------------------------
        photons_per_batch = 10_000
        backward_batches = 16
        
        options["photons_per_batch_backward"] = photons_per_batch
        options["batches_backward"] = backward_batches

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
        
        options["total_time_min"] = (time.time()-start) / 60
        
        # xmin =  -1259.7848010107764 + np.dot(np.array([3.81108559e+01, 3.56798396e+01, 1.87900348e-07]), np.array([options["flying_height"], options["water_depth"], options["sample_rate"]]))
        # xmax =  -1384.4673576234168 + np.dot(np.array([4.17736964e+01, 4.21878048e+01, 2.06038173e-07]), np.array([options["flying_height"], options["water_depth"], options["sample_rate"]]))
        print(predict_xlim(options["flying_height"], options["water_depth"], options["sample_rate"], options["sample_multiplier"]))
        plot_2d_better(total_waveform, title="waveform", ylabel="Intensity", xlabel="Sample", save_path=f"images/hypercube-sample-v8/{index}-full.png", params=options) # , xlim=(xmin, xmax)
        plot_2d_better(total_waveform, title="waveform", ylabel="Intensity", xlabel="Sample", save_path=f"images/hypercube-sample-v8/{index}.png", params=options, xlim=predict_xlim(options["flying_height"], options["water_depth"], options["sample_rate"], options["sample_multiplier"])) # , xlim=(xmin, xmax)
