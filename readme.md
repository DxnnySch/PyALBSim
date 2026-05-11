# PyALBSim

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20068111.svg)](https://doi.org/10.5281/zenodo.20068111)

This repository contains a Python-based simulation framework for airborne laser bathymetry (ALB) using photon tracing and photon mapping.

The simulation models:

- Emission of laser photons in air
- Interaction with the water surface (reflection and refraction)
- Scattering and absorption within the water column
- Reflection at the seafloor
- Propagation back through the water surface to the sensor

The code is optimized for vectorized NumPy operations and supports multiple execution strategies, including multiprocessing on Linux-based systems.

---

## Project Structure

```txt
PyALBSim/
├── src/alb_sim/        # Core simulation package
├── scripts/            # Executable scripts (batch runs, multiprocessing)
├── notebooks/          # Jupyter notebooks (explanations, visualization)
├── pyproject.toml
└── readme.md
```

The core simulation code lives in `src/alb_sim` and can be imported as a regular Python package.

---

## System Requirements

- Python 3.9 or newer
- Linux (recommended)
- Windows supported via WSL (multiprocessing is limited on native Windows)
- macOS: untested

---

## Setup

### 1. Clone the Repository

### 2. Create and activate a virtual environment

Linux / WSL:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Ensure that python and pip now point to the virtual environment.

### 3. Install the core package (editable mode)

```bash
pip install -e .
```

This installs:

- The core simulation package
- All required runtime dependencies
- An editable link to the source code (changes take effect immediately)

---

## Optional Dependencies

The project defines optional dependency groups for different use cases.

### Jupyter notebooks (recommended for inexperienced users)

```bash
pip install -e .[notebooks]
```

This installs:

- ipykernel
- jupyterlab
- plotting dependencies used in notebooks

Register the kernel once:

```bash
python -m ipykernel install --user --name alb-sim --display-name "ALB Simulation"
```

### Plotting and analysis tools

```bash
pip install -e .[plotting]
```

Installs libraries required for plotting and visualization outside notebooks.

### Development dependencies

```bash
pip install -e .[dev]
```

Includes formatting, linting and type checking tools.

Multiple extras can be combined:

```bash
pip install -e .[notebooks,dev]
```

---

## Usage

### Running a simulation script

Example (linear execution):

```bash
python scripts/run_linear.py
```

Multiprocessing execution (Linux / WSL only):

```bash
python scripts/run_parallel.py
```

Simulation parameters are typically defined via configuration objects or files imported by the script.

### Jupyter notebooks

After activating the virtual environment, installing the notebook dependencies and registering the kernel:

```bash
jupyter lab
```

This will start a webserver that give a graphical user interface for running jupyter notebooks in the browser.
Open one of the notebooks in `notebooks/` and select the ALB Simulation kernel.

---

## Development

### Editable installs

This project uses editable installs (`pip install -e .`) to ensure:

- Imports work consistently across scripts, tests, and notebooks
- Code changes take effect immediately
- No manual PYTHONPATH configuration is required

### Code style

- Prefer vectorized NumPy operations
- Avoid side effects at import time
- Keep execution logic separate from core simulation code
- Multiprocessing logic should live outside the core simulation classes

### Platform Notes

- Multiprocessing is implemented using the `multiprocess` module (as multiprocessing is not reliable in jupyter notebooks) and works best on Linux.
- On Windows, multiprocessing is limited due to process spawning behavior.
- For Windows users, WSL is strongly recommended.

### Development tools

- *Black* for formatting, `black src/ notebooks/ scripts/`
- *Ruff* for linting, `ruff check --fix src/ notebooks/ scripts/`
- *Mypy* for type checking, `mypy --check-untyped-defs src/`

---

## License

All rights reserved. See `LICENSE` for more information.

---

## References

- **Simulation is based on the implementation of Yang et al.**:  
  F. Yang et al., ‘Modeling and Analyzing Water Column Forward Scattering Effect on Airborne LiDAR Bathymetry’, IEEE Journal of Oceanic Engineering, vol. 48, no. 4, pp. 1373–1388, Oct. 2023, doi: 10.1109/JOE.2023.3275695.
- **Water interaction, scattering and absorption**:  
  C. D. Mobley, ‘The Oceanic Optics Book’, 2022, International Ocean Colour Coordinating Group (IOCCG), Dartmouth, NS, Canada. doi: 10.25607/OBP-1710.
- **Microfacet and lambertian BRDF**:  
  J. Boksansky, ‘Crash course in BRDF implementation’. Feb. 2021. Accessed: Jan. 21, 2026. [Online]. Available: <https://boksajak.github.io/blog/BRDF>
- **Photon Mapping**:  
  H. W. Jensen, ‘Global Illumination using Photon Maps’, in Rendering Techniques ’96, X. Pueyo and P. Schröder, Eds, Vienna: Springer, 1996, pp. 21–30. doi: 10.1007/978-3-7091-7484-5_3.
