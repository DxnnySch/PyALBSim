import numpy as np
import matplotlib.pyplot as plt

from alb_sim.config.water import FournierForandConfig
from alb_sim.physics.models.water import FournierForandModel
from alb_sim.physics.scatter.fournier_forand_phase_function import calculate_phase_function_matlab


default_fournier_forand_bp_02 = FournierForandModel(FournierForandConfig())
default_fournier_forand_bp_20 = FournierForandModel(FournierForandConfig(refractive_index_ratio=1.21, junge_slope=4.4))
matlab_fournier_forand = calculate_phase_function_matlab(refractive_index_ratio=1.21, junge_slope=4.4)

plt.figure(figsize=(8, 4))
plt.semilogx(np.degrees(default_fournier_forand_bp_02.theta), default_fournier_forand_bp_02.cdf, label=f"B_p = 0.02")
plt.semilogx(np.degrees(default_fournier_forand_bp_20.theta), default_fournier_forand_bp_20.cdf, label=f"B_p = 0.20")
plt.semilogx(np.degrees(matlab_fournier_forand[0]), matlab_fournier_forand[1], label=f"B_p = 0.20 (Matlab)")

plt.xlabel('Scattering Angle (degrees)')
plt.ylabel('Phase Function p(theta)')
plt.title('Fournier-Forand Phase Function')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()