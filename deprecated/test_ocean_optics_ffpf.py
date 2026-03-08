import numpy as np
import matplotlib.pyplot as plt

from utils.gpt_ffscatter import generate_ff_phase_function
from alb_sim.plotting.plot_waveform import plot_waveform
from alb_sim.plotting.plot_histogram import plot_histogram
from utils.ffscatter import FFScatter

n = 1.1
mu = 3.5
M = 18000
epsilon = 1e-6

def ff_scatter(n, mu, M, term3fac = 1):
    psi = np.linspace(epsilon, np.pi, M)

    v = (3-mu) / 2
    delta = (4 / (3 * (n - 1)**2)) * np.sin(psi / 2)**2
    delta_180 = (4 / (3 * (n - 1)**2)) * np.sin(np.pi / 2)**2

    term1 = 1/(4 * np.pi * (1-delta)**2 * delta**v)
    term2 = v * (1-delta) - (1-delta**v) + (delta * (1-delta**v) - v*(1-delta)) * np.sin(psi/2)**(-2)
    term3 = ((1 - delta_180**v)/(16 * np.pi * (delta_180 - 1) * delta_180**v)) * (3 * np.cos(psi)**2 - 1)
    ff = term1 * term2 + term3fac*term3

    return psi, ff

def ff_backscatter_fraction(n, mu):
    v = (3-mu) / 2
    delta_90 = (4 / (3 * (n - 1)**2)) * np.sin(np.pi / 4)**2

    return 1 - ((1 - delta_90**(v+1) - 0.5 * (1 - delta_90**v)) / ((1-delta_90) * delta_90**v))

def sample_scattering_angle(cdf, psi):
    rng = np.random.default_rng(42)
    u = rng.random()
    return np.interp(u, cdf, psi)

def sample_scattering_angle_batch(num_samples, cdf, psi):
    rng = np.random.default_rng(42)
    u = rng.random(num_samples)
    return np.interp(u, cdf, psi)

def get_bin_probability(psi_val, psi, cdf, M):
    delta = (np.pi - epsilon) / M
    i = int((psi_val - epsilon) / delta)
    i = min(i, M - 1)  # clamp to avoid overflow at edge

    psi_lower = i * delta
    psi_upper = (i + 1) * delta

    cdf_lower = np.interp(psi_lower, psi, cdf)
    cdf_upper = np.interp(psi_upper, psi, cdf)

    return cdf_upper - cdf_lower


plt.figure(figsize=(8, 4))
psi, ff = ff_scatter(1.1, 3.62, 18000)
cdf = np.cumsum(ff * np.sin(psi))
cdf /= cdf[-1]
print("backscatter fraction", ff_backscatter_fraction(1.1, 3.62))
plt.semilogx(np.degrees(psi), cdf, label=f"Ocean optics WB, Bp = {ff_backscatter_fraction(1.1, 3.62):.3f}")

psi2, ff2 = ff_scatter(1.15, 4.5, 18000)
cdf2 = np.cumsum(ff2 * np.sin(psi2))
cdf2 /= cdf2[-1]
print("backscatter fraction", ff_backscatter_fraction(1.15, 4.5))
plt.semilogx(np.degrees(psi2), cdf2, label=f"Ocean optics WB, Bp = {ff_backscatter_fraction(1.15, 4.5):.3f}")

# psi3, ff3 = ff_scatter(1.15, 4.5, 18000, 0)
# cdf3 = np.cumsum(ff3 * np.sin(psi3))
# cdf3 /= cdf3[-1]
# print("backscatter fraction", ff_backscatter_fraction(1.15, 4.5))
# plt.semilogx(np.degrees(psi3), cdf3, label=f"Ocean optics WB, Bp = {ff_backscatter_fraction(1.15, 4.5):.3f}, no term3")

_, _, p_ct_r_1, _, ct_r_1 = FFScatter(n_FF=1.10, M=18000)
print(p_ct_r_1)
print(ct_r_1)
matlab_cdf = np.cumsum(p_ct_r_1 * np.sin(ct_r_1))
matlab_cdf /= matlab_cdf[-1]
plt.semilogx(np.degrees(ct_r_1), p_ct_r_1, label="Matlab")

theta_gpt, ff_gpt = generate_ff_phase_function(1.1, 18000)
orig_cdf = np.cumsum(ff_gpt * np.sin(theta_gpt))
orig_cdf /= orig_cdf[-1]
plt.semilogx(np.degrees(theta_gpt), orig_cdf, label="Original GPT")

plt.xlabel('Scattering Angle (degrees)')
plt.ylabel('Phase Function p(theta)')
plt.title('Fournier-Forand Cumulative Phase Function')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# plot_histogram(np.degrees(sample_scattering_angle_batch(5000, cdf, psi)), 180*5, "Sampled angles", "Angle")

# plt.figure(figsize=(8, 4))
# plt.semilogx(np.degrees(psi), np.array([get_bin_probability(x, psi, cdf, M) for x in np.linspace(epsilon, np.pi, M)]))
# plt.xlabel('Angle')
# plt.ylabel('Probability')
# plt.title('Angle probabilities')
# plt.grid(True)
# plt.tight_layout()
# plt.show()