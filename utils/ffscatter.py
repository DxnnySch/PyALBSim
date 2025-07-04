import numpy as np
import matplotlib.pyplot as plt

def FFScatter(n_FF=1.10, M=18000):
    """
    Compute the Fourier-Forand (FF) phase function and related parameters.

    Parameters:
        n_FF (float): Refractive index ratio (particles vs water), default 1.10.
        M (int): Number of angular divisions, default 18000.

    Returns:
        v (float): Shape parameter from PSD.
        Bp (float): Proportion of backscattering.
        p_ct_r (ndarray): FF phase function values (length M).
        Delta_pi (float): Delta value at 180 degrees (used in FF model).
    """
    miu = 3 + (n_FF - 1.01) / 0.1542
    print("mu", miu)
    v = 0.5 * (3 - miu)  # FF parameter v

    k = np.arange(1, M + 1)
    ct_r = k * np.pi / M  # scattering angles from 0 to π

    Delta = 4 / (3 * (n_FF - 1)**2) * np.sin(ct_r / 2)**2
    Delta_pi = 4 / (3 * (n_FF - 1)**2)  # when theta = π
    Delta_90 = 4 / (3 * (n_FF - 1)**2) * np.sin(np.pi / 4)**2

    # Backscattering proportion
    numerator = 1 - Delta_90**(v + 1) - 0.5 * (1 - Delta_90**v)
    denominator = (1 - Delta_90) * Delta_90**v
    Bp = 1 - numerator / denominator

    # FF phase function
    term1 = 1 / ((1 - Delta) * Delta**v)
    term2 = (1 - Delta**(v + 1)) - (1 - Delta**v) * (np.sin(ct_r / 2)**2)
    term3 = (1/8) * (1 - Delta_pi**v) / ((Delta_pi - 1) * Delta_pi**v)
    term4 = np.cos(ct_r) * (np.sin(ct_r)**2)

    p_ct_r = term1 * term2 + term3 * term4

    return v, Bp, p_ct_r, Delta_pi, ct_r

if __name__ == "__main__":
    v_1, Bp_1, p_ct_r_1, Delta_pi_1, ct_r_1 = FFScatter(n_FF=1.10, M=18000)
    print(v_1, Bp_1)
    plt.figure(figsize=(8, 4))
    cdf = np.cumsum(p_ct_r_1 * np.sin(ct_r_1))
    cdf /= cdf[-1]
    plt.semilogx(np.degrees(ct_r_1), cdf)
    plt.xlabel('Scattering Angle (degrees)')
    plt.ylabel('Phase Function p(theta)')
    plt.title('Fournier-Forand Phase Function')
    plt.grid(True)
    plt.tight_layout()
    plt.show()