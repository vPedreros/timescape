import numpy as np
from scipy.integrate import quad


# Constants
c = 299792.458      # km / s
H0 = 67             # km / s / Mpc
fv0 = 0.695
Omm = 0.315


# ===== LCDM Functions =====
def Hubble(z, H0=H0, Omega_m=Omm):
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))


def DM_lcdm(z, H0=H0, Omega_m=Omm): 
    integrand = lambda zp: c / Hubble(zp, H0, Omega_m)
    distance, _ = quad(integrand, 0, z)
    return distance


def DA_lcdm(z, H0=H0, Omega_m=Omm):
    d_c = DM_lcdm(z, H0, Omega_m)
    return d_c / (1 + z)
DA_lcdm = np.vectorize(DA_lcdm)


def DH_lcdm(z, H0=H0, Omega_m=Omm):
    return c / Hubble(z, H0, Omega_m)


def DV_lcdm(z, H0=H0, Omega_m=Omm):
    d_a = DA_lcdm(z, H0, Omega_m)
    d_h = DH_lcdm(z, H0, Omega_m)
    return np.cbrt(d_a**2 * d_h * z)


# ===== Timescape Functions =====
def H_bar_0(fv0, H_0=H0):
    return 2 * (2 + fv0) * H_0 / (4 *fv0**2 + fv0 + 4)


def fv(t, fv0, H_0=H0):
    return (3 * fv0 * H_bar_0(fv0, H_0) * t) / (3 * fv0 * H_bar_0(fv0, H_0) * t + (1 - fv0) * (2 + fv0))


def H_timescape(t, fv0, H0):
    return (2 + fv(t, fv0, H0)) / (3 * t)


def b(fv0, H0):
    output = 2 * (1 - fv0) * (2 + fv0) / (9 * fv0 * H_bar_0(fv0, H0))
    return output


def F(t, fv0, H0):
    tt = np.cbrt(t)
    bb = np.cbrt(b(fv0, H0))
    term1 = 2 * tt
    term2 = bb / 6 * np.log( (tt + bb)**2 / (tt**2 - bb * tt + bb**2) )
    term3 = bb / np.sqrt(3) * np.arctan( (2 * tt - bb) / (np.sqrt(3) * bb) )
    return term1 + term2 + term3


def DA_timescape(t, t0, fv0, H0):
    return c * t ** (2 / 3) * (F(t0, fv0, H0) - F(t, fv0, H0))


def DH_timescape(t, fv0, H0):
    return c / H_timescape(t, fv0, H0)


def DV_timescape(t, t0, fv0, H0):
    d_a = DA_timescape(t, t0, fv0, H0)
    d_h = DH_timescape(t, fv0, H0)
    return np.cbrt(d_a**2 * d_h * z_of_t(t, fv0, H0))


def z_of_t(t, fv0, H0):
    return 2**(4/3) * t ** (1/3) * (t + b(fv0, H0)) / (fv0 **(1/3) * H_bar_0(fv0, H0)* t * (2*t+3*b(fv0, H0))**(4/3)) -1


def dressed_Omm(fv0):
    return .5 * (1 - fv0) * (2 + fv0)


def fv0_from_dressed_Omm(Omm):
    return 0.5 * (np.sqrt(9 - 8 * Omm) - 1)