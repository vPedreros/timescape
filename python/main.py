import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science'])

c = 3e5  # km/s

def H_bar_0(fv0, H_0):
    return 2 * (2 + fv0) * H_0 / (4 *fv0**2 + fv0 + 4)


def b(fv0, H_0):
    output = 2 * (1 - fv0) * (2 + fv0) / (9 * fv0 * H_bar_0(fv0, H_0))
    return output


def F(t, fv0, H_0):
    term1 = 2 * t ** (1/3)
    term2 = b(fv0, H_0)**(1/3) / 6 * np.log( (t**(1/3) + b(fv0, H_0)**(1/3))**2 / (t **(2/3) - b(fv0, H_0)**(1/3) * t**(1/3) + b(fv0, H_0)**(2/3)) )
    term3 = b(fv0, H_0)**(1/3) / np.sqrt(3) * np.arctan( (2 * t**(1/3) - b(fv0, H_0)**(1/3)) / (np.sqrt(3) * b(fv0, H_0)**(1/3)) )
    return term1 + term2 + term3


def angular_diameter_distance(t, t0, fv0, H_0):
    return c * t ** (2 / 3) * (F(t0, fv0, H_0) - F(t, fv0, H_0))

def z_of_t(t, fv0, H_0):
    return 2**(4/3) * t ** (1/3) * (t + b(fv0, H_0)) / (fv0 **(1/3) * H_bar_0(fv0, H_0)* t * (2*t+3*b(fv0, H_0)**(4/3))) 