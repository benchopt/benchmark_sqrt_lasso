import numpy as np
from numba import njit


@njit
def prox_conjugate_L2(z, step, y):
    # arg min_w ||y - w||^* + 1/(2*step) * ||w - z||^2

    # project `u = z - step * y` on the L2 unit ball
    u = z - step * y

    norm_u = np.linalg.norm(u)
    if norm_u <= 1.:
        return u
    return u / norm_u


@njit
def prox_L1(x, step, alpha):
    # arg min_w ||w||_1 + 1/(2*step) ||w - z||^2
    # entry-wise soft threshold
    return np.sign(x) * np.maximum(0., np.abs(x) - step*alpha)
