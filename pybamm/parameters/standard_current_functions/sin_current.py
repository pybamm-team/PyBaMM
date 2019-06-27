#
# Sinusoidal current
#
import numpy as np


def sin_current(t):
    """Sinusoidal current as a function of time in seconds"""
    # output has to have same shape and type as t
    return np.sin(2 * np.pi * t)
