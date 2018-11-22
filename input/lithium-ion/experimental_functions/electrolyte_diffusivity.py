import numpy as np


def lfp_diff(c):
    """
           Diffusivity of LiPF6 in EC:DMC as in Newman's Dualfoil code. This
           function is in dimensional form.

           Parameters
           ----------
           c: double
                lithium-ion concentration

           """
    return 5.34e-10 * np.exp(-0.65 * c / 1000)
