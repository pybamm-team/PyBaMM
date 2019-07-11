import autograd.numpy as np


def electrolyte_diffusivity_Capiglia1999(c_e, T, T_inf, E_D_e, R_g):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration. The original data
    is from [1]. The fit from Dualfoil [2].

    References
    ----------
    .. [1] C Capiglia et al. 7Li and 19F diffusion coefficients and thermal
    properties of non-aqueous electrolyte solutions for rechargeable lithium batteries.
    Journal of power sources 81 (1999): 859-862.
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    c_e: :class: `pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class: `pybamm.Symbol`
        Dimensional temperature
    T_inf: :class: `pybamm.Parameter`
        Reference temperature
    E_D_e: :class: `pybamm.Parameter`
        Electrolyte diffusion activation energy
    R_g: :class: `pybamm.Parameter`
        The ideal gas constant

    Returns
    -------
    :`pybamm.Symbol`
        Solid diffusivity
    """

    D_c_e = 5.34e-10 * np.exp(-0.65 * c_e / 1000)
    arrhenius = np.exp(E_D_e / R_g * (1 / T_inf - 1 / T))

    return D_c_e * arrhenius
