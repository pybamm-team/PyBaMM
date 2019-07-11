import autograd.numpy as np


def electrolyte_conductivity_Capiglia1999(c_e, T, T_inf, E_k_e, R_g):
    """
    Conductivity of LiPF6 in EC:DMC as a function of ion concentration. The original
    data is from [1]. The fit is from Dualfoil [2].

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
    E_k_e: :class: `pybamm.Parameter`
        Electrolyte conductivity activation energy
    R_g: :class: `pybamm.Parameter`
        The ideal gas constant

    Returns
    -------
    :`pybamm.Symbol`
        Solid diffusivity
    """

    sigma_e = (
        0.0911
        + 1.9101 * (c_e / 1000)
        - 1.052 * (c_e / 1000) ** 2
        + 0.1554 * (c_e / 1000) ** 3
    )

    arrhenius = np.exp(E_k_e / R_g * (1 / T_inf - 1 / T))

    return sigma_e * arrhenius
