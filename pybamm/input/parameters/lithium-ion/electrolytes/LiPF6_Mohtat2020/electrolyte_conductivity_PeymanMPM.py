import pybamm


def electrolyte_conductivity_PeymanMPM(c_e, T, T_inf, E_k_e, R_g):
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
    c_e: :class: `numpy.Array`
        Dimensional electrolyte concentration
    T: :class: `numpy.Array`
        Dimensional temperature
    T_inf: double
        Reference temperature
    E_k_e: double
        Electrolyte conductivity activation energy
    R_g: double
        The ideal gas constant

    Returns
    -------
    :`numpy.Array`
        Electrolyte conductivity
    """

    sigma_e = 1.3

    arrhenius = pybamm.exp(E_k_e / R_g * (1 / T_inf - 1 / T))

    return sigma_e * arrhenius
