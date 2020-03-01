from pybamm import exp


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
    c_e : :class:`pybamm.Symbol`
        Dimensional electrolyte concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Dimensional temperature [K]
    T_inf: :class:`pybamm.Symbol`
        Reference temperature [K]
    E_D_e: :class:`pybamm.Symbol`
        Electrolyte diffusion activation energy [J.mol-1]
    R_g: :class:`pybamm.Symbol`
        The ideal gas constant [J.mol-1.K-1]

    Returns
    -------
    :class:`pybamm.Symbol`
        Dimensional electrolyte diffusivity [m2.s-1]
    """

    D_c_e = 5.34e-10 * exp(-0.65 * c_e / 1000)
    arrhenius = exp(E_D_e / R_g * (1 / T_inf - 1 / T))

    return D_c_e * arrhenius
