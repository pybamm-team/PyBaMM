from pybamm import exp


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
    c_e : :class:`pybamm.Symbol`
        Dimensional electrolyte concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Dimensional temperature [K]
    T_inf: :class:`pybamm.Symbol`
        Reference temperature [K]
    E_k_e: :class:`pybamm.Symbol`
        Electrolyte conductivity activation energy [J.mol-1]
    R_g: :class:`pybamm.Symbol`
        The ideal gas constant [J.mol-1.K-1]

    Returns
    -------
    :class:`pybamm.Symbol`
        Dimensional electrolyte conductivity [S.m-1]
    """

    sigma_e = (
        0.0911
        + 1.9101 * (c_e / 1000)
        - 1.052 * (c_e / 1000) ** 2
        + 0.1554 * (c_e / 1000) ** 3
    )

    arrhenius = exp(E_k_e / R_g * (1 / T_inf - 1 / T))

    return sigma_e * arrhenius
