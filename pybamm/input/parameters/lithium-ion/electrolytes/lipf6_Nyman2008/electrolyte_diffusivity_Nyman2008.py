from pybamm import exp


def electrolyte_diffusivity_Nyman2008(c_e, T, T_inf, E_D_e, R_g):
    """
    Diffusivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1]

    References
    ----------
    .. [1] A. Nyman, M. Behm, and G. Lindbergh, "Electrochemical characterisation and
    modelling of the mass transport phenomena in LiPF6-EC-EMC electrolyte,"
    Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.

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

    D_c_e = 8.794e-11 * (c_e / 1000) ** 2 - 3.972e-10 * (c_e / 1000) + 4.862e-10
    arrhenius = exp(E_D_e / R_g * (1 / T_inf - 1 / T))

    return D_c_e * arrhenius
