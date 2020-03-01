from pybamm import exp


def electrolyte_conductivity_Valoen2005(c_e, T, T_inf, E_k_e, R_g):
    """
    Conductivity of LiPF6 in EC:DMC as a function of ion concentration, from [1]

    References
    ----------
    .. [1] Valøen, Lars Ole, and Jan N. Reimers. "Transport properties of LiPF6-based
    Li-ion battery electrolytes." Journal of The Electrochemical Society 152.5 (2005):
    A882-A891.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Dimensional electrolyte concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Dimensional temperature [K]
    T_inf: :class:`pybamm.Symbol`
        Reference temperature [K]
    E_k_e: :class:`pybamm.Symbol`
        Electrolyte diffusion activation energy [J.mol-1]
    R_g: :class:`pybamm.Symbol`
        The ideal gas constant [J.mol-1.K-1]

    Returns
    -------
    :class:`pybamm.Symbol`
        Dimensional electrolyte conductivity [S.m-1]
    """
    # mol/m3 to molar
    c_e = c_e / 1000
    # mS/cm to S/m
    return (1e-3 / 1e-2) * (
        c_e
        * (
            (-10.5 + 0.0740 * T - 6.96e-5 * T ** 2)
            + c_e * (0.668 - 0.0178 * T + 2.80e-5 * T ** 2)
            + c_e ** 2 * (0.494 - 8.86e-4 * T)
        )
        ** 2
    )
