from pybamm import exp


def nmc_LGM50_electrolyte_reaction_rate_Chen2020(T, T_inf, E_r, R_g):
    """
    Reaction rate for Butler-Volmer reactions between NMC and LiPF6 in EC:DMC.
    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Submitted for
    publication (2020).
    Parameters
    ----------
    T : :class:`pybamm.Symbol`
        Dimensional temperature [K]
    T_inf: :class:`pybamm.Symbol`
        Reference temperature [K]
    E_r: :class:`pybamm.Symbol`
        Reaction activation energy [J.mol-1]
    R_g: :class:`pybamm.Symbol`
        The ideal gas constant [J.mol-1.K-1]
    Returns
    -------
    : :class:`pybamm.Symbol`
        Reaction rate [(A.m-2)(m3.mol-1)^1.5]
    """
    m_ref = 3.59e-6
    arrhenius = exp(E_r / R_g * (1 / T_inf - 1 / T))

    return m_ref * arrhenius
