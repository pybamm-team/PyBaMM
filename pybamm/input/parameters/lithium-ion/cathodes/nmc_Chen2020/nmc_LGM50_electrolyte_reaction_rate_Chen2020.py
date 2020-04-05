from pybamm import exp, constants


def nmc_LGM50_electrolyte_reaction_rate_Chen2020(T):
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
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Reaction rate
    """
    m_ref = 3.59e-6
    E_r = 17800
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius
