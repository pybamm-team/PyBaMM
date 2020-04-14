from pybamm import exp, constants


def NMC_electrolyte_reaction_rate_PeymanMPM(T):
    """
    Reaction rate for Butler-Volmer reactions between NMC and LiPF6 in EC:DMC.

    References
    ----------
    .. Peyman MPM manuscript (to be submitted)

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
        Dimensional temperature
    Returns
    -------
    :class:`pybamm.Symbol`
        Reaction rate
    """
    m_ref = 4.824 * 10 ** (-6)
    E_r = 39570
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius
