import pybamm


def NMC_electrolyte_reaction_rate_PeymanMPM(T, T_inf, E_r, R_g):
    """
    Reaction rate for Butler-Volmer reactions between NMC and LiPF6 in EC:DMC.

    References
    ----------
    .. Peyman MPM manuscript (to be submitted)

    Parameters
    ----------
    T: :class: `numpy.Array`
        Dimensional temperature
    T_inf: double
        Reference temperature
    E_r: double
        Reaction activation energy
    R_g: double
        The ideal gas constant

    Returns
    -------
    : double
        Reaction rate
    """
    m_ref = 4.824 * 10 ** (-6)
    arrhenius = pybamm.exp(E_r / R_g * (1 / T_inf - 1 / T))

    return m_ref * arrhenius
