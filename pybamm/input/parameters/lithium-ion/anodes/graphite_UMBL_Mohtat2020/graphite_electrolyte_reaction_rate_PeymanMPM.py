from pybamm import exp, constants


def graphite_electrolyte_reaction_rate_PeymanMPM(T):
    """
    Reaction rate for Butler-Volmer reactions between graphite and LiPF6 in EC:DMC.
    Check the unit of Reaction rate constant k0 is from Peyman MPM.

    References
    ----------
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Reaction rate
    """
    m_ref = 1.061 * 10 ** (-6)  # unit has been converted
    E_r = 37480
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius
