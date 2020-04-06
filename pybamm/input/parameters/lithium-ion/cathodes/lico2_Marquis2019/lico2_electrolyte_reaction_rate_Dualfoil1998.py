from pybamm import exp, constants


def lico2_electrolyte_reaction_rate_Dualfoil1998(T):
    """
    Reaction rate for Butler-Volmer reactions between lico2 and LiPF6 in EC:DMC.

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
    m_ref = 6 * 10 ** (-7)
    E_r = 39570
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius
