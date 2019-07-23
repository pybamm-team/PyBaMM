import autograd.numpy as np


def lico2_electrolyte_reaction_rate(T, T_inf, E_r, R_g):
    """
    Reaction rate for Butler-Volmer reactions between lico2 and LiPF6 in EC:DMC.

    References
    ----------
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

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
    m_ref = 6 * 10 ** (-7)
    arrhenius = np.exp(E_r / R_g * (1 / T_inf - 1 / T))

    return m_ref * arrhenius
