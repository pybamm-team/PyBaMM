import autograd.np as np


def graphite_electrolyte_reaction_rate(T, T_inf, E_r, R_g):
    """
    Reaction rate for Butler-Volmer reactions between graphite and LiPF6 in EC:DMC.

    References
    ----------
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    T: :class: `pybamm.Symbol`
        Dimensional temperature
    T_inf: :class: `pybamm.Parameter`
        Reference temperature
    E_r: :class: `pybamm.Parameter`
        Reaction activation energy
    R_g: :class: `pybamm.Parameter`
        The ideal gas constant

    Returns
    -------
    :`pybamm.Symbol`
        Reaction rate
    """
    m_ref = 2 * 10 ** (-5)
    arrhenius = np.exp(E_r / R_g * (1 / T_inf - 1 / T))

    return m_ref * arrhenius

