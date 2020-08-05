import autograd.numpy as np


def lico2_electrolyte_reaction_rate_Dualfoil1998(T, T_inf, E_r, R_g,sto,c_e, c_n_max):
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
	sto: double
            Stochiometry of material (li-fraction)
    c_e: :class: `numpy.Array`
        Dimensional electrolyte concentration
    c_p_max: double
        Maximum concentration in anode


    Returns
    -------
    :`numpy.Array`
        Reaction rate
    """
    m_ref =1E-11* c_e**0.5*(sto*c_p_max)**0.5*(c_p_max-sto*c_p_max)**0.5 
    arrhenius = np.exp(E_r / R_g * (1 / T_inf - 1 / T))

    return m_ref * arrhenius
