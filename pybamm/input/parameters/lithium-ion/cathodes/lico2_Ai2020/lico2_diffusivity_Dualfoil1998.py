from pybamm import exp, constants, Parameter


def lico2_diffusivity_Dualfoil1998(sto, T):
    """
    LiCo2 diffusivity as a function of stochiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from Dualfoil [1].

    References
    ----------
    .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature, [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity [m2.s-1]
    """
    D_ref = 5.387 * 10 ** (-15)
    E_D_s = 5000
    T_ref = Parameter("Reference temperature [K]")
    arrhenius = exp(E_D_s / constants.R * (1 / T_ref - 1 / T))
    return D_ref * arrhenius
