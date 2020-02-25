from pybamm import exp, standard_parameters_lithium_ion


def lico2_diffusivity_Dualfoil1998(sto, T):
    """
    LiCo2 diffusivity as a function of stochiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from Dualfoil [1].

    References
    ----------
    .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Electrode stochiometry
    T : :class:`pybamm.Symbol`
        Dimensional temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity [m2.s-1]
    """
    param = standard_parameters_lithium_ion
    D_ref = 1 * 10 ** (-13)
    arrhenius = exp(param.E_D_s_p / param.R * (1 / param.T_ref - 1 / T))

    return D_ref * arrhenius
