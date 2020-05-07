from pybamm import exp, constants, Scalar


def electrolyte_conductivity_Capiglia1999(c_e, T):
    """
    Conductivity of LiPF6 in EC:DMC as a function of ion concentration. The original
    data is from [1]. The fit is from Dualfoil [2].

    References
    ----------
    .. [1] C Capiglia et al. 7Li and 19F diffusion coefficients and thermal
    properties of non-aqueous electrolyte solutions for rechargeable lithium batteries.
    Journal of power sources 81 (1999): 859-862.
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature


    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    sigma_e = (
        Scalar(0.0911, "[S.m-1]")
        + Scalar(1.9101, "[S.m-1]") * (c_e / Scalar(1000, "[mol.m-3]"))
        - Scalar(1.052, "[S.m-1]") * (c_e / Scalar(1000, "[mol.m-3]")) ** 2
        + Scalar(0.1554, "[S.m-1]") * (c_e / Scalar(1000, "[mol.m-3]")) ** 3
    )

    E_k_e = Scalar(34700, "[J.mol-1]")
    arrhenius = exp(E_k_e / constants.R * (1 / Scalar(298.15, "[K]") - 1 / T))

    return sigma_e * arrhenius
