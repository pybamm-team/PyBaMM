from pybamm import standard_parameters_lead_acid


def lead_dioxide_exchange_current_density_Sulzer2019(c_e, T):
    """
    Dimensional exchange-current density in the positive electrode, from [1]_

    References
    ----------
    .. [1] V. Sulzer, S. J. Chapman, C. P. Please, D. A. Howey, and C. W. Monroe,
    “Faster lead-acid battery simulations from porous-electrode theory: Part I. Physical
    model.”
    [Journal of the Electrochemical Society](https://doi.org/10.1149/2.0301910jes),
    166(12), 2363 (2019).

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]

    """
    c_ox = 0
    c_hy = 0
    param = standard_parameters_lead_acid
    c_w_dim = (1 - c_e * param.V_e - c_ox * param.V_ox - c_hy * param.V_hy) / param.V_w
    c_w_ref = (1 - param.c_e_typ * param.V_e) / param.V_w
    c_w = c_w_dim / c_w_ref

    j0_ref = 0.004  # srinivasan2003mathematical
    j0 = j0_ref * (c_e / param.c_e_typ) ** 2 * c_w

    return j0
