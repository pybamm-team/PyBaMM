from pybamm import Parameter


def lead_exchange_current_density_Sulzer2019(c_e, T):
    """
    Dimensional exchange-current density in the negative (lead) electrode, from [1]_

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
    j0_ref = 0.06  # srinivasan2003mathematical
    c_e_typ = Parameter("Typical electrolyte concentration [mol.m-3]")
    j0 = j0_ref * (c_e / c_e_typ)

    return j0
