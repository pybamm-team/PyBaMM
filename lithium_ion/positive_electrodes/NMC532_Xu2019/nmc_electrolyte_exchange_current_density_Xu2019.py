from pybamm import constants, Parameter


def nmc_electrolyte_exchange_current_density_Xu2019(c_e, c_s_surf, T):
    """
    Exchange-current density for Butler-Volmer reactions between NMC and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Xu, Shanshan, Chen, Kuan-Hung, Dasgupta, Neil P., Siegel, Jason B. and
    Stefanopoulou, Anna G. "Evolution of Dead Lithium Growth in Lithium Metal Batteries:
    Experimentally Validated Model of the Apparent Capacity Loss." Journal of The
    Electrochemical Society 166.14 (2019): A3456-A3463.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    # assuming implicit correction of incorrect units from the paper
    m_ref = 5.76e-11 * constants.F  # (A/m2)(mol/m3)**1.5 - includes ref concentrations
    c_p_max = Parameter("Maximum concentration in positive electrode [mol.m-3]")

    return m_ref * c_e ** 0.5 * c_s_surf ** 0.5 * (c_p_max - c_s_surf) ** 0.5
