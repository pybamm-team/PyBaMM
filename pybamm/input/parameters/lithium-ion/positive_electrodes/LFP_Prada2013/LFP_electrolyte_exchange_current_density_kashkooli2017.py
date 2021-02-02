from pybamm import exp, constants, Parameter


def LFP_electrolyte_exchange_current_density_kashkooli2017(c_e, c_s_surf, T):  # , 1
    """
    Exchange-current density for Butler-Volmer reactions between LFP and electrolyte

    References
    ----------
    .. [1] Kashkooli, A. G., Amirfazli, A., Farhad, S., Lee, D. U., Felicelli, S., Park,
    H. W., ... & Chen, Z. (2017). Representative volume element model of lithium-ion
    battery electrodes based on X-ray nano-tomography. Journal of Applied
    Electrochemistry, 47(3), 281-293.

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

    m_ref = 6 * 10 ** (-7)  # (A/m2)(mol/m3)**1.5 - includes ref concentrations
    E_r = 39570
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))
    c_p_max = Parameter("Maximum concentration in positive electrode [mol.m-3]")

    return (
        m_ref * arrhenius * c_e ** 0.5 * c_s_surf ** 0.5 * (c_p_max - c_s_surf) ** 0.5
    )
