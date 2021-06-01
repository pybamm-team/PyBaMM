from pybamm import exp, constants, Parameter


def nca_electrolyte_exchange_current_density_Kim2011(c_e, c_s_surf, T):
    """
    Exchange-current density for Butler-Volmer reactions between NCA and LiPF6 in EC:DMC
    [1].

     References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

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
    i0_ref = 4  # reference exchange current density at 100% SOC
    sto = 0.41  # stochiometry at 100% SOC
    c_s_max = Parameter("Maximum concentration in positive electrode [mol.m-3]")
    c_s_ref = sto * c_s_max  # reference electrode concentration
    c_e_ref = Parameter("Typical electrolyte concentration [mol.m-3]")
    alpha = 0.5  # charge transfer coefficient

    m_ref = i0_ref / (
        c_e_ref ** alpha * (c_s_max - c_s_ref) ** alpha * c_s_ref ** alpha
    )
    E_r = 3e4
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref
        * arrhenius
        * c_e ** alpha
        * c_s_surf ** alpha
        * (c_s_max - c_s_surf) ** alpha
    )
