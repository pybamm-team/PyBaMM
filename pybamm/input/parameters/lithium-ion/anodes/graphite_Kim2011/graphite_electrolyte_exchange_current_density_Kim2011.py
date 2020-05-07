from pybamm import exp, constants, standard_parameters_lithium_ion


def graphite_electrolyte_exchange_current_density_Kim2011(c_e, c_s_surf, T):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC
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

    i0_ref = 36  # reference exchange current density at 100% SOC
    sto = 0.36  # stochiometry at 100% SOC
    c_s_n_max = standard_parameters_lithium_ion.c_n_max  # max electrode concentration
    c_s_n_ref = sto * c_s_n_max  # reference electrode concentration
    c_e_ref = standard_parameters_lithium_ion.c_e_typ  # ref electrolyte concentration
    alpha = 0.5  # charge transfer coefficient

    m_ref = i0_ref / (
        c_e_ref ** alpha * (c_s_n_max - c_s_n_ref) ** alpha * c_s_n_ref ** alpha
    )

    E_r = 3e4
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref
        * arrhenius
        * c_e ** alpha
        * c_s_surf ** alpha
        * (c_s_n_max - c_s_surf) ** alpha
    )
