from pybamm import exp, constants, standard_parameters_lithium_ion


def nco_electrolyte_exchange_current_density_Ecker2015(c_e, c_s_surf, T):
    """
    Exchange-current density for Butler-Volmer reactions between NCO and LiPF6 in
    EC:DMC [1, 2, 3].

    References
    ----------
       .. [1] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery i. determination of parameters." Journal of the
    Electrochemical Society 162.9 (2015): A1836-A1848.
    .. [2] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery ii. model validation." Journal of The Electrochemical
    Society 162.9 (2015): A1849-A1857.
    .. [3] Richardson, Giles, et. al. "Generalised single particle models for
    high-rate operation of graded lithium-ion electrodes: Systematic derivation
    and validation." Electrochemica Acta 339 (2020): 135862

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

    k_ref = 5.196e-11

    # multiply by Faraday's constant to get correct units
    m_ref = constants.F * k_ref  # (A/m2)(mol/m3)**1.5 - includes ref concentrations

    E_r = 4.36e4
    arrhenius = exp(-E_r / (constants.R * T)) * exp(E_r / (constants.R * 296.15))

    c_p_max = standard_parameters_lithium_ion.c_p_max

    return (
        m_ref * arrhenius * c_e ** 0.5 * c_s_surf ** 0.5 * (c_p_max - c_s_surf) ** 0.5
    )
