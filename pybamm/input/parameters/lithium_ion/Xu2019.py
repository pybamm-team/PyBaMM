def li_metal_electrolyte_exchange_current_density_Xu2019(c_e, c_Li, T):
    """
    Exchange-current density for Butler-Volmer reactions between li metal and LiPF6 in
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
    c_Li : :class:`pybamm.Symbol`
        Pure metal lithium concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref = 3.5e-8 * constants.F  # (A/m2)(mol/m3) - includes ref concentrations

    return m_ref * c_Li**0.7 * c_e**0.3

def nmc_ocp_Xu2019(sto):
    """
    Nickel Managanese Cobalt Oxide (NMC) Open Circuit Potential (OCP) as a
    function of the stochiometry, from [1].

    References
    ----------
    .. [1] Xu, Shanshan, Chen, Kuan-Hung, Dasgupta, Neil P., Siegel, Jason B. and
    Stefanopoulou, Anna G. "Evolution of Dead Lithium Growth in Lithium Metal Batteries:
    Experimentally Validated Model of the Apparent Capacity Loss." Journal of The
    Electrochemical Society 166.14 (2019): A3456-A3463.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    # Values from Mohtat2020, might be more accurate
    u_eq = (
        4.3452
        - 1.6518 * sto
        + 1.6225 * (sto**2)
        - 2.0843 * (sto**3)
        + 3.5146 * (sto**4)
        - 2.2166 * (sto**5)
        - 0.5623e-4 * pybamm.exp(109.451 * sto - 100.006)
    )

    # # only valid in range ~(0.25,0.95)
    # u_eq = (
    #     5744.862289 * sto ** 9
    #     - 35520.41099 * sto ** 8
    #     + 95714.29862 * sto ** 7
    #     - 147364.5514 * sto ** 6
    #     + 142718.3782 * sto ** 5
    #     - 90095.81521 * sto ** 4
    #     + 37061.41195 * sto ** 3
    #     - 9578.599274 * sto ** 2
    #     + 1409.309503 * sto
    #     - 85.31153081
    #     - 0.0003 * pybamm.exp(7.657 * sto ** 115)
    # )

    return u_eq

def nmc_electrolyte_exchange_current_density_Xu2019(c_e, c_s_surf, c_s_max, T):
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
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    # assuming implicit correction of incorrect units from the paper
    m_ref = 5.76e-11 * constants.F  # (A/m2)(mol/m3)**1.5 - includes ref concentrations

    return m_ref * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5

def electrolyte_diffusivity_Valoen2005(c_e, T):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration, from [1] (eqn 14)

    References
    ----------
    .. [1] Valøen, Lars Ole, and Jan N. Reimers. "Transport properties of LiPF6-based
    Li-ion battery electrolytes." Journal of The Electrochemical Society 152.5 (2005):
    A882-A891.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Dimensional electrolyte concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Dimensional temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Dimensional electrolyte diffusivity [m2.s-1]
    """
    # mol/m3 to molar
    c_e = c_e / 1000

    T_g = 229 + 5 * c_e
    D_0 = -4.43 - 54 / (T - T_g)
    D_1 = -0.22

    # cm2/s to m2/s
    # note, in the Valoen paper, ln means log10, so its inverse is 10^x
    return (10 ** (D_0 + D_1 * c_e)) * 1e-4

def electrolyte_conductivity_Valoen2005(c_e, T):
    """
    Conductivity of LiPF6 in EC:DMC as a function of ion concentration, from [1]
    (eqn 17)

    References
    ----------
    .. [1] Valøen, Lars Ole, and Jan N. Reimers. "Transport properties of LiPF6-based
    Li-ion battery electrolytes." Journal of The Electrochemical Society 152.5 (2005):
    A882-A891.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Dimensional electrolyte concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Dimensional temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Dimensional electrolyte conductivity [S.m-1]
    """
    # mol/m3 to molar
    c_e = c_e / 1000
    # mS/cm to S/m
    return (1e-3 / 1e-2) * (
        c_e
        * (
            (-10.5 + 0.0740 * T - 6.96e-5 * T**2)
            + c_e * (0.668 - 0.0178 * T + 2.80e-5 * T**2)
            + c_e**2 * (0.494 - 8.86e-4 * T)
        )
        ** 2
    )

def get_parameter_values():
    return {'1 + dlnf/dlnc': 1.0,
 'Ambient temperature [K]': 298.15,
 'Bulk solvent concentration [mol.m-3]': 2636.0,
 'Cation transference number': 0.38,
 'Current function [A]': 0.0024,
 'EC diffusivity [m2.s-1]': 2e-18,
 'EC initial concentration in electrolyte [mol.m-3]': 4541.0,
 'Electrode height [m]': 0.01,
 'Electrode width [m]': 0.0154,
 'Electrolyte conductivity [S.m-1]': electrolyte_conductivity_Valoen2005,
 'Electrolyte diffusivity [m2.s-1]': electrolyte_diffusivity_Valoen2005,
 'Exchange-current density for plating [A.m-2]': li_metal_electrolyte_exchange_current_density_Xu2019,
 'Heat transfer coefficient [W.m-2.K-1]': 10.0,
 'Initial concentration in electrolyte [mol.m-3]': 1000.0,
 'Initial concentration in positive electrode [mol.m-3]': 4631.0,
 'Initial inner SEI thickness [m]': 2.5e-09,
 'Initial outer SEI thickness [m]': 2.5e-09,
 'Initial temperature [K]': 298.15,
 'Inner SEI electron conductivity [S.m-1]': 8.95e-14,
 'Inner SEI lithium interstitial diffusivity [m2.s-1]': 1e-20,
 'Inner SEI open-circuit potential [V]': 0.1,
 'Inner SEI partial molar volume [m3.mol-1]': 9.585e-05,
 'Inner SEI reaction proportion': 0.5,
 'Lithium interstitial reference concentration [mol.m-3]': 15.0,
 'Lower voltage cut-off [V]': 3.5,
 'Maximum concentration in positive electrode [mol.m-3]': 48230.0,
 'Negative electrode OCP [V]': 0.0,
 'Negative electrode OCP entropic change [V.K-1]': 0.0,
 'Negative electrode cation signed stoichiometry': -1.0,
 'Negative electrode charge transfer coefficient': 0.5,
 'Negative electrode conductivity [S.m-1]': 10776000.0,
 'Negative electrode double-layer capacity [F.m-2]': 0.2,
 'Negative electrode electrons in reaction': 1.0,
 'Negative electrode reaction-driven LAM factor [m3.mol-1]': 0.0,
 'Negative electrode thickness [m]': 0.0007,
 'Nominal cell capacity [A.h]': 0.0024,
 'Number of cells connected in series to make a battery': 1.0,
 'Number of electrodes connected in parallel to make a cell': 1.0,
 'Outer SEI open-circuit potential [V]': 0.8,
 'Outer SEI partial molar volume [m3.mol-1]': 9.585e-05,
 'Outer SEI solvent diffusivity [m2.s-1]': 2.5000000000000002e-22,
 'Positive electrode Bruggeman coefficient (electrode)': 1.5,
 'Positive electrode Bruggeman coefficient (electrolyte)': 1.5,
 'Positive electrode OCP [V]': nmc_ocp_Xu2019,
 'Positive electrode OCP entropic change [V.K-1]': 0.0,
 'Positive electrode active material volume fraction': 0.518,
 'Positive electrode cation signed stoichiometry': -1.0,
 'Positive electrode charge transfer coefficient': 0.5,
 'Positive electrode conductivity [S.m-1]': 100.0,
 'Positive electrode diffusivity [m2.s-1]': 1e-14,
 'Positive electrode double-layer capacity [F.m-2]': 0.2,
 'Positive electrode electrons in reaction': 1.0,
 'Positive electrode exchange-current density [A.m-2]': nmc_electrolyte_exchange_current_density_Xu2019,
 'Positive electrode porosity': 0.331,
 'Positive electrode reaction-driven LAM factor [m3.mol-1]': 0.0,
 'Positive electrode thickness [m]': 4.2e-05,
 'Positive particle radius [m]': 5.3e-06,
 'Ratio of lithium moles to SEI moles': 2.0,
 'Reference temperature [K]': 298.15,
 'SEI growth activation energy [J.mol-1]': 0.0,
 'SEI kinetic rate constant [m.s-1]': 1e-12,
 'SEI open-circuit potential [V]': 0.4,
 'SEI reaction exchange current density [A.m-2]': 1.5e-07,
 'SEI resistivity [Ohm.m]': 200000.0,
 'Separator Bruggeman coefficient (electrode)': 1.5,
 'Separator Bruggeman coefficient (electrolyte)': 1.5,
 'Separator porosity': 0.39,
 'Separator thickness [m]': 2.5e-05,
 'Typical current [A]': 0.0024,
 'Typical electrolyte concentration [mol.m-3]': 1000.0,
 'Typical plated lithium concentration [mol.m-3]': 76900.0,
 'Upper voltage cut-off [V]': 4.2}