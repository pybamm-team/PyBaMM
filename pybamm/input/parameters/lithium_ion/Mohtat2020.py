import pybamm


def graphite_diffusivity_PeymanMPM(sto, T):
    """
    Graphite diffusivity as a function of stochiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from Peyman MPM.

    References
    ----------
    .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_ref = 5.0 * 10 ** (-15)
    E_D_s = 42770
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def graphite_ocp_PeymanMPM(sto):
    """
    Graphite Open-circuit Potential (OCP) as a function of the
    stochiometry. The fit is taken from Peyman MPM [1].

    References
    ----------
    .. [1] Peyman Mohtat et al, MPM (to be submitted)
    """

    u_eq = (
        0.063
        + 0.8 * pybamm.exp(-75 * (sto + 0.001))
        - 0.0120 * pybamm.tanh((sto - 0.127) / 0.016)
        - 0.0118 * pybamm.tanh((sto - 0.155) / 0.016)
        - 0.0035 * pybamm.tanh((sto - 0.220) / 0.020)
        - 0.0095 * pybamm.tanh((sto - 0.190) / 0.013)
        - 0.0145 * pybamm.tanh((sto - 0.490) / 0.020)
        - 0.0800 * pybamm.tanh((sto - 1.030) / 0.055)
    )

    return u_eq


def graphite_electrolyte_exchange_current_density_PeymanMPM(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.
    Check the unit of Reaction rate constant k0 is from Peyman MPM.

    References
    ----------
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

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
    m_ref = 1.061 * 10 ** (-6)  # unit has been converted
    # units are (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 37480
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )


def graphite_entropic_change_PeymanMPM(sto, c_s_max):
    """
    Graphite entropic change in open-circuit potential (OCP) at a temperature of
    298.15K as a function of the stochiometry taken from [1]

    References
    ----------
    .. [1] K.E. Thomas, J. Newman, "Heats of mixing and entropy in porous insertion
           electrode", J. of Power Sources 119 (2003) 844-849

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Stochiometry of material (li-fraction)

    """

    du_dT = 10 ** (-3) * (
        0.28
        - 1.56 * sto
        - 8.92 * sto ** (2)
        + 57.21 * sto ** (3)
        - 110.7 * sto ** (4)
        + 90.71 * sto ** (5)
        - 27.14 * sto ** (6)
    )

    return du_dT


def NMC_diffusivity_PeymanMPM(sto, T):
    """
    NMC diffusivity as a function of stochiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from Peyman MPM.

    References
    ----------
    .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_ref = 8 * 10 ** (-15)
    E_D_s = 18550
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def NMC_ocp_PeymanMPM(sto):
    """
    Nickel Managanese Cobalt Oxide (NMC) Open-circuit Potential (OCP) as a
    function of the stochiometry. The fit is taken from Peyman MPM.

    References
    ----------
    Peyman MPM manuscript (to be submitted)

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    u_eq = (
        4.3452
        - 1.6518 * sto
        + 1.6225 * (sto**2)
        - 2.0843 * (sto**3)
        + 3.5146 * (sto**4)
        - 2.2166 * (sto**5)
        - 0.5623e-4 * pybamm.exp(109.451 * sto - 100.006)
    )

    return u_eq


def NMC_electrolyte_exchange_current_density_PeymanMPM(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between NMC and LiPF6 in
    EC:DMC.

    References
    ----------
    .. Peyman MPM manuscript (to be submitted)

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
    m_ref = 4.824 * 10 ** (-6)  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 39570
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )


def NMC_entropic_change_PeymanMPM(sto, c_s_max):
    """
    Nickel Manganese Cobalt (NMC) entropic change in open-circuit potential (OCP) at
    a temperature of 298.15K as a function of the OCP. The fit is taken from [1].

    References
    ----------
    .. [1] W. Le, I. Belharouak, D. Vissers, K. Amine, "In situ thermal study of
    li1+ x [ni1/ 3co1/ 3mn1/ 3] 1- x o2 using isothermal micro-clorimetric
    techniques",
    J. of the Electrochemical Society 153 (11) (2006) A2147–A2151.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Stochiometry of material (li-fraction)

    """

    # Since the equation uses the OCP at each stoichiometry as input,
    # we need OCP function here

    u_eq = (
        4.3452
        - 1.6518 * sto
        + 1.6225 * sto**2
        - 2.0843 * sto**3
        + 3.5146 * sto**4
        - 0.5623 * 10 ** (-4) * pybamm.exp(109.451 * sto - 100.006)
    )

    du_dT = (
        -800 + 779 * u_eq - 284 * u_eq**2 + 46 * u_eq**3 - 2.8 * u_eq**4
    ) * 10 ** (-3)

    return du_dT


def electrolyte_diffusivity_PeymanMPM(c_e, T):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration. The original data
    is from [1]. The fit from Dualfoil [2].

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
        Electrolyte diffusivity
    """

    D_c_e = 5.35 * 10 ** (-10)
    E_D_e = 37040
    arrhenius = pybamm.exp(E_D_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_c_e * arrhenius


def electrolyte_conductivity_PeymanMPM(c_e, T):
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
        Electrolyte conductivity
    """

    sigma_e = 1.3
    E_k_e = 34700
    arrhenius = pybamm.exp(E_k_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return sigma_e * arrhenius


# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for a graphite/NMC532 pouch cell from the paper

        Peyman Mohtat, Suhak Lee, Valentin Sulzer, Jason B. Siegel, and Anna G.
        Stefanopoulou. Differential Expansion and Voltage Model for Li-ion Batteries at
        Practical Charging Rates. Journal of The Electrochemical Society,
        167(11):110561, 2020. doi:10.1149/1945-7111/aba5d1.

    and references therein.

    Some example parameters for SEI growth from the papers:

        Ramadass, P., Haran, B., Gomadam, P. M., White, R., & Popov, B. N. (2004).
        Development of first principles capacity fade model for Li-ion cells. Journal of
        the Electrochemical Society, 151(2), A196-A203.

        Ploehn, H. J., Ramadass, P., & White, R. E. (2004). Solvent diffusion model for
        aging of lithium-ion battery cells. Journal of The Electrochemical Society,
        151(3), A456-A462.

        Single, F., Latz, A., & Horstmann, B. (2018). Identifying the mechanism of
        continued growth of the solid-electrolyte interphase. ChemSusChem, 11(12),
        1950-1955.

        Safari, M., Morcrette, M., Teyssot, A., & Delacour, C. (2009). Multimodal
        Physics- Based Aging Model for Life Prediction of Li-Ion Batteries. Journal of
        The Electrochemical Society, 156(3),

        Yang, X., Leng, Y., Zhang, G., Ge, S., Wang, C. (2017). Modeling of lithium
        plating induced aging of lithium-ion batteries: Transition from linear to
        nonlinear aging. Journal of Power Sources, 360, 28-40.

    Note: this parameter set does not claim to be representative of the true parameter
    values. Instead these are parameter values that were used to fit SEI models to
    observed experimental data in the referenced papers.

    SEI parameters
    ^^^^^^^^^^^^^^

    Parameters for lithium plating are from the paper:

        Yang, X., Leng, Y., Zhang, G., Ge, S., Wang, C. (2017). Modeling of lithium
        plating induced aging of lithium-ion batteries: Transition from linear to
        nonlinear aging. Journal of Power Sources, 360, 28-40.

    Note: this parameter set does not claim to be representative of the true parameter
    values. Instead these are parameter values that were used to fit SEI models to
    observed experimental data in the referenced papers.
    """

    return {
        "chemistry": "lithium_ion",
        # lithium plating
        "Lithium metal partial molar volume [m3.mol-1]": 1.3e-05,
        "Exchange-current density for plating [A.m-2]": 0.001,
        "Initial plated lithium concentration [mol.m-3]": 0.0,
        "Typical plated lithium concentration [mol.m-3]": 1000.0,
        "Lithium plating transfer coefficient": 0.7,
        # sei
        "Ratio of lithium moles to SEI moles": 2.0,
        "Inner SEI reaction proportion": 0.5,
        "Inner SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Outer SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "SEI reaction exchange current density [A.m-2]": 1.5e-07,
        "SEI resistivity [Ohm.m]": 200000.0,
        "Outer SEI solvent diffusivity [m2.s-1]": 2.5000000000000002e-22,
        "Bulk solvent concentration [mol.m-3]": 2636.0,
        "Inner SEI open-circuit potential [V]": 0.1,
        "Outer SEI open-circuit potential [V]": 0.8,
        "Inner SEI electron conductivity [S.m-1]": 8.95e-14,
        "Inner SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
        "Lithium interstitial reference concentration [mol.m-3]": 15.0,
        "Initial inner SEI thickness [m]": 2.5e-09,
        "Initial outer SEI thickness [m]": 2.5e-09,
        "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "EC diffusivity [m2.s-1]": 2e-18,
        "SEI kinetic rate constant [m.s-1]": 1e-12,
        "SEI open-circuit potential [V]": 0.4,
        "SEI growth activation energy [J.mol-1]": 0.0,
        "Negative electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        "Positive electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        # cell
        "Negative current collector thickness [m]": 2.5e-05,
        "Negative electrode thickness [m]": 6.2e-05,
        "Separator thickness [m]": 1.2e-05,
        "Positive electrode thickness [m]": 6.7e-05,
        "Positive current collector thickness [m]": 2.5e-05,
        "Electrode height [m]": 1.0,
        "Electrode width [m]": 0.205,
        "Cell cooling surface area [m2]": 0.41,
        "Cell volume [m3]": 3.92e-05,
        "Negative current collector conductivity [S.m-1]": 59600000.0,
        "Positive current collector conductivity [S.m-1]": 35500000.0,
        "Negative current collector density [kg.m-3]": 8954.0,
        "Positive current collector density [kg.m-3]": 2707.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        "Nominal cell capacity [A.h]": 5.0,
        "Current function [A]": 5.0,
        "Contact resistance [Ohm]": 0,
        # negative electrode
        "Negative electrode conductivity [S.m-1]": 100.0,
        "Maximum concentration in negative electrode [mol.m-3]": 28746.0,
        "Negative electrode diffusivity [m2.s-1]": graphite_diffusivity_PeymanMPM,
        "Negative electrode OCP [V]": graphite_ocp_PeymanMPM,
        "Negative electrode porosity": 0.3,
        "Negative electrode active material volume fraction": 0.61,
        "Negative particle radius [m]": 2.5e-06,
        "Negative electrode Bruggeman coefficient (electrode)": 1.5,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode transport efficiency": 0.16,
        "Negative electrode cation signed stoichiometry": -1.0,
        "Negative electrode electrons in reaction": 1.0,
        "Negative electrode reference exchange-current density [A.m-2(m3.mol)1.5]"
        "": 1.061e-06,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Negative electrode exchange-current density [A.m-2]"
        "": graphite_electrolyte_exchange_current_density_PeymanMPM,
        "Negative electrode density [kg.m-3]": 3100.0,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 1100.0,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        "Negative electrode OCP entropic change [V.K-1]"
        "": graphite_entropic_change_PeymanMPM,
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 100.0,
        "Maximum concentration in positive electrode [mol.m-3]": 35380.0,
        "Positive electrode diffusivity [m2.s-1]": NMC_diffusivity_PeymanMPM,
        "Positive electrode OCP [V]": NMC_ocp_PeymanMPM,
        "Positive electrode porosity": 0.3,
        "Positive electrode active material volume fraction": 0.445,
        "Positive particle radius [m]": 3.5e-06,
        "Positive electrode Bruggeman coefficient (electrode)": 1.5,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode transport efficiency": 0.16,
        "Positive electrode cation signed stoichiometry": -1.0,
        "Positive electrode electrons in reaction": 1.0,
        "Positive electrode reference exchange-current density [A.m-2(m3.mol)1.5]"
        "": 4.824e-06,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode exchange-current density [A.m-2]"
        "": NMC_electrolyte_exchange_current_density_PeymanMPM,
        "Positive electrode density [kg.m-3]": 3100.0,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 1100.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
        "Positive electrode OCP entropic change [V.K-1]"
        "": NMC_entropic_change_PeymanMPM,
        # separator
        "Separator porosity": 0.4,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 397.0,
        "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.16,
        "Separator transport efficiency ": 0.25,
        # electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number": 0.38,
        "1 + dlnf/dlnc": 1.0,
        "Typical lithium ion diffusivity [m2.s-1]": 5.34e-10,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_PeymanMPM,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_PeymanMPM,
        # experiment
        "Reference temperature [K]": 298.15,
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]"
        "": 0.0,
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]"
        "": 0.0,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Edge heat transfer coefficient [W.m-2.K-1]": 5.0,
        "Total heat transfer coefficient [W.m-2.K-1]": 5.0,
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 2.8,
        "Upper voltage cut-off [V]": 4.2,
        "Initial concentration in negative electrode [mol.m-3]": 48.8682,
        "Initial concentration in positive electrode [mol.m-3]": 31513.0,
        "Initial temperature [K]": 298.15,
        # citations
        "citations": ["Mohtat2020"],
    }
