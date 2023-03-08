import pybamm


def graphite_mcmb2528_diffusivity_Dualfoil1998(sto, T):
    """
    Graphite MCMB 2528 diffusivity as a function of stochiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from Dualfoil [1].

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

    D_ref = 3.9 * 10 ** (-14)
    E_D_s = 42770
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def graphite_ocp_Ramadass2004(sto):
    """
    Graphite Open-circuit Potential (OCP) as a function of the
    stochiometry (theta?). The fit is taken from Ramadass 2004.

    References
    ----------
    .. [1] P. Ramadass, Bala Haran, Parthasarathy M. Gomadam, Ralph White, and Branko
    N. Popov. "Development of First Principles Capacity Fade Model for Li-Ion Cells."
    (2004)
    """

    u_eq = (
        0.7222
        + 0.1387 * sto
        + 0.029 * (sto**0.5)
        - 0.0172 / sto
        + 0.0019 / (sto**1.5)
        + 0.2808 * pybamm.exp(0.9 - 15 * sto)
        - 0.7984 * pybamm.exp(0.4465 * sto - 0.4108)
    )

    return u_eq


def graphite_electrolyte_exchange_current_density_Ramadass2004(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] P. Ramadass, Bala Haran, Parthasarathy M. Gomadam, Ralph White, and Branko
    N. Popov. "Development of First Principles Capacity Fade Model for Li-Ion Cells."
    (2004)

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
    m_ref = 4.854 * 10 ** (-6)  # (A/m2)(m3/mol)**1.5
    E_r = 37480
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )


def graphite_entropic_change_Moura2016(sto, c_s_max):
    """
    Graphite entropic change in open-circuit potential (OCP) at a temperature of
    298.15K as a function of the stochiometry taken from Scott Moura's FastDFN code
    [1].

    References
    ----------
    .. [1] https://github.com/scott-moura/fastDFN

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Stochiometry of material (li-fraction)

    """
    du_dT = (
        -1.5 * (120.0 / c_s_max) * pybamm.exp(-120 * sto)
        + (0.0351 / (0.083 * c_s_max)) * ((pybamm.cosh((sto - 0.286) / 0.083)) ** (-2))
        - (0.0045 / (0.119 * c_s_max)) * ((pybamm.cosh((sto - 0.849) / 0.119)) ** (-2))
        - (0.035 / (0.05 * c_s_max)) * ((pybamm.cosh((sto - 0.9233) / 0.05)) ** (-2))
        - (0.0147 / (0.034 * c_s_max)) * ((pybamm.cosh((sto - 0.5) / 0.034)) ** (-2))
        - (0.102 / (0.142 * c_s_max)) * ((pybamm.cosh((sto - 0.194) / 0.142)) ** (-2))
        - (0.022 / (0.0164 * c_s_max)) * ((pybamm.cosh((sto - 0.9) / 0.0164)) ** (-2))
        - (0.011 / (0.0226 * c_s_max)) * ((pybamm.cosh((sto - 0.124) / 0.0226)) ** (-2))
        + (0.0155 / (0.029 * c_s_max)) * ((pybamm.cosh((sto - 0.105) / 0.029)) ** (-2))
    )

    return du_dT


def lico2_diffusivity_Ramadass2004(sto, T):
    """
    LiCo2 diffusivity as a function of stochiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from Ramadass 2004.

    References
    ----------
    .. [1] P. Ramadass, Bala Haran, Parthasarathy M. Gomadam, Ralph White, and Branko
    N. Popov. "Development of First Principles Capacity Fade Model for Li-Ion Cells."
    (2004)

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
    D_ref = 1 * 10 ** (-14)
    E_D_s = 18550
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def lico2_ocp_Ramadass2004(sto):
    """
    Lithium Cobalt Oxide (LiCO2) Open-circuit Potential (OCP) as a a function of the
    stochiometry. The fit is taken from Ramadass 2004. Stretch is considered the
    overhang area negative electrode / area positive electrode, in Ramadass 2002.

    References
    ----------
    .. [1] P. Ramadass, Bala Haran, Parthasarathy M. Gomadam, Ralph White, and Branko
    N. Popov. "Development of First Principles Capacity Fade Model for Li-Ion Cells."
    (2004)

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    stretch = 1.13
    sto = stretch * sto

    u_eq = (
        -4.656
        + 88.669 * (sto**2)
        - 401.119 * (sto**4)
        + 342.909 * (sto**6)
        - 462.471 * (sto**8)
        + 433.434 * (sto**10)
    ) / (
        -1
        + 18.933 * (sto**2)
        - 79.532 * (sto**4)
        + 37.311 * (sto**6)
        - 73.083 * (sto**8)
        + 95.96 * (sto**10)
    )

    return u_eq


def lico2_electrolyte_exchange_current_density_Ramadass2004(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between lico2 and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] P. Ramadass, Bala Haran, Parthasarathy M. Gomadam, Ralph White, and Branko
    N. Popov. "Development of First Principles Capacity Fade Model for Li-Ion Cells."
    (2004)

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
    m_ref = 2.252 * 10 ** (-6)  # (A/m2)(m3/mol)**1.5
    E_r = 39570
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )


def lico2_entropic_change_Moura2016(sto, c_s_max):
    """
    Lithium Cobalt Oxide (LiCO2) entropic change in open-circuit potential (OCP) at
    a temperature of 298.15K as a function of the stochiometry. The fit is taken
    from Scott Moura's FastDFN code [1].

    References
    ----------
    .. [1] https://github.com/scott-moura/fastDFN

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Stochiometry of material (li-fraction)
    """
    # Since the equation for LiCo2 from this ref. has the stretch factor,
    # should this too? If not, the "bumps" in the OCV don't line up.
    stretch = 1.062
    sto = stretch * sto

    du_dT = (
        0.07645
        * (-54.4806 / c_s_max)
        * ((1.0 / pybamm.cosh(30.834 - 54.4806 * sto)) ** 2)
        + 2.1581 * (-50.294 / c_s_max) * ((pybamm.cosh(52.294 - 50.294 * sto)) ** (-2))
        + 0.14169
        * (19.854 / c_s_max)
        * ((pybamm.cosh(11.0923 - 19.8543 * sto)) ** (-2))
        - 0.2051 * (5.4888 / c_s_max) * ((pybamm.cosh(1.4684 - 5.4888 * sto)) ** (-2))
        - (0.2531 / 0.1316 / c_s_max)
        * ((pybamm.cosh((-sto + 0.56478) / 0.1316)) ** (-2))
        - (0.02167 / 0.006 / c_s_max) * ((pybamm.cosh((sto - 0.525) / 0.006)) ** (-2))
    )

    return du_dT


def electrolyte_diffusivity_Ramadass2004(c_e, T):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration.

    References
    ----------
    .. [1] P. Ramadass, Bala Haran, Parthasarathy M. Gomadam, Ralph White, and Branko
    N. Popov. "Development of First Principles Capacity Fade Model for Li-Ion Cells."
    (2004)

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

    D_c_e = 7.5e-10
    E_D_e = 37040
    arrhenius = pybamm.exp(E_D_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_c_e * arrhenius


def electrolyte_conductivity_Ramadass2004(c_e, T):
    """
    Conductivity of LiPF6 in EC:DMC as a function of ion concentration.
    Concentration should be in dm3 in the function.

    References
    ----------
    .. [1] P. Ramadass, Bala Haran, Parthasarathy M. Gomadam, Ralph White, and Branko
    N. Popov. "Development of First Principles Capacity Fade Model for Li-Ion Cells."
    (2004)

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
    # mol.m-3 to mol.dm-3, original function is likely in mS/cm
    # The function is not in Arora 2000 as reported in Ramadass 2004

    cm = 1e-6 * c_e  # here it should be only 1e-3

    sigma_e = (
        4.1253 * (10 ** (-4))
        + 5.007 * cm
        - 4.7212 * (10**3) * (cm**2)
        + 1.5094 * (10**6) * (cm**3)
        - 1.6018 * (10**8) * (cm**4)
    ) * 1e3  # and here there should not be an exponent

    E_k_e = 34700
    arrhenius = pybamm.exp(E_k_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return sigma_e * arrhenius


# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Ramadass2004 parameter set. This is a bit of a Frankenstein parameter set and should
    be used with caution.

    Parameters for a graphite negative electrode, Lithium Cobalt Oxide positive
    electrode, and LiPF6 electrolyte are from the papers

        Scott G. Marquis, Valentin Sulzer, Robert Timms, Colin P. Please, and S. Jon
        Chapman. An asymptotic derivation of a single particle model with electrolyte.
        Journal of The Electrochemical Society, 166(15):A3693-A3706, 2019.
        doi:10.1149/2.0341915jes.

        P Ramadass, Bala Haran, Parthasarathy M Gomadam, Ralph White, and Branko N
        Popov. Development of first principles capacity fade model for li-ion cells.
        Journal of the Electrochemical Society, 151(2):A196, 2004.
        doi:10.1149/1.1634273.

    and references therein.

    Parameters for the separator are from the papers

        Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of a
        lithium-ion battery i. determination of parameters." Journal of the
        Electrochemical Society 162.9 (2015): A1836-A1848.

    The thermal material properties are for a 5 Ah power pouch cell by Kokam. The data
    are extracted from

        Zhao, Y., et al. "Modeling the effects of thermal gradients induced by tab and
        surface cooling on lithium ion cell performance."" Journal of The
        Electrochemical Society, 165.13 (2018): A3169-A3178. # Lithium Cobalt Oxide
        positive electrode parameters

    Parameters for SEI growth are from the papers

        Ramadass, P., Haran, B., Gomadam, P. M., White, R., & Popov, B. N. (2004).
        Development of first principles capacity fade model for Li-ion cells. Journal of
        the Electrochemical Society, 151(2), A196-A203.

        Safari, M., Morcrette, M., Teyssot, A., & Delacour, C. (2009). Multimodal
        Physics- Based Aging Model for Life Prediction of Li-Ion Batteries. Journal of
        The Electrochemical Society, 156(3),

    Note: Ramadass 2004 has mistakes in units and values of SEI parameters, corrected by
    Safari 2009.
    """

    return {
        "chemistry": "lithium_ion",
        # sei
        "Ratio of lithium moles to SEI moles": 2.0,
        "Inner SEI reaction proportion": 0.5,
        "Inner SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Outer SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "SEI reaction exchange current density [A.m-2]": 1.5e-06,
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
        "SEI open-circuit potential [V]": 0.0,
        "SEI growth activation energy [J.mol-1]": 0.0,
        # cell
        "Negative current collector thickness [m]": 1.7e-05,
        "Negative electrode thickness [m]": 8.8e-05,
        "Separator thickness [m]": 2.5e-05,
        "Positive electrode thickness [m]": 8e-05,
        "Positive current collector thickness [m]": 2.3e-05,
        "Electrode height [m]": 0.057,
        "Electrode width [m]": 1.060692,
        "Negative tab width [m]": 0.04,
        "Negative tab centre y-coordinate [m]": 0.06,
        "Negative tab centre z-coordinate [m]": 0.137,
        "Positive tab width [m]": 0.04,
        "Positive tab centre y-coordinate [m]": 0.147,
        "Positive tab centre z-coordinate [m]": 0.137,
        "Negative current collector conductivity [S.m-1]": 59600000.0,
        "Positive current collector conductivity [S.m-1]": 35500000.0,
        "Negative current collector density [kg.m-3]": 8954.0,
        "Positive current collector density [kg.m-3]": 2707.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        "Nominal cell capacity [A.h]": 1.0,
        "Current function [A]": 1.0,
        "Contact resistance [Ohm]": 0,
        # negative electrode
        "Negative electrode conductivity [S.m-1]": 100.0,
        "Maximum concentration in negative electrode [mol.m-3]": 30555.0,
        "Negative electrode diffusivity [m2.s-1]"
        "": graphite_mcmb2528_diffusivity_Dualfoil1998,
        "Negative electrode OCP [V]": graphite_ocp_Ramadass2004,
        "Negative electrode porosity": 0.485,
        "Negative electrode active material volume fraction": 0.49,
        "Negative particle radius [m]": 2e-06,
        "Negative electrode Bruggeman coefficient (electrolyte)": 4.0,
        "Negative electrode Bruggeman coefficient (electrode)": 4.0,
        "Negative electrode cation signed stoichiometry": -1.0,
        "Negative electrode electrons in reaction": 1.0,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Negative electrode exchange-current density [A.m-2]"
        "": graphite_electrolyte_exchange_current_density_Ramadass2004,
        "Negative electrode density [kg.m-3]": 1657.0,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        "Negative electrode OCP entropic change [V.K-1]"
        "": graphite_entropic_change_Moura2016,
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 100.0,
        "Maximum concentration in positive electrode [mol.m-3]": 51555.0,
        "Positive electrode diffusivity [m2.s-1]": lico2_diffusivity_Ramadass2004,
        "Positive electrode OCP [V]": lico2_ocp_Ramadass2004,
        "Positive electrode porosity": 0.385,
        "Positive electrode active material volume fraction": 0.59,
        "Positive particle radius [m]": 2e-06,
        "Positive electrode Bruggeman coefficient (electrolyte)": 4.0,
        "Positive electrode Bruggeman coefficient (electrode)": 4.0,
        "Positive electrode cation signed stoichiometry": -1.0,
        "Positive electrode electrons in reaction": 1.0,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode exchange-current density [A.m-2]"
        "": lico2_electrolyte_exchange_current_density_Ramadass2004,
        "Positive electrode density [kg.m-3]": 3262.0,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
        "Positive electrode OCP entropic change [V.K-1]"
        "": lico2_entropic_change_Moura2016,
        # separator
        "Separator porosity": 0.508,
        "Separator Bruggeman coefficient (electrolyte)": 1.9804586773134945,
        "Separator density [kg.m-3]": 1017.0,
        "Separator specific heat capacity [J.kg-1.K-1]": 1978.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.34,
        # electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number": 0.363,
        "1 + dlnf/dlnc": 1.0,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Ramadass2004,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Ramadass2004,
        # experiment
        "Reference temperature [K]": 298.15,
        "Ambient temperature [K]": 298.15,
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]"
        "": 0.0,
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]"
        "": 0.0,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Edge heat transfer coefficient [W.m-2.K-1]": 0.3,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 2.8,
        "Upper voltage cut-off [V]": 4.2,
        "Initial concentration in negative electrode [mol.m-3]": 22610.7,
        "Initial concentration in positive electrode [mol.m-3]": 25777.5,
        "Initial temperature [K]": 298.15,
        # citations
        "citations": ["Ramadass2004"],
    }
