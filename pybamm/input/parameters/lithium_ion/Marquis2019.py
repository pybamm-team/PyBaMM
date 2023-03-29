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


def graphite_mcmb2528_ocp_Dualfoil1998(sto):
    """
    Graphite MCMB 2528 Open-circuit Potential (OCP) as a function of the
    stochiometry. The fit is taken from Dualfoil [1]. Dualfoil states that the data
    was measured by Chris Bogatu at Telcordia and PolyStor materials, 2000. However,
    we could not find any other records of this measurment.

    References
    ----------
    .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html
    """

    u_eq = (
        0.194
        + 1.5 * pybamm.exp(-120.0 * sto)
        + 0.0351 * pybamm.tanh((sto - 0.286) / 0.083)
        - 0.0045 * pybamm.tanh((sto - 0.849) / 0.119)
        - 0.035 * pybamm.tanh((sto - 0.9233) / 0.05)
        - 0.0147 * pybamm.tanh((sto - 0.5) / 0.034)
        - 0.102 * pybamm.tanh((sto - 0.194) / 0.142)
        - 0.022 * pybamm.tanh((sto - 0.9) / 0.0164)
        - 0.011 * pybamm.tanh((sto - 0.124) / 0.0226)
        + 0.0155 * pybamm.tanh((sto - 0.105) / 0.029)
    )

    return u_eq


def graphite_electrolyte_exchange_current_density_Dualfoil1998(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

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
    m_ref = 2 * 10 ** (-5)  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
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


def lico2_diffusivity_Dualfoil1998(sto, T):
    """
    LiCo2 diffusivity as a function of stochiometry, in this case the
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
    D_ref = 1 * 10 ** (-13)
    E_D_s = 18550
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def lico2_ocp_Dualfoil1998(sto):
    """
    Lithium Cobalt Oxide (LiCO2) Open-circuit Potential (OCP) as a a function of the
    stochiometry. The fit is taken from Dualfoil [1]. Dualfoil states that the data
    was measured by Oscar Garcia 2001 using Quallion electrodes for 0.5 < sto < 0.99
    and by Marc Doyle for sto<0.4 (for unstated electrodes). We could not find any
    other records of the Garcia measurements. Doyles fits can be found in his
    thesis [2] but we could not find any other record of his measurments.

    References
    ----------
    .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html
    .. [2] CM Doyle. Design and simulation of lithium rechargeable batteries,
           1995.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    stretch = 1.062
    sto = stretch * sto

    u_eq = (
        2.16216
        + 0.07645 * pybamm.tanh(30.834 - 54.4806 * sto)
        + 2.1581 * pybamm.tanh(52.294 - 50.294 * sto)
        - 0.14169 * pybamm.tanh(11.0923 - 19.8543 * sto)
        + 0.2051 * pybamm.tanh(1.4684 - 5.4888 * sto)
        + 0.2531 * pybamm.tanh((-sto + 0.56478) / 0.1316)
        - 0.02167 * pybamm.tanh((sto - 0.525) / 0.006)
    )

    return u_eq


def lico2_electrolyte_exchange_current_density_Dualfoil1998(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between lico2 and LiPF6 in
    EC:DMC.

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
    m_ref = 6 * 10 ** (-7)  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
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


def electrolyte_diffusivity_Capiglia1999(c_e, T):
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
        Solid diffusivity
    """

    D_c_e = 5.34e-10 * pybamm.exp(-0.65 * c_e / 1000)
    E_D_e = 37040
    arrhenius = pybamm.exp(E_D_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_c_e * arrhenius


def electrolyte_conductivity_Capiglia1999(c_e, T):
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
        Solid diffusivity
    """

    sigma_e = (
        0.0911
        + 1.9101 * (c_e / 1000)
        - 1.052 * (c_e / 1000) ** 2
        + 0.1554 * (c_e / 1000) ** 3
    )

    E_k_e = 34700
    arrhenius = pybamm.exp(E_k_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return sigma_e * arrhenius


# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for a Kokam SLPB78205130H cell, from the paper

        Scott G. Marquis, Valentin Sulzer, Robert Timms, Colin P. Please, and S. Jon
        Chapman. An asymptotic derivation of a single particle model with electrolyte.
        Journal of The Electrochemical Society, 166(15):A3693-A3706, 2019.
        doi:10.1149/2.0341915jes.

    and references therein.

    SEI parameters are example parameters for SEI growth from the papers:

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
    """

    return {
        "chemistry": "lithium_ion",
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
        "Negative electrode thickness [m]": 0.0001,
        "Separator thickness [m]": 2.5e-05,
        "Positive electrode thickness [m]": 0.0001,
        "Positive current collector thickness [m]": 2.5e-05,
        "Electrode height [m]": 0.137,
        "Electrode width [m]": 0.207,
        "Negative tab width [m]": 0.04,
        "Negative tab centre y-coordinate [m]": 0.06,
        "Negative tab centre z-coordinate [m]": 0.137,
        "Positive tab width [m]": 0.04,
        "Positive tab centre y-coordinate [m]": 0.147,
        "Positive tab centre z-coordinate [m]": 0.137,
        "Cell cooling surface area [m2]": 0.0569,
        "Cell volume [m3]": 7.8e-06,
        "Negative current collector conductivity [S.m-1]": 59600000.0,
        "Positive current collector conductivity [S.m-1]": 35500000.0,
        "Negative current collector density [kg.m-3]": 8954.0,
        "Positive current collector density [kg.m-3]": 2707.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        "Nominal cell capacity [A.h]": 0.680616,
        "Current function [A]": 0.680616,
        "Contact resistance [Ohm]": 0,
        # negative electrode
        "Negative electrode conductivity [S.m-1]": 100.0,
        "Maximum concentration in negative electrode [mol.m-3]": 24983.2619938437,
        "Negative electrode diffusivity [m2.s-1]"
        "": graphite_mcmb2528_diffusivity_Dualfoil1998,
        "Negative electrode OCP [V]": graphite_mcmb2528_ocp_Dualfoil1998,
        "Negative electrode porosity": 0.3,
        "Negative electrode active material volume fraction": 0.6,
        "Negative particle radius [m]": 1e-05,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 1.5,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Negative electrode exchange-current density [A.m-2]"
        "": graphite_electrolyte_exchange_current_density_Dualfoil1998,
        "Negative electrode density [kg.m-3]": 1657.0,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        "Negative electrode OCP entropic change [V.K-1]"
        "": graphite_entropic_change_Moura2016,
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 10.0,
        "Maximum concentration in positive electrode [mol.m-3]": 51217.9257309275,
        "Positive electrode diffusivity [m2.s-1]": lico2_diffusivity_Dualfoil1998,
        "Positive electrode OCP [V]": lico2_ocp_Dualfoil1998,
        "Positive electrode porosity": 0.3,
        "Positive electrode active material volume fraction": 0.5,
        "Positive particle radius [m]": 1e-05,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 1.5,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode exchange-current density [A.m-2]"
        "": lico2_electrolyte_exchange_current_density_Dualfoil1998,
        "Positive electrode density [kg.m-3]": 3262.0,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
        "Positive electrode OCP entropic change [V.K-1]"
        "": lico2_entropic_change_Moura2016,
        # separator
        "Separator porosity": 1.0,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 397.0,
        "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.16,
        # electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number": 0.4,
        "Thermodynamic factor": 1.0,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Capiglia1999,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Capiglia1999,
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
        "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 3.105,
        "Upper voltage cut-off [V]": 4.1,
        "Initial concentration in negative electrode [mol.m-3]": 19986.609595075,
        "Initial concentration in positive electrode [mol.m-3]": 30730.7554385565,
        "Initial temperature [K]": 298.15,
        # citations
        "citations": ["Marquis2019"],
    }
