import pybamm


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
    m_ref = 3.5e-8 * pybamm.constants.F  # (A/m2)(mol/m3) - includes ref concentrations

    return m_ref * c_Li**0.7 * c_e**0.3


def graphite_diffusivity_Ecker2015(sto, T):
    """
    Graphite diffusivity as a function of stoichiometry [1, 2, 3].

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
    sto: :class:`pybamm.Symbol`
        Electrode stoichiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_ref = 8.4e-13 * pybamm.exp(-11.3 * sto) + 8.2e-15
    E_D_s = 3.03e4
    arrhenius = pybamm.exp(-E_D_s / (pybamm.constants.R * T)) * pybamm.exp(
        E_D_s / (pybamm.constants.R * 296)
    )

    return D_ref * arrhenius


def graphite_ocp_Ecker2015(sto):
    """
    Graphite OCP as a function of stoichiometry [1, 2, 3].

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
    sto: :class:`pybamm.Symbol`
        Electrode stoichiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open-circuit potential
    """

    # Graphite electrode from Ecker, Kabitz, Laresgoiti et al.
    # Analytical fit (WebPlotDigitizer + gnuplot)
    a = 0.716502
    b = 369.028
    c = 0.12193
    d = 35.6478
    e = 0.0530947
    g = 0.0169644
    h = 27.1365
    i = 0.312832
    j = 0.0199313
    k = 28.5697
    m = 0.614221
    n = 0.931153
    o = 36.328
    p = 1.10743
    q = 0.140031
    r = 0.0189193
    s = 21.1967
    t = 0.196176

    u_eq = (
        a * pybamm.exp(-b * sto)
        + c * pybamm.exp(-d * (sto - e))
        - r * pybamm.tanh(s * (sto - t))
        - g * pybamm.tanh(h * (sto - i))
        - j * pybamm.tanh(k * (sto - m))
        - n * pybamm.exp(o * (sto - p))
        + q
    )

    return u_eq


def graphite_electrolyte_exchange_current_density_Ecker2015(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

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
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    k_ref = 1.11 * 1e-10

    # multiply by Faraday's constant to get correct units
    m_ref = (
        pybamm.constants.F * k_ref
    )  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 53400

    arrhenius = pybamm.exp(-E_r / (pybamm.constants.R * T)) * pybamm.exp(
        E_r / (pybamm.constants.R * 296.15)
    )

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def plating_exchange_current_density_OKane2020(c_e, c_Li, T):
    """
    Exchange-current density for Li plating reaction [A.m-2].
    References
    ----------
    .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
    Monica Marinescu. "Physical origin of the differential voltage minimum associated
    with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
    167, no. 9 (2020): 090540.
    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_Li : :class:`pybamm.Symbol`
        Plated lithium concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]
    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    k_plating = pybamm.Parameter("Lithium plating kinetic rate constant [m.s-1]")

    return pybamm.constants.F * k_plating * c_e


def stripping_exchange_current_density_OKane2020(c_e, c_Li, T):
    """
    Exchange-current density for Li stripping reaction [A.m-2].

    References
    ----------

    .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
    Monica Marinescu. "Physical origin of the differential voltage minimum associated
    with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
    167, no. 9 (2020): 090540.

    Parameters
    ----------

    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_Li : :class:`pybamm.Symbol`
        Plated lithium concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------

    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    k_plating = pybamm.Parameter("Lithium plating kinetic rate constant [m.s-1]")

    return pybamm.constants.F * k_plating * c_Li


def SEI_limited_dead_lithium_OKane2022(L_sei):
    """
    Decay rate for dead lithium formation [s-1].
    References
    ----------
    .. [1] Simon E. J. O'Kane, Weilong Ai, Ganesh Madabattula, Diega Alonso-Alvarez,
    Robert Timms, Valentin Sulzer, Jaqueline Sophie Edge, Billy Wu, Gregory J. Offer
    and Monica Marinescu. "Lithium-ion battery degradation: how to model it."
    Physical Chemistry: Chemical Physics 24, no. 13 (2022): 7909-7922.
    Parameters
    ----------
    L_sei : :class:`pybamm.Symbol`
        Total SEI thickness [m]
    Returns
    -------
    :class:`pybamm.Symbol`
        Dead lithium decay rate [s-1]
    """

    gamma_0 = pybamm.Parameter("Dead lithium decay constant [s-1]")
    L_sei_0 = pybamm.Parameter("Initial SEI thickness [m]")

    gamma = gamma_0 * L_sei_0 / L_sei

    return gamma


def electrolyte_diffusivity_Ecker2015(c_e, T):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration [1, 2, 3].

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
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    # The diffusivity epends on the electrolyte conductivity
    inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
    sigma_e = pybamm.FunctionParameter("Electrolyte conductivity [S.m-1]", inputs)

    D_c_e = (
        (pybamm.constants.k_b / (pybamm.constants.F * pybamm.constants.q_e))
        * sigma_e
        * T
        / c_e
    )

    return D_c_e


def electrolyte_conductivity_Ecker2015(c_e, T):
    """
    Conductivity of LiPF6 in EC:DMC as a function of ion concentration [1, 2, 3].

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
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    # mol/m^3 to mol/l
    cm = 1e-3 * c_e

    # value at T = 296K
    sigma_e_296 = 0.2667 * cm**3 - 1.2983 * cm**2 + 1.7919 * cm + 0.1726

    # add temperature dependence
    E_k_e = 1.71e4
    C = 296 * pybamm.exp(E_k_e / (pybamm.constants.R * 296))
    sigma_e = C * sigma_e_296 * pybamm.exp(-E_k_e / (pybamm.constants.R * T)) / T

    return sigma_e


# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for a graphite half-cell based on a Kokam SLPB 75106100 cell, from papers

        Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of a
        lithium-ion battery I. determination of parameters." Journal of the
        Electrochemical Society 162.9 (2015): A1836-A1848.

        Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of a
        lithium-ion battery II. Model validation." Journal of The Electrochemical
        Society 162.9 (2015): A1849-A1857.

    The tab placement parameters are taken from measurements in

        Hales, Alastair, et al. "The cell cooling coefficient: a standard to define heat
        rejection from lithium-ion batteries." Journal of The Electrochemical Society
        166.12 (2019): A2383.

    The thermal material properties are for a 5 Ah power pouch cell by Kokam. The data
    are extracted from

        Zhao, Y., et al. "Modeling the effects of thermal gradients induced by tab and
        surface cooling on lithium ion cell performance."" Journal of The
        Electrochemical Society, 165.13 (2018): A3169-A3178.

    Graphite electrode parameters
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    The fits to data for the electrode and electrolyte properties are those provided
    by Dr. Simon O'Kane in the paper:

        Richardson, Giles, et. al. "Generalised single particle models for high-rate
        operation of graded lithium-ion electrodes: Systematic derivation and
        validation." Electrochemica Acta 339 (2020): 135862

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
        # lithium plating
        "Lithium plating kinetic rate constant [m.s-1]": 1e-10,
        "Exchange-current density for plating [A.m-2]"
        "": plating_exchange_current_density_OKane2020,
        "Exchange-current density for stripping [A.m-2]"
        "": stripping_exchange_current_density_OKane2020,
        "Initial plated lithium concentration [mol.m-3]": 0.0,
        "Typical plated lithium concentration [mol.m-3]": 1000.0,
        "Lithium plating transfer coefficient": 0.5,
        "Dead lithium decay constant [s-1]": 1e-06,
        "Dead lithium decay rate [s-1]": SEI_limited_dead_lithium_OKane2022,
        # sei
        "Ratio of lithium moles to SEI moles": 2.0,
        "SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "SEI reaction exchange current density [A.m-2]": 1.5e-07,
        "SEI resistivity [Ohm.m]": 200000.0,
        "SEI solvent diffusivity [m2.s-1]": 2.5e-22,
        "Bulk solvent concentration [mol.m-3]": 2636.0,
        "SEI open-circuit potential [V]": 0.4,
        "SEI electron conductivity [S.m-1]": 8.95e-14,
        "SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
        "Lithium interstitial reference concentration [mol.m-3]": 15.0,
        "Initial SEI thickness [m]": 5e-09,
        "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "EC diffusivity [m2.s-1]": 2e-18,
        "SEI kinetic rate constant [m.s-1]": 1e-12,
        "SEI growth activation energy [J.mol-1]": 0.0,
        "Positive electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        # cell
        "Negative current collector thickness [m]": 1.4e-05,
        "Negative electrode thickness [m]": 0.0007,
        "Positive current collector thickness [m]": 1.4e-05,
        "Positive electrode thickness [m]": 7.4e-05,
        "Separator thickness [m]": 2e-05,
        "Electrode height [m]": 0.101,
        "Electrode width [m]": 0.085,
        "Positive tab width [m]": 0.007,
        "Positive tab centre y-coordinate [m]": 0.0045,
        "Positive tab centre z-coordinate [m]": 0.101,
        "Cell cooling surface area [m2]": 0.0172,
        "Cell volume [m3]": 1.52e-06,
        "Positive current collector conductivity [S.m-1]": 58411000.0,
        "Positive current collector density [kg.m-3]": 8933.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 398.0,
        "Nominal cell capacity [A.h]": 0.15625,
        "Current function [A]": 0.15652,
        "Contact resistance [Ohm]": 0,
        # negative electrode
        "Negative electrode OCP [V]": 0.0,
        "Negative electrode conductivity [S.m-1]": 10776000.0,
        "Negative electrode OCP entropic change [V.K-1]": 0.0,
        "Lithium metal partial molar volume [m3.mol-1]": 1.3e-05,
        "Exchange-current density for lithium metal electrode [A.m-2]"
        "": li_metal_electrolyte_exchange_current_density_Xu2019,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 14.0,
        "Maximum concentration in positive electrode [mol.m-3]": 31920.0,
        "Positive particle diffusivity [m2.s-1]": graphite_diffusivity_Ecker2015,
        "Positive electrode OCP [V]": graphite_ocp_Ecker2015,
        "Positive electrode porosity": 0.329,
        "Positive electrode active material volume fraction": 0.372403,
        "Positive particle radius [m]": 1.37e-05,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.6372789338386007,
        "Positive electrode Bruggeman coefficient (electrode)": 0.0,
        "Positive electrode exchange-current density [A.m-2]"
        "": graphite_electrolyte_exchange_current_density_Ecker2015,
        "Positive electrode density [kg.m-3]": 1555.0,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 1437.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 1.58,
        "Positive electrode OCP entropic change [V.K-1]": 0.0,
        # separator
        "Separator porosity": 0.508,
        "Separator Bruggeman coefficient (electrolyte)": 1.9804586773134945,
        "Separator density [kg.m-3]": 1017.0,
        "Separator specific heat capacity [J.kg-1.K-1]": 1978.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.34,
        # electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number": 0.26,
        "Thermodynamic factor": 1.0,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Ecker2015,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Ecker2015,
        # experiment
        "Reference temperature [K]": 296.15,
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]"
        "": 10.0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Edge heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 0,
        "Upper voltage cut-off [V]": 1.5,
        "Open-circuit voltage at 0% SOC [V]": 0,
        "Open-circuit voltage at 100% SOC [V]": 1.5,
        "Initial concentration in positive electrode [mol.m-3]": 26120.05,
        "Initial temperature [K]": 298.15,
        # citations
        "citations": [
            "Ecker2015i",
            "Ecker2015ii",
            "Zhao2018",
            "Hales2019",
            "Xu2019",
            "Richardson2020",
            "OKane2020",
        ],
    }
