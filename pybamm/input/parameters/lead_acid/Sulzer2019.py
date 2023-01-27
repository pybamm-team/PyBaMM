import pybamm


def lead_ocp_Bode1977(m):
    """
    Dimensional open-circuit voltage in the negative (lead) electrode [V], from [1]_,
    as a function of the molar mass m [mol.kg-1].

    References
    ----------
    .. [1] H Bode. Lead-acid batteries. John Wiley and Sons, Inc., New York, NY, 1977.

    """
    U = (
        -0.294
        - 0.074 * pybamm.log10(m)
        - 0.030 * pybamm.log10(m) ** 2
        - 0.031 * pybamm.log10(m) ** 3
        - 0.012 * pybamm.log10(m) ** 4
    )
    return U


def lead_exchange_current_density_Sulzer2019(c_e, T):
    """
    Dimensional exchange-current density in the negative (lead) electrode, from [1]_

    References
    ----------
    .. [1] V. Sulzer, S. J. Chapman, C. P. Please, D. A. Howey, and C. W. Monroe,
    “Faster lead-acid battery simulations from porous-electrode theory: Part I. Physical
    model.”
    [Journal of the Electrochemical Society](https://doi.org/10.1149/2.0301910jes),
    166(12), 2363 (2019).

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]

    """
    j0_ref = 0.06  # srinivasan2003mathematical
    c_e_typ = pybamm.Parameter("Typical electrolyte concentration [mol.m-3]")
    j0 = j0_ref * (c_e / c_e_typ)

    return j0


def lead_dioxide_ocp_Bode1977(m):
    """
    Dimensional open-circuit voltage in the positive (lead-dioxide) electrode [V],
    from [1]_, as a function of the molar mass m [mol.kg-1].

    References
    ----------
    .. [1] H Bode. Lead-acid batteries. John Wiley and Sons, Inc., New York, NY, 1977.

    """
    U = (
        1.628
        + 0.074 * pybamm.log10(m)
        + 0.033 * pybamm.log10(m) ** 2
        + 0.043 * pybamm.log10(m) ** 3
        + 0.022 * pybamm.log10(m) ** 4
    )
    return U


def lead_dioxide_exchange_current_density_Sulzer2019(c_e, T):
    """
    Dimensional exchange-current density in the positive electrode, from [1]_

    References
    ----------
    .. [1] V. Sulzer, S. J. Chapman, C. P. Please, D. A. Howey, and C. W. Monroe,
    “Faster lead-acid battery simulations from porous-electrode theory: Part I. Physical
    model.”
    [Journal of the Electrochemical Society](https://doi.org/10.1149/2.0301910jes),
    166(12), 2363 (2019).

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]

    """
    c_ox = 0
    c_hy = 0
    param = pybamm.LeadAcidParameters()
    c_w_dim = (1 - c_e * param.V_e - c_ox * param.V_ox - c_hy * param.V_hy) / param.V_w
    c_w_ref = (1 - param.c_e_typ * param.V_e) / param.V_w
    c_w = c_w_dim / c_w_ref

    j0_ref = 0.004  # srinivasan2003mathematical
    j0 = j0_ref * (c_e / param.c_e_typ) ** 2 * c_w

    return j0


def oxygen_exchange_current_density_Sulzer2019(c_e, T):
    """
    Dimensional oxygen exchange-current density in the positive electrode, from [1]_

    References
    ----------
    .. [1] Valentin Sulzer, S. Jon Chapman, Colin P. Please, David A. Howey, and Charles
        W. Monroe. Faster Lead-Acid Battery Simulations from Porous-Electrode Theory:
        Part I. Physical Model. Journal of The Electrochemical Society,
        166(12):A2363-A2371, 2019. doi:10.1149/2.0301910jes.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]

    """
    j0_ref = 2.5e-23  # srinivasan2003mathematical
    c_e_typ = pybamm.Parameter("Typical electrolyte concentration [mol.m-3]")
    j0 = j0_ref * (c_e / c_e_typ)

    return j0


def conductivity_Gu1997(c_e):
    """
    Dimensional conductivity of sulfuric acid [S.m-1], from [1]_ citing [2]_ and
    agreeing with data in [3]_, as a function of the electrolyte concentration
    c_e [mol.m-3].

    References
    ----------
    .. [1] WB Gu, CY Wang, and BY Liaw. Numerical modeling of coupled electrochemical
           and transport processes in lead-acid batteries. Journal of The
           Electrochemical Society, 144(6):2053–2061, 1997.
    .. [2] WH Tiedemann and J Newman. Battery design and optimization. Journal of
           Electrochemical Society, Softbound Proceeding Series, Princeton, New York,
           79(1):23, 1979.
    .. [3] TW Chapman and J Newman. Compilation of selected thermodynamic and transport
           properties of binary electrolytes in aqueous solution. Technical report,
           California Univ., Berkeley. Lawrence Radiation Lab., 1968.

    """
    return c_e * pybamm.exp(6.23 - 1.34e-4 * c_e - 1.61e-8 * c_e**2) * 1e-4


def darken_thermodynamic_factor_Chapman1968(c_e):
    """
    Dimensional Darken thermodynamic factor of sulfuric acid, from data in
    [1, 2]_, as a function of the electrolyte concentration c_e [mol.m-3].

    References
    ----------
    .. [1] TW Chapman and J Newman. Compilation of selected thermodynamic and transport
           properties of binary electrolytes in aqueous solution. Technical report,
           California Univ., Berkeley. Lawrence Radiation Lab., 1968.
    .. [2] KS Pitzer, RN Roy, and LF Silvester. Thermodynamics of electrolytes. 7.
           sulfuric acid. Journal of the American Chemical Society, 99(15):4930–4936,
           1977.

    """
    return 0.49 + 4.1e-4 * c_e


def diffusivity_Gu1997(c_e):
    """
    Dimensional Fickian diffusivity of sulfuric acid [m2.s-1], from [1]_ citing [2]_
    and agreeing with data in [3]_, as a function of the electrolyte concentration
    c_e [mol.m-3].

    References
    ----------
    .. [1] WB Gu, CY Wang, and BY Liaw. Numerical modeling of coupled electrochemical
           and transport processes in lead-acid batteries. Journal of The
           Electrochemical Society, 144(6):2053–2061, 1997.
    .. [2] WH Tiedemann and J Newman. Battery design and optimization. Journal of
           Electrochemical Society, Softbound Proceeding Series, Princeton, New York,
           79(1):23, 1979.
    .. [3] TW Chapman and J Newman. Compilation of selected thermodynamic and transport
           properties of binary electrolytes in aqueous solution. Technical report,
           California Univ., Berkeley. Lawrence Radiation Lab., 1968.

    """
    return (1.75 + 260e-6 * c_e) * 1e-9


def viscosity_Chapman1968(c_e):
    """
    Dimensional viscosity of sulfuric acid [kg.m-1.s-1], from data in [1]_, as a
    function of the electrolyte concentration c_e [mol.m-3].

    References
    ----------
    .. [1] TW Chapman and J Newman. Compilation of selected thermodynamic and transport
           properties of binary electrolytes in aqueous solution. Technical report,
           California Univ., Berkeley. Lawrence Radiation Lab., 1968.

    """
    return 0.89e-3 + 1.11e-7 * c_e + 3.29e-11 * c_e**2


# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for BBOXX lead-acid cells, from the paper

        Valentin Sulzer, S. Jon Chapman, Colin P. Please, David A. Howey, and Charles W.
        Monroe. Faster Lead-Acid Battery Simulations from Porous-Electrode Theory: Part
        I. Physical Model. Journal of The Electrochemical Society, 166(12):A2363-A2371,
        2019. doi:10.1149/2.0301910jes.

    and references therein.
    """

    return {
        "chemistry": "lead_acid",
        # cell
        "Negative current collector thickness [m]": 0.0,
        "Negative electrode thickness [m]": 0.0009,
        "Separator thickness [m]": 0.0015,
        "Positive electrode thickness [m]": 0.00125,
        "Positive current collector thickness [m]": 0.0,
        "Electrode height [m]": 0.114,
        "Electrode width [m]": 0.065,
        "Negative tab width [m]": 0.04,
        "Negative tab centre y-coordinate [m]": 0.06,
        "Negative tab centre z-coordinate [m]": 0.114,
        "Positive tab width [m]": 0.04,
        "Positive tab centre y-coordinate [m]": 0.147,
        "Positive tab centre z-coordinate [m]": 0.114,
        "Cell cooling surface area [m2]": 0.154,
        "Cell volume [m3]": 0.00027,
        "Nominal cell capacity [A.h]": 17.0,
        "Typical current [A]": 1.0,
        "Current function [A]": 1.0,
        "Negative current collector density [kg.m-3]": 11300.0,
        "Positive current collector density [kg.m-3]": 9375.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 130.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 256.0,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 35.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 35.0,
        "Contact resistance [Ohm]": 0,
        # negative electrode
        "Negative electrode conductivity [S.m-1]": 4800000.0,
        "Negative electrode pore size [m]": 1e-07,
        "Maximum porosity of negative electrode": 0.53,
        "Molar volume of lead [m3.mol-1]": 1.82539682539683e-05,
        "Negative electrode volumetric capacity [C.m-3]": 3473000000.0,
        "Negative electrode open-circuit potential [V]": lead_ocp_Bode1977,
        "Negative electrode surface area to volume ratio [m-1]": 2300000.0,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 1.5,
        "Negative electrode morphological parameter": 0.6,
        "Negative electrode capacity [C.m-3]": 3473000000.0,
        "Negative electrode cation signed stoichiometry": 1.0,
        "Negative electrode electrons in reaction": 2.0,
        "Negative electrode exchange-current density [A.m-2]"
        "": lead_exchange_current_density_Sulzer2019,
        "Signed stoichiometry of cations (oxygen reaction)": 4.0,
        "Signed stoichiometry of water (oxygen reaction)": -1.0,
        "Signed stoichiometry of oxygen (oxygen reaction)": 1.0,
        "Electrons in oxygen reaction": 4.0,
        "Negative electrode reference exchange-current density (oxygen) [A.m-2]"
        "": 2.5e-32,
        "Reference oxygen molecule concentration [mol.m-3]": 1000.0,
        "Oxygen reference OCP vs SHE [V]": 1.229,
        "Signed stoichiometry of cations (hydrogen reaction)": 2.0,
        "Signed stoichiometry of hydrogen (hydrogen reaction)": -1.0,
        "Electrons in hydrogen reaction": 2.0,
        "Negative electrode reference exchange-current density (hydrogen) [A.m-2]"
        "": 1.56e-11,
        "Hydrogen reference OCP vs SHE [V]": 0.0,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Negative electrode density [kg.m-3]": 11300.0,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 130.0,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 35.0,
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 80000.0,
        "Positive electrode pore size [m]": 1e-07,
        "Maximum porosity of positive electrode": 0.57,
        "Molar volume of lead-dioxide [m3.mol-1]": 2.54797441364606e-05,
        "Molar volume of lead sulfate [m3.mol-1]": 4.81717011128776e-05,
        "Positive electrode volumetric capacity [C.m-3]": 2745000000.0,
        "Positive electrode open-circuit potential [V]": lead_dioxide_ocp_Bode1977,
        "Positive electrode surface area to volume ratio [m-1]": 23000000.0,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 1.5,
        "Positive electrode morphological parameter": 0.6,
        "Positive electrode capacity [C.m-3]": 2745000000.0,
        "Positive electrode cation signed stoichiometry": 3.0,
        "Positive electrode electrons in reaction": 2.0,
        "Positive electrode exchange-current density [A.m-2]"
        "": lead_dioxide_exchange_current_density_Sulzer2019,
        "Positive electrode oxygen exchange-current density [A.m-2]"
        "": oxygen_exchange_current_density_Sulzer2019,
        "Positive electrode reference exchange-current density (hydrogen) [A.m-2]"
        "": 0.0,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode density [kg.m-3]": 9375.0,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 256.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 35.0,
        # separator
        "Maximum porosity of separator": 0.92,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 1680.0,
        "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.04,
        # electrolyte
        "Typical electrolyte concentration [mol.m-3]": 5650.0,
        "Cation transference number": 0.7,
        "1 + dlnf/dlnc": 1.0,
        "Partial molar volume of water [m3.mol-1]": 1.75e-05,
        "Partial molar volume of anions [m3.mol-1]": 3.15e-05,
        "Partial molar volume of cations [m3.mol-1]": 1.35e-05,
        "Cation stoichiometry": 1.0,
        "Anion stoichiometry": 1.0,
        "Molar mass of water [kg.mol-1]": 0.01801,
        "Molar mass of cations [kg.mol-1]": 0.001,
        "Molar mass of anions [kg.mol-1]": 0.097,
        "Volume change factor": 1.0,
        "Electrolyte conductivity [S.m-1]": conductivity_Gu1997,
        "Darken thermodynamic factor": darken_thermodynamic_factor_Chapman1968,
        "Electrolyte diffusivity [m2.s-1]": diffusivity_Gu1997,
        "Electrolyte viscosity [kg.m-1.s-1]": viscosity_Chapman1968,
        "Oxygen diffusivity [m2.s-1]": 2.1e-09,
        "Typical oxygen concentration [mol.m-3]": 1000.0,
        "Hydrogen diffusivity [m2.s-1]": 4.5e-09,
        "Partial molar volume of oxygen molecules [m3.mol-1]": 3.21e-05,
        "Partial molar volume of hydrogen molecules [m3.mol-1]": 2.31e-05,
        "Molar mass of oxygen molecules [kg.mol-1]": 0.032,
        "Molar mass of hydrogen molecules [kg.mol-1]": 0.002,
        # experiment
        "Reference temperature [K]": 294.85,
        "Maximum temperature [K]": 333.15,
        "Ambient temperature [K]": 294.85,
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]"
        "": 0.0,
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]"
        "": 0.0,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Edge heat transfer coefficient [W.m-2.K-1]": 0.3,
        "Number of electrodes connected in parallel to make a cell": 8.0,
        "Number of cells connected in series to make a battery": 6.0,
        "Lower voltage cut-off [V]": 1.75,
        "Upper voltage cut-off [V]": 2.42,
        "Initial State of Charge": 1.0,
        "Initial oxygen concentration [mol.m-3]": 0.0,
        "Initial temperature [K]": 294.85,
        # citations
        "citations": ["Sulzer2019physical"],
    }
