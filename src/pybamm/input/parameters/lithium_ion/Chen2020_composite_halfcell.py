import pybamm
import os
import numpy as np


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


def graphite_LGM50_electrolyte_exchange_current_density_Chen2020(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

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
    m_ref = 6.48e-7  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 35000
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def silicon_ocp_lithiation_Mark2016(sto):
    """
    silicon Open-circuit Potential (OCP) as a a function of the
    stoichiometry. The fit is taken from the Enertech cell [1], which is only accurate
    for 0 < sto < 1.

    References
    ----------
    .. [1] Verbrugge M, Baker D, Xiao X. Formulation for the treatment of multiple
    electrochemical reactions and associated speciation for the Lithium-Silicon
    electrode[J]. Journal of The Electrochemical Society, 2015, 163(2): A262.

    Parameters
    ----------
    sto: double
       stoichiometry of material (li-fraction)

    Returns
    -------
    :class:`pybamm.Symbol`
        OCP [V]
    """
    p1 = -96.63
    p2 = 372.6
    p3 = -587.6
    p4 = 489.9
    p5 = -232.8
    p6 = 62.99
    p7 = -9.286
    p8 = 0.8633

    U_lithiation = (
        p1 * sto**7
        + p2 * sto**6
        + p3 * sto**5
        + p4 * sto**4
        + p5 * sto**3
        + p6 * sto**2
        + p7 * sto
        + p8
    )
    return U_lithiation


def silicon_ocp_delithiation_Mark2016(sto):
    """
    silicon Open-circuit Potential (OCP) as a a function of the
    stoichiometry. The fit is taken from the Enertech cell [1], which is only accurate
    for 0 < sto < 1.

    References
    ----------
    .. [1] Verbrugge M, Baker D, Xiao X. Formulation for the treatment of multiple
    electrochemical reactions and associated speciation for the Lithium-Silicon
    electrode[J]. Journal of The Electrochemical Society, 2015, 163(2): A262.

    Parameters
    ----------
    sto: double
       stoichiometry of material (li-fraction)

    Returns
    -------
    :class:`pybamm.Symbol`
        OCP [V]
    """
    p1 = -51.02
    p2 = 161.3
    p3 = -205.7
    p4 = 140.2
    p5 = -58.76
    p6 = 16.87
    p7 = -3.792
    p8 = 0.9937

    U_delithiation = (
        p1 * sto**7
        + p2 * sto**6
        + p3 * sto**5
        + p4 * sto**4
        + p5 * sto**3
        + p6 * sto**2
        + p7 * sto
        + p8
    )
    return U_delithiation


def silicon_LGM50_electrolyte_exchange_current_density_Chen2020(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between silicon and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

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

    m_ref = (
        6.48e-7 * 28700 / 278000
    )  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 35000
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def electrolyte_diffusivity_Nyman2008(c_e, T):
    """
    Diffusivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1]

    References
    ----------
    .. [1] A. Nyman, M. Behm, and G. Lindbergh, "Electrochemical characterisation and
    modelling of the mass transport phenomena in LiPF6-EC-EMC electrolyte,"
    Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.

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

    D_c_e = 8.794e-11 * (c_e / 1000) ** 2 - 3.972e-10 * (c_e / 1000) + 4.862e-10

    # Nyman et al. (2008) does not provide temperature dependence

    return D_c_e


def electrolyte_conductivity_Nyman2008(c_e, T):
    """
    Conductivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1].

    References
    ----------
    .. [1] A. Nyman, M. Behm, and G. Lindbergh, "Electrochemical characterisation and
    modelling of the mass transport phenomena in LiPF6-EC-EMC electrolyte,"
    Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.

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
        0.1297 * (c_e / 1000) ** 3 - 2.51 * (c_e / 1000) ** 1.5 + 3.329 * (c_e / 1000)
    )

    # Nyman et al. (2008) does not provide temperature dependence

    return sigma_e


# Load data in the appropriate format
path, _ = os.path.split(os.path.abspath(__file__))
graphite_ocp_Enertech_Ai2020_data = pybamm.parameters.process_1D_data(
    "graphite_ocp_Enertech_Ai2020.csv", path=path
)


def graphite_ocp_Enertech_Ai2020(sto):
    name, (x, y) = graphite_ocp_Enertech_Ai2020_data
    return pybamm.Interpolant(x, y, sto, name=name, interpolator="cubic")


# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for a composite graphite/silicon negative electrode, from the paper
    :footcite:t:`Ai2022`, based on the paper :footcite:t:`Chen2020`, and references
    therein.

    SEI parameters are example parameters for composite SEI on silicon/graphite. Both
    phases use the same values, from the paper :footcite:t:`Yang2017`
    """

    return {
        "chemistry": "lithium_ion",
        # sei
        "Ratio of lithium moles to negative SEI moles": 2.0,
        "Negative inner SEI reaction proportion": 0.5,
        "Negative inner SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Negative outer SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Negative SEI reaction exchange current density [A.m-2]": 1.5e-07,
        "Negative SEI resistivity [Ohm.m]": 200000.0,
        "Negative outer SEI solvent diffusivity [m2.s-1]": 2.5000000000000002e-22,
        "Bulk solvent concentration for negative SEI [mol.m-3]": 2636.0,
        "Negative inner SEI open-circuit potential [V]": 0.1,
        "Negative outer SEI open-circuit potential [V]": 0.8,
        "Negative inner SEI electron conductivity [S.m-1]": 8.95e-14,
        "Negative inner SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
        "Negative lithium interstitial reference concentration [mol.m-3]": 15.0,
        "Initial negative inner SEI thickness [m]": 2.5e-09,
        "Initial negative outer SEI thickness [m]": 2.5e-09,
        "Negative EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "EC diffusivity through negative SEI [m2.s-1]": 2e-18,
        "Negative SEI kinetic rate constant [m.s-1]": 1e-12,
        "Negative SEI open-circuit potential [V]": 0.4,
        "Negative SEI growth activation energy [J.mol-1]": 0.0,
        "Primary: Ratio of lithium moles to positive SEI moles": 2.0,
        "Primary: Positive inner SEI reaction proportion": 0.5,
        "Primary: Positive inner SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Primary: Posituve outer SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Primary: Positive SEI reaction exchange current density [A.m-2]": 1.5e-07,
        "Primary: Positive SEI resistivity [Ohm.m]": 200000.0,
        "Primary: Positive outer SEI solvent diffusivity [m2.s-1]"
        "": 2.5000000000000002e-22,
        "Primary: Bulk solvent concentration for positive SEI [mol.m-3]": 2636.0,
        "Primary: Positive inner SEI open-circuit potential [V]": 0.1,
        "Primary: Positive outer SEI open-circuit potential [V]": 0.8,
        "Primary: Positive inner SEI electron conductivity [S.m-1]": 8.95e-14,
        "Primary: Positive inner SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
        "Primary: Positive lithium interstitial reference concentration [mol.m-3]"
        "": 15.0,
        "Primary: Initial positive inner SEI thickness [m]": 2.5e-09,
        "Primary: Initial positive outer SEI thickness [m]": 2.5e-09,
        "Primary: Positive EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "Primary: EC diffusivity through positive SEI [m2.s-1]": 2e-18,
        "Primary: Positive SEI kinetic rate constant [m.s-1]": 1e-12,
        "Primary: Positive SEI open-circuit potential [V]": 0.4,
        "Primary: Positive SEI growth activation energy [J.mol-1]": 0.0,
        "Secondary: Ratio of lithium moles to positive SEI moles": 2.0,
        "Secondary: Positive inner SEI reaction proportion": 0.5,
        "Secondary: Positive inner SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Secondary: Positive outer SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Secondary: Positive SEI reaction exchange current density [A.m-2]": 1.5e-07,
        "Secondary: Positive SEI resistivity [Ohm.m]": 200000.0,
        "Secondary: Positive outer SEI solvent diffusivity [m2.s-1]"
        "": 2.5000000000000002e-22,
        "Secondary: Bulk solvent concentration for positive SEI [mol.m-3]": 2636.0,
        "Secondary: Positive inner SEI open-circuit potential [V]": 0.1,
        "Secondary: Positive outer SEI open-circuit potential [V]": 0.8,
        "Secondary: Positive inner SEI electron conductivity [S.m-1]": 8.95e-14,
        "Secondary: Positive inner SEI lithium interstitial diffusivity [m2.s-1]"
        "": 1e-20,
        "Secondary: Positive lithium interstitial reference concentration [mol.m-3]"
        "": 15.0,
        "Secondary: Initial positive inner SEI thickness [m]": 2.5e-09,
        "Secondary: Initial positive outer SEI thickness [m]": 2.5e-09,
        "Secondary: Positive EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "Secondary: EC diffusivity through positive SEI [m2.s-1]": 2e-18,
        "Secondary: Positive SEI kinetic rate constant [m.s-1]": 1e-12,
        "Secondary: Positive SEI open-circuit potential [V]": 0.4,
        "Secondary: Positive SEI growth activation energy [J.mol-1]": 0.0,
        # cell
        "Negative current collector thickness [m]": 1.2e-05,
        "Negative electrode thickness [m]": 0.0007,
        "Positive current collector thickness [m]": 1.2e-05,
        "Positive electrode thickness [m]": 8.52e-05,
        "Separator thickness [m]": 1.2e-05,
        "Electrode height [m]": 0.065,
        "Electrode width [m]": 1.58,
        "Cell cooling surface area [m2]": 0.00531,
        "Cell volume [m3]": 2.42e-05,
        "Cell thermal expansion coefficient [m.K-1]": 1.1e-06,
        "Positive current collector conductivity [S.m-1]": 58411000.0,
        "Positive current collector density [kg.m-3]": 8960.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Nominal cell capacity [A.h]": 5.0,
        "Current function [A]": 5.0,
        "Contact resistance [Ohm]": 0,
        # negative electrode
        "Negative electrode OCP [V]": 0.0,
        "Negative electrode conductivity [S.m-1]": 10776000.0,
        "Negative electrode OCP entropic change [V.K-1]": 0.0,
        "Exchange-current density for lithium metal electrode [A.m-2]"
        "": li_metal_electrolyte_exchange_current_density_Xu2019,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 215.0,
        "Primary: Maximum concentration in positive electrode [mol.m-3]": 28700.0,
        "Primary: Initial concentration in positive electrode [mol.m-3]": 27700.0,
        "Primary: Positive particle diffusivity [m2.s-1]": 5.5e-14,
        "Primary: Positive electrode OCP [V]": graphite_ocp_Enertech_Ai2020,
        "Positive electrode porosity": 0.25,
        "Primary: Positive electrode active material volume fraction": 0.735,
        "Primary: Positive particle radius [m]": 5.86e-06,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 0,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Primary: Positive electrode exchange-current density [A.m-2]"
        "": graphite_LGM50_electrolyte_exchange_current_density_Chen2020,
        "Primary: Positive electrode density [kg.m-3]": 1657.0,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 1.7,
        "Primary: Positive electrode OCP entropic change [V.K-1]": 0.0,
        "Secondary: Maximum concentration in positive electrode [mol.m-3]": 278000.0,
        "Secondary: Initial concentration in positive electrode [mol.m-3]": 276610.0,
        "Secondary: Positive particle diffusivity [m2.s-1]": 1.67e-14,
        "Secondary: Positive electrode lithiation OCP [V]"
        "": silicon_ocp_lithiation_Mark2016,
        "Secondary: Positive electrode delithiation OCP [V]"
        "": silicon_ocp_delithiation_Mark2016,
        "Secondary: Positive electrode active material volume fraction": 0.015,
        "Secondary: Positive particle radius [m]": 1.52e-06,
        "Secondary: Positive electrode exchange-current density [A.m-2]"
        "": silicon_LGM50_electrolyte_exchange_current_density_Chen2020,
        "Secondary: Positive electrode density [kg.m-3]": 2650.0,
        "Secondary: Positive electrode OCP entropic change [V.K-1]": 0.0,
        # separator
        "Separator porosity": 0.47,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 397.0,
        "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.16,
        # electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number": 0.2594,
        "Thermodynamic factor": 1.0,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Nyman2008,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Nyman2008,
        # experiment
        "Reference temperature [K]": 298.15,
        "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 0.005,
        "Upper voltage cut-off [V]": 1.5,
        "Open-circuit voltage at 0% SOC [V]": 0.005,
        "Open-circuit voltage at 100% SOC [V]": 1.5,
        "Initial concentration in positive electrode [mol.m-3]": 29866.0,
        "Initial temperature [K]": 298.15,
        # citations
        "citations": ["Chen2020", "Ai2022", "Xu2019"],
    }
