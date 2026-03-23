import numpy as np

import pybamm


def graphite_CircularLIB_ocp_Hebenbrock2025(sto):
    """
    CircularLIB pouchcell open circuit potential of NMC811 vs. Li/Li+ as a function of stochiometry [1].

    References
    ----------
    .. [1] Hebenbrock, A., Mohni, V. N., Blumberg, A., Schade, W., Schröder, D., & Turek, T., 2026.
        Operando strain analysis of electrodes using internal sensors and holistic electrochemical-microstructural modeling.
        J. Energy Storage, 154, 121255. https://doi.org/10.1016/j.est.2026.121255

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open circuit potential
    """
    u_eq = (
        -1.230130877183463e03 * sto**9
        + 5.269131139767095e03 * sto**8
        - 9.667028448586449e03 * sto**7
        + 9.920895571681862e03 * sto**6
        - 6.239925663346142e03 * sto**5
        + 2.479205053606367e03 * sto**4
        - 6.199144563918534e02 * sto**3
        + 94.840048789313710 * sto**2
        - 8.496726235715633 * sto
        + 0.200992370674797
        + 0.3545045  # last addition is potential shift from lithatited gold refence electrode to Li+/Li
    )

    return u_eq


def graphite_exchange_current_density_CircularLIB(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC. Parameters taken from Ecker.2015.

    References
    ----------
    .. [1] Hebenbrock, A., Mohni, V. N., Blumberg, A., Schade, W., Schröder, D., & Turek, T., 2026.
        Operando strain analysis of electrodes using internal sensors and holistic electrochemical-microstructural modeling.
        J. Energy Storage, 154, 121255. https://doi.org/10.1016/j.est.2026.121255

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
    k = 1.44832e-8  # m/s
    m_ref = (
        k * pybamm.constants.F / 1000**0.5
    )  # Chen.2020 Eq. (22) solved for k and then in (19)
    E_r = 53400
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def nmc_CircularLIB_ocp_Hebenbrock2025(sto):
    """
    CircularLIB pouchcell open circuit potential of NMC811 vs. Li/Li+ as a function of stochiometry [1].

    References
    ----------
    .. [1] Hebenbrock, A., Mohni, V. N., Blumberg, A., Schade, W., Schröder, D., & Turek, T., 2026.
        Operando strain analysis of electrodes using internal sensors and holistic electrochemical-microstructural modeling.
        J. Energy Storage, 154, 121255. https://doi.org/10.1016/j.est.2026.121255

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open circuit potential
    """

    u_eq = (
        -8.508257873565149e04 * sto**9
        + 4.010622215898777e05 * sto**8
        - 8.213096574015446e05 * sto**7
        + 9.576574867257422e05 * sto**6
        - 6.997532677585998e05 * sto**5
        + 3.318855422125536e05 * sto**4
        - 1.020654738082277e05 * sto**3
        + 1.960625290978436e04 * sto**2
        - 2.133874350173679e03 * sto
        + 104.1824
        + 0.3545045  # last addition is potential shift from lithatited gold refence electrode to Li+/Li
    )

    return u_eq


def nmc811_exchange_current_density_CircularLIB(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between NMC and LiPF6 in
    EC:DMC. Parameters taken from [1].

    References
    ----------
    .. [1] Chen, C.-H., Brosa Planella, F., O’Regan, K., Gastol, D., Widanage, W. D., & Kendrick, E., 2020.
    Development of Experimental Techniques for Parameterization of Multi-scale Lithium-ion Battery Models.
    J. Electrochem. Soc., 167(8), 080534. https://doi.org/10.1149/1945-7111/ab9050


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
    k = 1.12e-9
    m_ref = k * pybamm.constants.F / 1000**0.5  # same value as Chen.2020
    # m_ref = 3.42e-6  # (A/m2)(m3/mol)**1.5 - includes ref concentrations #Chen2020
    E_r = 17800  # J/mol
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def electrolyte_diffusivity_Landesfeind2019(c_e, T):
    """
    Diffusivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1]

    References
    ----------
    .. [1] Landesfeind, J., Gasteiger, H.A., 2019.
    Temperature and Concentration Dependence of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes.
    J. Electrochem. Soc. 166, A3079. https://doi.org/10.1149/2.0571912jes


    Parameters
    ----------
    c_e: :class:`pybamm.Symbol` # input: mol/m**3 - need: mol/L
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`   # input: K - need: K
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    p1 = 1.01e3
    p2 = 1.01e0
    p3 = -1.56e3
    p4 = -4.87e2

    c_e_inp = c_e / 1000
    D_c_e = (
        p1 * np.exp(p2 * c_e_inp) * np.exp(p3 / T) * np.exp(p4 / T * c_e_inp) * 10e-11
    )  # m**2/s # corrected from 10e-10 because result was wrong by one order

    return D_c_e


def electrolyte_conductivity_Landesfeind2019(c_e, T):
    """
    Conductivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1].

    References
    ----------
    .. [1] Landesfeind, J., Gasteiger, H.A., 2019.
    Temperature and Concentration Dependence of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes.
    J. Electrochem. Soc. 166, A3079. https://doi.org/10.1149/2.0571912jes


    Parameters
    ----------
    c_e: :class:`pybamm.Symbol` # input: mol/m**3 - need: mol/L
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`   # input: K - need: K
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """
    p1 = 5.21e-1
    p2 = 2.28e2
    p3 = -1.06e0
    p4 = 3.53e-1
    p5 = -3.59e-3
    p6 = 1.48e-3

    c_e_inp = c_e / 1000  # conversion from "mol/m**3" in "mol/L"
    sigma_e = (
        p1
        * (1 + (T - p2))
        * c_e_inp
        * (1 + p3 * c_e_inp**0.5 + p4 * (1 + p5 * np.exp(1000 / T)) * c_e_inp)
        / (1 + c_e_inp**4 * (p6 * np.exp(1000 / T)))
    ) / 10  # S/m

    return sigma_e


def electrolyte_thermodynamic_factor_Landesfeind2019(c_e, T):
    """
    Thermodynamic factor of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1].

    References
    ----------
    .. [1] Landesfeind, J., Gasteiger, H.A., 2019.
    Temperature and Concentration Dependence of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes.
    J. Electrochem. Soc. 166, A3079. https://doi.org/10.1149/2.0571912jes


    Parameters
    ----------
    c_e: :class:`pybamm.Symbol` # input: mol/m**3 - need: mol/L
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`   # input: K - need: K
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """
    p1 = 2.57e1
    p2 = -4.51e1
    p3 = -1.77e-1
    p4 = 1.94e0
    p5 = 2.95e-1
    p6 = 3.08e-4
    p7 = 2.59e-1
    p8 = -9.46e-3
    p9 = -4.54e-4

    c_e_inp = c_e / 1000
    TDF = (
        p1
        + p2 * c_e_inp
        + p3 * T
        + p4 * c_e_inp**2
        + p5 * c_e_inp * T
        + p6 * T**2
        + p7 * c_e_inp**3
        + p8 * c_e_inp**2 * T
        + p9 * c_e_inp * T**2
    )

    return TDF


def electrolyte_tansference_number_Landesfeind2019(c_e, T):
    """
    Transference number of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1].

    References
    ----------
    .. [1] Landesfeind, J., Gasteiger, H.A., 2019.
    Temperature and Concentration Dependence of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes.
    J. Electrochem. Soc. 166, A3079. https://doi.org/10.1149/2.0571912jes


    Parameters
    ----------
    c_e: :class:`pybamm.Symbol` # input: mol/m**3 - need: mol/L
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`   # input: K - need: K
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """
    p1 = -1.28e1
    p2 = -6.12e0
    p3 = 8.21e-2
    p4 = 9.04e-1
    p5 = 3.18e-2
    p6 = -1.27e-4
    p7 = 1.75e-2
    p8 = -3.12e-3
    p9 = -3.96e-5

    c_e_inp = c_e / 1000
    tranfer_number = (
        p1
        + p2 * c_e_inp
        + p3 * T
        + p4 * c_e_inp**2
        + p5 * c_e_inp * T
        + p6 * T**2
        + p7 * c_e_inp**3
        + p8 * c_e_inp**2 * T
        + p9 * c_e_inp * T**2
    )

    return tranfer_number


def get_parameter_values():
    """
    Parameters for an multilayer cell, from the paper

        Hebenbrock, A., Mohni, V. N., Blumberg, A., Schade, W., Schröder, D., & Turek, T., 2026.
        Operando strain analysis of electrodes using internal sensors and holistic electrochemical-microstructural modeling.
        J. Energy Storage, 154, 121255. https://doi.org/10.1016/j.est.2026.121255

    and references therein.
    """

    return {
        "chemistry": "lithium_ion",
        # cell
        "Negative current collector thickness [m]": 1.0e-05,
        "Negative electrode thickness [m]": 8.93e-05,
        "Separator thickness [m]": 2.5e-05,
        "Positive electrode thickness [m]": 6.56e-05,
        "Positive current collector thickness [m]": 1.5e-05,
        "Electrode height [m]": 0.065,
        "Electrode width [m]": 0.045,
        "Nominal cell capacity [A.h]": 1.1,
        "Current function [A]": 1.1,
        "Contact resistance [Ohm]": 0.01,
        # negative electrode
        "Negative electrode conductivity [S.m-1]": 1.766e3,
        "Maximum concentration in negative electrode [mol.m-3]": 30713.24709,
        "Negative electrode diffusivity [m2.s-1]": 2.35e-15,
        "Negative electrode OCP [V]": graphite_CircularLIB_ocp_Hebenbrock2025,
        "Negative electrode porosity": 0.260,
        "Negative electrode active material volume fraction": 0.74 * 0.93,
        "Negative particle radius [m]": 7.85e-06,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 0,
        "Negative electrode exchange-current density [A.m-2]"
        "": graphite_exchange_current_density_CircularLIB,
        "Negative electrode OCP entropic change [V.K-1]": 0.0,
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 14.53,
        "Maximum concentration in positive electrode [mol.m-3]": 45020.0,
        "Positive electrode diffusivity [m2.s-1]": 5.8e-15,
        "Positive electrode OCP [V]": nmc_CircularLIB_ocp_Hebenbrock2025,
        "Positive electrode porosity": 0.222,
        "Positive electrode active material volume fraction": 0.788 * 0.94,
        "Positive particle radius [m]": 5.5e-06,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 0,
        "Positive electrode exchange-current density [A.m-2]"
        "": nmc811_exchange_current_density_CircularLIB,
        "Positive electrode OCP entropic change [V.K-1]": 0.0,
        # separator
        "Separator porosity": 0.405,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        # electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number": electrolyte_tansference_number_Landesfeind2019,
        "Thermodynamic factor": electrolyte_thermodynamic_factor_Landesfeind2019,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Landesfeind2019,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Landesfeind2019,
        # experiment
        "Reference temperature [K]": 298.15,
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 11,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 3.0,
        "Upper voltage cut-off [V]": 4.2,
        "Open-circuit voltage at 0% SOC [V]": 3.0,
        "Open-circuit voltage at 100% SOC [V]": 4.2,
        "Initial concentration in negative electrode [mol.m-3]": 0.4620 * 30713.24709,
        "Initial concentration in positive electrode [mol.m-3]": 0.5434 * 45020,
        "Initial temperature [K]": 298.15,
        # citations
        "citations": ["Hebenbrock2026"],
    }
