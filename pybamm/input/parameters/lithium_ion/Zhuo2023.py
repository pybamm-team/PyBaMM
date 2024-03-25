import pybamm
import os
import numpy as np


# loading data from csv files in the appropriate format
"""
References
----------
    [1] Sturm, J., et al. "Modeling and Simulation of Inhomogeneities in a
18650 Nickel-Rich, Silicon-Graphite Lithium-Ion Cell during Fast Charging".
Journal of Power Sources, vol. 412, Feb. 2019, pp. 204–23.
doi: 10.1016/j.jpowsour.2018.11.043.
"""
path, _ = os.path.split(os.path.abspath(__file__))
sic_18650_ocp_Zhuo2023_data = pybamm.parameters.process_1D_data(
    "sic_18650_ocp_Zhuo2023.csv", path=path
)

sic_18650_dUdT_Zhuo2023_data = pybamm.parameters.process_1D_data(
    "sic_18650_dUdT_Zhuo2023.csv", path=path
)

nmc811_18650_ocp_Zhuo2023_data = pybamm.parameters.process_1D_data(
    "nmc811_18650_ocp_Zhuo2023.csv", path=path
)

nmc811_18650_dUdT_Zhuo2023_data = pybamm.parameters.process_1D_data(
    "nmc811_18650_dUdT_Zhuo2023.csv", path=path
)


def sic_18650_ocp_Zhuo2023(sto):
    name, (x, y) = sic_18650_ocp_Zhuo2023_data
    return pybamm.Interpolant(x, y, sto, name=name)


def sic_18650_dUdT_Zhuo2023(sto, c_max):
    name, (x, y) = sic_18650_dUdT_Zhuo2023_data
    return pybamm.Interpolant(x, y, sto, name=name)


def nmc811_18650_ocp_Zhuo2023(sto):
    name, (x, y) = nmc811_18650_ocp_Zhuo2023_data
    return pybamm.Interpolant(x, y, sto, name=name)


def nmc811_18650_dUdT_Zhuo2023(sto, c_max):
    name, (x, y) = nmc811_18650_dUdT_Zhuo2023_data
    return pybamm.Interpolant(x, y, sto, name=name)


def sic_18650_electrolyte_exchange_current_density_Zhuo2023(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

    References
    ----------
        [1] Sturm, J., et al. "Modeling and Simulation of Inhomogeneities in a
    18650 Nickel-Rich, Silicon-Graphite Lithium-Ion Cell during Fast Charging".
    Journal of Power Sources, vol. 412, Feb. 2019, pp. 204–23.
    doi: 10.1016/j.jpowsour.2018.11.043.

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

    m_ref = 1.0e-11 * pybamm.constants.F  # [m2.5.mol-0.5.s-1]
    E_r = 29931.48
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def nmc811_18650_electrolyte_exchange_current_density_Zhuo2023(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between NMC and LiPF6 in
    EC:DMC.

    References
    ----------
        [1] Sturm, J., et al. "Modeling and Simulation of Inhomogeneities in a
    18650 Nickel-Rich, Silicon-Graphite Lithium-Ion Cell during Fast Charging".
    Journal of Power Sources, vol. 412, Feb. 2019, pp. 204–23.
    doi: 10.1016/j.jpowsour.2018.11.043.

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
    m_ref = 3.2e-11 * pybamm.constants.F  # [m2.5.mol-0.5.s-1]
    E_r = 29931.48
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def sic_18650_diffusivity_Zhuo2023(c_s, T):
    """
    SiC diffusivity as a function of temperature.

    References
    ----------
        [1] Ghosh, Abir, et al. "A Shrinking-Core Model for the Degradation of
    High-Nickel Cathodes (NMC811) in Li-Ion Batteries: Passivation Layer Growth
    and Oxygen Evolution." Journal of The Electrochemical Society,
    168(2) (2021): 020509.

    Parameters
    ----------
    c_s: :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_ref = 1.0e-14

    aEne = 9977.16

    arrhenius = np.exp(aEne / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def nmc811_18650_diffusivity_Zhuo2023(c_s, T):
    """
    NMC811 diffusivity as a function of temperature.

    References
    ----------
        [1] Ghosh, Abir, et al. "A Shrinking-Core Model for the Degradation of
    High-Nickel Cathodes (NMC811) in Li-Ion Batteries: Passivation Layer Growth
    and Oxygen Evolution." Journal of The Electrochemical Society,
    168(2) (2021): 020509.

    Parameters
    ----------
    c_s: :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_ref = 1.0e-14

    aEne = 9977.16

    arrhenius = np.exp(aEne / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


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

    D_c_e = 5.34e-10 * np.exp(-0.65 * c_e / 1000)
    E_D_e = 37040
    arrhenius = np.exp(E_D_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

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
        Solid conductivity
    """

    sigma_e = (
        0.0911
        + 1.9101 * (c_e / 1000)
        - 1.052 * (c_e / 1000) ** 2
        + 0.1554 * (c_e / 1000) ** 3
    )

    E_k_e = 34700
    arrhenius = np.exp(E_k_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return sigma_e * arrhenius


def initial_oxygen_concentration(r_sh_nd, x):
    """
     Initial oxygen concentration profile.

     References
     ----------
     .. [1] Ghosh, Abir, et al. "A Shrinking-Core Model for the Degradation of
     High-Nickel Cathodes (NMC811) in Li-Ion Batteries: Passivation Layer Growth
     and Oxygen Evolution." Journal of The Electrochemical Society,
     168(2) (2021): 020509.
     .. [2] Mingzhao Zhuo, Gregory Offer, Monica Marinescu, "Degradation model of
    high-nickel positive electrodes: Effects of loss of active material and
    cyclable lithium on capacity fade", Journal of Power Sources},
    556 (2023): 232461. doi: 10.1016/j.jpowsour.2022.232461.

     Parameters
     ----------
     r_sh_nd: :class:`pybamm.SpatialVariable`
         r_sh / R_typ, Dimensionless spatial variable (cartesian) in changed 1D domain,
         varying from 0 to 1

     # R_typ: :class:`pybamm.FunctionParameter`
     #     The typical particle radius in the middle of the electrode

     x: :class:`pybamm.SpatialVariable`
         through-cell distance (x) [m]

     Returns
     -------
     :class:`pybamm.Symbol`
         Initial concentration
    """

    c_o_ini_ref = 15219.321

    return c_o_ini_ref * ((1 - r_sh_nd) ** 2)


# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for an INR18650-MJ1 cell, LG Chem, from the paper
    :footcite:t:`Zhuo2023` and references therein.

    SEI parameters are example parameters for SEI growth from the papers
    :footcite:t:`Ramadass2004`, :footcite:t:`ploehn2004solvent`,
    :footcite:t:`single2018identifying`, :footcite:t:`safari2008multimodal`, and
    :footcite:t:`Yang2017`

    .. note::
        This parameter set does not claim to be representative of the true parameter
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
        "SEI resistivity [Ohm.m]": 2.0e5,
        "Outer SEI solvent diffusivity [m2.s-1]": 2.5e-22,
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
        # PE degradation caused by phase transition
        "Positive shell oxygen diffusivity [m2.s-1]": 1e-17,
        "Forward chemical reaction coefficient [m.s-1]": 0.8544e-11,
        "Reverse chemical reaction coefficient [m4.mol-1.s-1]": 1.732e-16,
        "Trapped lithium concentration in the shell [mol.m-3]"
        "": 20000,  # for Fig 7 in Zhuo2023
        # "": 10953.48, # 0.222 * c_max_p for Fig 5 in Zhuo2023
        "Minimum concentration in positive electrode "
        "when fully discharged [mol.m-3]": 10953.48,  # 0.222 * c_max_p
        "Minimum concentration in negative electrode "
        "when fully discharged [mol.m-3]": 68.514,  # 0.002 * c_max_n
        "Threshold lithium concentration for phase transition [mol.m-3]"
        "": 14802,  # 0.3 * c_max_p
        "Positive electrode shell resistivity [Ohm.m]": 1e6,  # Safari2009
        "Constant oxygen concentration in the core [mol.m-3]": 152193.21,
        "Initial lithium concentration in positive core [mol.m-3]"
        "": 46478.28,  # 0.942 * c_max_p
        # same as 'Initial concentration in positive electrode [mol.m-3]' below
        "Initial oxygen concentration in positive shell [mol.m-3]"
        "": initial_oxygen_concentration,
        "Initial core-shell phase boundary location" "": 0.9868421,  # 3.75e-6 / 3.8e-6
        # cell
        # "Negative current collector thickness [m]": 1.2e-05,
        "Negative electrode thickness [m]": 86.7e-6,
        "Separator thickness [m]": 12e-6,
        "Positive electrode thickness [m]": 66.2e-6,
        # "Positive current collector thickness [m]": 1.6e-05,
        "Electrode height [m]": 5.8e-2,
        "Electrode width [m]": 1.23,
        "Cell cooling surface area [m2]": 0.00531,
        "Cell volume [m3]": 2.42e-05,
        # "Cell thermal expansion coefficient [m.K-1]": 1.1e-06,
        "Negative current collector conductivity [S.m-1]": 58411000.0,
        "Positive current collector conductivity [S.m-1]": 36914000.0,
        "Negative current collector density [kg.m-3]": 8960.0,
        "Positive current collector density [kg.m-3]": 2700.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        # # PE 49340*0.72*0.745*4.7227e-6*96487/3600 NE 34257*0.85*0.694*6.1852e-6*96487/3600
        # total cyclable lithium in PE
        # 49340 * (0.942 - 0.222) * 0.745 * 1.23 * 5.8e-2 * 66.2e-6 = 0.1250
        # total cyclable lithium in NE
        # 34257 * (0.852 - 0.002) * 0.694 * 1.23 * 5.8e-2 * 86.7e-6 = 0.1250
        # capacity 0.125 * 96487 / 3600
        "Nominal cell capacity [A.h]": 3.35,
        "Current function [A]": 3.35,
        "Contact resistance [Ohm]": 0,
        # negative electrode
        "Negative electrode conductivity [S.m-1]": 100.0,
        "Maximum concentration in negative electrode [mol.m-3]": 34257.0,
        "Negative electrode diffusivity [m2.s-1]": sic_18650_diffusivity_Zhuo2023,
        "Negative electrode OCP [V]": sic_18650_ocp_Zhuo2023,
        "Negative electrode porosity": 0.216,
        "Negative electrode active material volume fraction": 0.694,
        "Negative particle radius [m]": 6.1e-06,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 1.5,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Negative electrode exchange-current density [A.m-2]"
        "": sic_18650_electrolyte_exchange_current_density_Zhuo2023,
        "Negative electrode density [kg.m-3]": 1657.0,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 918.8,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        "Negative electrode OCP entropic change [V.K-1]": sic_18650_dUdT_Zhuo2023,
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 0.17,
        "Maximum concentration in positive electrode [mol.m-3]": 49340.0,
        "Positive electrode diffusivity [m2.s-1]": nmc811_18650_diffusivity_Zhuo2023,
        "Positive electrode OCP [V]": nmc811_18650_ocp_Zhuo2023,
        "Positive electrode porosity": 0.171,
        "Positive electrode active material volume fraction": 0.745,
        "Positive particle radius [m]": 3.8e-6,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.85,
        "Positive electrode Bruggeman coefficient (electrode)": 1.5,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode exchange-current density [A.m-2]"
        "": nmc811_18650_electrolyte_exchange_current_density_Zhuo2023,
        "Positive electrode density [kg.m-3]": 3262.0,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 918.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
        "Positive electrode OCP entropic change [V.K-1]": nmc811_18650_dUdT_Zhuo2023,
        # separator
        "Separator porosity": 0.45,
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
        # "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 2.8,
        "Upper voltage cut-off [V]": 4.2,
        "Open-circuit voltage at 0% SOC [V]": 2.8,
        "Open-circuit voltage at 100% SOC [V]": 4.2,
        "Initial concentration in negative electrode [mol.m-3]": 68.514,
        # same as in positive core, both defined for different submodel
        "Initial concentration in positive electrode [mol.m-3]"
        "": 46478.28,  # 0.942 * c_max_p
        "Initial temperature [K]": 298.15,
        # citations
        "citations": ["Zhuo2023"],
    }
