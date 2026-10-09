from pathlib import Path

import numpy as np

import pybamm
from pybamm import constants, exp

BASE_DIR = Path(__file__).resolve().parent
NMC_OCP = np.loadtxt(BASE_DIR / "data/NMC_OCP.csv", delimiter=",")
NMC_dUdT1 = np.loadtxt(BASE_DIR / "data/NMC_dUdT1.csv", delimiter=",")
SiC_OCP = np.loadtxt(BASE_DIR / "data/SIC_OCP.csv", delimiter=",")
SiC_dUdT1 = np.loadtxt(BASE_DIR / "data/SIC_dUdT1.csv", delimiter=",")


def initial_oxygen_concentration(x):
    """
    Initial oxygen concentration in the positive electrode shell
    :footcite:t:`Zhuo2023`, which is zero for a pristine particle.

    Parameters
    ----------
    x : :class:`pybamm.Symbol`
        Dimensionless radial position in the shell

    Returns
    -------
    :class:`pybamm.Symbol`
        Oxygen concentration [mol.m-3]
    """

    c_o_ini_ref = 0

    return c_o_ini_ref * (1 - x**2)


def nmc811_diffusivity_degraded(sto, T):
    """
    Lithium diffusivity in the degraded shell of an NMC811 particle, with
    Arrhenius temperature dependence from :footcite:t:`Ghosh2021`.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Electrode stoichiometry
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity [m2.s-1]
    """

    D_ref = 1.0e-15
    aEne = 9977.16

    arrhenius = exp(aEne / constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def nmc811_diffusivity(sto, T):
    """
    Lithium diffusivity in NMC811, with Arrhenius temperature dependence from :footcite:t:`Ghosh2021`.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Electrode stoichiometry
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity [m2.s-1]
    """

    D_ref = 1.0e-14
    aEne = 9977.16

    arrhenius = exp(aEne / constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def sic_diffusivity(sto, T):
    """
    Lithium diffusivity in SiC, with Arrhenius temperature dependence from :footcite:t:`Ghosh2021`.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Electrode stoichiometry
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity [m2.s-1]
    """

    D_ref = 1.0e-14
    aEne = 9977.16

    arrhenius = exp(aEne / constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def sic_electrolyte_exchange_current_density(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between SiC and LiPF6 in
    EC:DMC :footcite:t:`Ghosh2021`.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle surface concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    m_ref = 1.0e-11 * constants.F  # [m2.5.mol-0.5.s-1]
    E_r = 29931.48
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    c_n_max = c_s_max

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_n_max - c_s_surf) ** 0.5


def nmc811_electrolyte_exchange_current_density(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between NMC811 and LiPF6
    in EC:DMC :footcite:t:`Ghosh2021`.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle surface concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    m_ref = 3.2e-11 * constants.F  # [m2.5.mol-0.5.s-1]
    E_r = 29931.48
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    c_p_max = c_s_max

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_p_max - c_s_surf) ** 0.5


def electrolyte_conductivity_Capiglia1999(c_e, T):
    """
    Conductivity of LiPF6 in EC:DMC as a function of ion concentration. The original
    data is from [1]. The fit is from Dualfoil [2].

    References
    ----------
    .. [1] C Capiglia et al. 7Li and 19F diffusion coefficients and thermal
    properties of non-aqueous electrolyte solutions for rechargeable lithium batteries.
    Journal of power sources 81 (1999): 859-862.
    .. [2] John Newman, Dualfoil

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte conductivity [S.m-1]
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


def electrolyte_diffusivity_Capiglia1999(c_e, T):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration. The original
    data is from [1]. The fit is from Dualfoil [2].

    References
    ----------
    .. [1] C Capiglia et al. 7Li and 19F diffusion coefficients and thermal
    properties of non-aqueous electrolyte solutions for rechargeable lithium batteries.
    Journal of power sources 81 (1999): 859-862.
    .. [2] John Newman, Dualfoil

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte diffusivity [m2.s-1]
    """

    D_c_e = 5.34e-10 * np.exp(-0.65 * c_e / 1000)
    E_D_e = 37040
    arrhenius = np.exp(E_D_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_c_e * arrhenius


def _zhuo_ocp_asymptote(sto):
    """
    OCP asymptote used by :footcite:t:`Zhuo2023`, less the asymptote that
    :meth:`pybamm.LithiumIonParameters` adds to every open-circuit potential.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Electrode stoichiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Correction to the open-circuit potential [V]
    """
    from pybamm.parameters.lithium_ion_parameters import U_asymptotes

    return 1e-6 * (1 / sto + 1 / (sto - 1)) - U_asymptotes(sto)


def nmc811_ocp_zhuo(sto):
    """
    NMC811 open-circuit potential from the data of :footcite:t:`Ghosh2021`.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Electrode stoichiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open-circuit potential [V]
    """
    return pybamm.Interpolant(
        NMC_OCP[:, 0],
        NMC_OCP[:, 1],
        sto,
        name="NMC_OCP",
        interpolator="cubic",
        extrapolate=True,
    ) + _zhuo_ocp_asymptote(sto)


def sic_dudt_zhuo(sto):
    """
    SiC entropic change in open-circuit potential, measured by
    :footcite:t:`Sturm2019` and tabulated in mV.K-1.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Electrode stoichiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Entropic change [V.K-1]
    """
    return 1e-3 * pybamm.Interpolant(
        SiC_dUdT1[:, 0],
        SiC_dUdT1[:, 1],
        sto,
        name="SiC_dUdT1",
        interpolator="cubic",
        extrapolate=True,
    )


def nmc811_dudt_zhuo(sto):
    """
    NMC811 entropic change in open-circuit potential, measured by
    :footcite:t:`Sturm2019` and tabulated in mV.K-1.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Electrode stoichiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Entropic change [V.K-1]
    """
    return 1e-3 * pybamm.Interpolant(
        NMC_dUdT1[:, 0],
        NMC_dUdT1[:, 1],
        sto,
        name="NMC_dUdT1",
        interpolator="cubic",
        extrapolate=True,
    )


def sic_ocp_zhuo(sto):
    """
    SiC open-circuit potential from the data of :footcite:t:`Ghosh2021`.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Electrode stoichiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open-circuit potential [V]
    """
    return pybamm.Interpolant(
        SiC_OCP[:, 0],
        SiC_OCP[:, 1],
        sto,
        name="SiC_OCP",
        interpolator="cubic",
        extrapolate=True,
    ) + _zhuo_ocp_asymptote(sto)


def get_parameter_values():
    """
    Parameters for an NMC811/SiC 18650 cell with positive electrode degradation,
    from the paper :footcite:t:`Zhuo2023`. The cell geometry and separator are from
    :footcite:t:`Sturm2019`, the electrodes and shrinking-core degradation
    parameters from :footcite:t:`Ghosh2021`, the electrolyte from
    :footcite:t:`Marquis2019`, and the shell resistivity from :footcite:t:`Safari2008`.
    """

    return {
        "chemistry": "lithium_ion",
        "Negative electrode thickness [m]": 8.67e-05,
        "Separator thickness [m]": 1.2e-05,
        "Positive electrode thickness [m]": 6.62e-05,
        "Electrode height [m]": 0.058,
        "Electrode width [m]": 1.23,
        "Nominal cell capacity [A.h]": 3.35,
        "Typical current [A]": 3.35,
        "Current function [A]": 3.35,
        "Negative electrode conductivity [S.m-1]": 100.0,
        "Maximum concentration in negative electrode [mol.m-3]": 34257.0,
        "Negative particle diffusivity [m2.s-1]": sic_diffusivity,
        "Negative electrode OCP [V]": sic_ocp_zhuo,
        "Negative electrode porosity": 0.216,
        "Negative electrode active material volume fraction": 0.694,
        "Negative particle radius [m]": 6.1e-06,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 1.5,
        "Negative electrode electrons in reaction": 1.0,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode exchange-current density [A.m-2]": sic_electrolyte_exchange_current_density,
        "Negative electrode OCP entropic change [V.K-1]": sic_dudt_zhuo,
        "Positive electrode conductivity [S.m-1]": 0.17,
        "Maximum concentration in positive electrode [mol.m-3]": 49340.0,
        "Positive particle diffusivity [m2.s-1]": nmc811_diffusivity,
        "Positive electrode OCP [V]": nmc811_ocp_zhuo,
        "Positive electrode porosity": 0.171,
        "Positive electrode active material volume fraction": 0.745,
        "Positive particle radius [m]": 3.8e-06,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.85,
        "Positive electrode Bruggeman coefficient (electrode)": 1.5,
        "Positive electrode electrons in reaction": 1.0,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode exchange-current density [A.m-2]": nmc811_electrolyte_exchange_current_density,
        "Positive electrode OCP entropic change [V.K-1]": nmc811_dudt_zhuo,
        "Separator porosity": 0.45,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Typical electrolyte concentration [mol.m-3]": 1000.0,
        "Cation transference number": 0.4,
        "Thermodynamic factor": 1.0,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Capiglia1999,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Capiglia1999,
        "Reference temperature [K]": 298.15,
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 2.5,
        "Upper voltage cut-off [V]": 4.4,
        "Initial concentration in negative electrode [mol.m-3]": 68.514,
        "Initial concentration in positive electrode [mol.m-3]": 46478.28,
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Initial temperature [K]": 298.15,
        "Positive core diffusivity [m2.s-1]": nmc811_diffusivity,
        "Positive shell diffusivity [m2.s-1]": nmc811_diffusivity_degraded,
        "Positive shell oxygen diffusivity [m2.s-1]": 1e-17,
        "Forward chemical reaction coefficient [m.s-1]": 8.544e-12,
        "Reverse chemical reaction coefficient [m4.mol-1.s-1]": 1.732e-16,
        "Initial concentration in positive core [mol.m-3]": 46478.28,
        "Trapped lithium concentration in shell [mol.m-3]": 10953.48,
        "Minimum concentration in positive core when fully charged [mol.m-3]": 10953.48,
        "Minimum concentration in negative particle when fully discharged [mol.m-3]": 68.514,
        "Initial oxygen concentration in positive shell [mol.m-3]": initial_oxygen_concentration,
        "Constant oxygen concentration in particle core [mol.m-3]": 152193.21,
        "Initial phase boundary location [m]": 3.75e-06,
        "Threshold concentration for phase transition [mol.m-3]": 14802.0,
        "Positive electrode shell resistivity [Ohm.m]": 0.0,
        # citations
        "citations": [
            "Safari2008",
            "Marquis2019",
            "Sturm2019",
            "Ghosh2021",
            "Zhuo2023",
        ],
    }
