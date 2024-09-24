import pybamm
import os

path, _ = os.path.split(os.path.abspath(__file__))

U_n_data = pybamm.parameters.process_1D_data("U_n.csv", path=path)
U_p_data = pybamm.parameters.process_1D_data("U_p.csv", path=path)
D_n_data = pybamm.parameters.process_1D_data("D_n.csv", path=path)
D_p_data = pybamm.parameters.process_1D_data("D_p.csv", path=path)
k_n_data = pybamm.parameters.process_1D_data("k_n.csv", path=path)
k_p_data = pybamm.parameters.process_1D_data("k_p.csv", path=path)
D_e_data = pybamm.parameters.process_1D_data("D_e.csv", path=path)
sigma_e_data = pybamm.parameters.process_1D_data("sigma_e.csv", path=path)


def HC_ocp_Chayambuka2022(sto):
    """
    HC open-circuit potential as a function of stochiometry, data taken
    from [1].

    References
    ----------
    .. [1] K. Chayambuka, G. Mulder, D.L. Danilov, P.H.L. Notten, Physics-based
    modeling of sodium-ion batteries part II. Model and validation, Electrochimica
    Acta 404 (2022) 139764. https://doi.org/10.1016/j.electacta.2021.139764.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open-circuit potential
    """

    name, (x, y) = U_n_data
    return pybamm.Interpolant(x, y, sto, name)


def HC_diffusivity_Chayambuka2022(sto, T):
    """
    HC diffusivity as a function of stochiometry, the data is taken from [1].

    References
    ----------
    .. [1] K. Chayambuka, G. Mulder, D.L. Danilov, P.H.L. Notten, Physics-based
    modeling of sodium-ion batteries part II. Model and validation, Electrochimica
    Acta 404 (2022) 139764. https://doi.org/10.1016/j.electacta.2021.139764.

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

    name, (x, y) = D_n_data
    c_max = pybamm.Parameter("Maximum concentration in negative electrode [mol.m-3]")
    return pybamm.Interpolant(x, y, sto * c_max, name)


def HC_electrolyte_exchange_current_density_Chayambuka2022(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between HC and NaPF6 in
    EC:PC.

    References
    ----------
    .. [1] K. Chayambuka, G. Mulder, D.L. Danilov, P.H.L. Notten, Physics-based
    modeling of sodium-ion batteries part II. Model and validation, Electrochimica
    Acta 404 (2022) 139764. https://doi.org/10.1016/j.electacta.2021.139764.

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
    name, (x, y) = k_n_data
    k_n = pybamm.Interpolant(x, y, c_s_surf, name)
    c_e0 = pybamm.Parameter("Initial concentration in electrolyte [mol.m-3]")

    return (
        pybamm.constants.F
        * k_n
        * (c_e / c_e0) ** 0.5
        * c_s_surf**0.5
        * (c_s_max - c_s_surf) ** 0.5
        / 2
    )


def NVPF_ocp_Chayambuka2022(sto):
    """
    NVPF open-circuit potential as a function of stochiometry, data taken
    from [1].

    References
    ----------
    .. [1] K. Chayambuka, G. Mulder, D.L. Danilov, P.H.L. Notten, Physics-based
    modeling of sodium-ion batteries part II. Model and validation, Electrochimica
    Acta 404 (2022) 139764. https://doi.org/10.1016/j.electacta.2021.139764.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open-circuit potential
    """

    name, (x, y) = U_p_data
    return pybamm.Interpolant(x, y, sto, name)


def NVPF_diffusivity_Chayambuka2022(sto, T):
    """
    NVPF diffusivity as a function of stochiometry, the data is taken from [1].

    References
    ----------
    .. [1] K. Chayambuka, G. Mulder, D.L. Danilov, P.H.L. Notten, Physics-based
    modeling of sodium-ion batteries part II. Model and validation, Electrochimica
    Acta 404 (2022) 139764. https://doi.org/10.1016/j.electacta.2021.139764.

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

    name, (x, y) = D_p_data
    c_max = pybamm.Parameter("Initial concentration in electrolyte [mol.m-3]")
    return pybamm.Interpolant(x, y, sto * c_max, name)


def NVPF_electrolyte_exchange_current_density_Chayambuka2022(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between NVPF and NaPF6 in
    EC:PC.

    References
    ----------
    .. [1] K. Chayambuka, G. Mulder, D.L. Danilov, P.H.L. Notten, Physics-based
    modeling of sodium-ion batteries part II. Model and validation, Electrochimica
    Acta 404 (2022) 139764. https://doi.org/10.1016/j.electacta.2021.139764.

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
    name, (x, y) = k_p_data
    k_p = pybamm.Interpolant(x, y, c_s_surf, name)
    c_e0 = pybamm.Parameter("Initial concentration in electrolyte [mol.m-3]")

    return (
        pybamm.constants.F
        * k_p
        * (c_e / c_e0) ** 0.5
        * c_s_surf**0.5
        * (c_s_max - c_s_surf) ** 0.5
        / 2
    )


def electrolyte_diffusivity_Chayambuka2022(c_e, T):
    """
    Diffusivity of NaPF6 in EC:PC (1:1) as a function of ion concentration. The data
    comes from [1]

    References
    ----------
    .. [1] K. Chayambuka, G. Mulder, D.L. Danilov, P.H.L. Notten, Physics-based
    modeling of sodium-ion batteries part II. Model and validation, Electrochimica
    Acta 404 (2022) 139764. https://doi.org/10.1016/j.electacta.2021.139764.

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

    name, (x, y) = D_e_data
    D_e = pybamm.Interpolant(x, y, c_e, name)

    # Chayambuka et al. (2022) does not provide temperature dependence

    return D_e


def electrolyte_conductivity_Chayambuka2022(c_e, T):
    """
    Conductivity of NaPF6 in EC:PC (1:1) as a function of ion concentration. The data
    comes from [1].

    References
    ----------
    .. [1] K. Chayambuka, G. Mulder, D.L. Danilov, P.H.L. Notten, Physics-based
    modeling of sodium-ion batteries part II. Model and validation, Electrochimica
    Acta 404 (2022) 139764. https://doi.org/10.1016/j.electacta.2021.139764.

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

    name, (x, y) = sigma_e_data
    sigma_e = pybamm.Interpolant(x, y, c_e, name)

    # Chayambuka et al. (2022) does not provide temperature dependence

    return sigma_e


# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for a sodium-ion cell, from the paper :footcite:t:`Chayambuka2022` and references
    therein. The specific parameter values are taken from the COMSOL implementation presented in
    [this example](https://www.comsol.com/model/1d-isothermal-sodium-ion-battery-117341).

    """

    return {
        "chemistry": "sodium_ion",
        # cell
        "Negative electrode thickness [m]": 64e-6,
        "Separator thickness [m]": 25e-6,
        "Positive electrode thickness [m]": 68e-6,
        "Electrode height [m]": 2.54e-4,
        "Electrode width [m]": 1,
        "Nominal cell capacity [A.h]": 3e-3,
        "Current function [A]": 3e-3,
        "Contact resistance [Ohm]": 0,
        # negative electrode
        "Negative electrode conductivity [S.m-1]": 256,
        "Maximum concentration in negative electrode [mol.m-3]": 14540,
        "Negative particle diffusivity [m2.s-1]": HC_diffusivity_Chayambuka2022,
        "Negative electrode OCP [V]": HC_ocp_Chayambuka2022,
        "Negative electrode porosity": 0.51,
        "Negative electrode active material volume fraction": 0.489,  # 1 - 0.51 - 0.001
        "Negative particle radius [m]": 3.48e-6,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 0,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode exchange-current density [A.m-2]"
        "": HC_electrolyte_exchange_current_density_Chayambuka2022,
        "Negative electrode OCP entropic change [V.K-1]": 0,
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 50,
        "Maximum concentration in positive electrode [mol.m-3]": 15320,
        "Positive particle diffusivity [m2.s-1]": NVPF_diffusivity_Chayambuka2022,
        "Positive electrode OCP [V]": NVPF_ocp_Chayambuka2022,
        "Positive electrode porosity": 0.23,
        "Positive electrode active material volume fraction": 0.55,  # 1 - 0.23 - 0.22
        "Positive particle radius [m]": 0.59e-6,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 0,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode exchange-current density [A.m-2]"
        "": NVPF_electrolyte_exchange_current_density_Chayambuka2022,
        "Positive electrode OCP entropic change [V.K-1]": 0,
        # separator
        "Separator porosity": 0.55,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        # electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1000,
        "Cation transference number": 0.45,
        "Thermodynamic factor": 1,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Chayambuka2022,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Chayambuka2022,
        # experiment
        "Reference temperature [K]": 298.15,
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 2.0,
        "Upper voltage cut-off [V]": 4.2,
        "Open-circuit voltage at 0% SOC [V]": 2.0,
        "Open-circuit voltage at 100% SOC [V]": 4.2,
        "Initial concentration in negative electrode [mol.m-3]": 13520,
        "Initial concentration in positive electrode [mol.m-3]": 3320,
        "Initial temperature [K]": 298.15,
        # citations
        "citations": ["Chayambuka2022"],
    }
