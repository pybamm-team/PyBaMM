import pybamm


def electrolyte_diffusivity_Nyman2008(c_e, T):
    D_c_e = 8.794e-11 * (c_e / 1000) ** 2 - 3.972e-10 * (c_e / 1000) + 4.862e-10
    return D_c_e


def electrolyte_conductivity_Nyman2008(c_e, T):
    sigma_e = (
        0.1297 * (c_e / 1000) ** 3 - 2.51 * (c_e / 1000) ** 1.5 + 3.329 * (c_e / 1000)
    )
    return sigma_e


def x_n(U):
    T = 298.15
    f = pybamm.constants.F / (pybamm.constants.R * T)
    xj = 0
    for i in range(6):
        U0 = pybamm.Parameter(f"U0_n_{i}")
        w = pybamm.Parameter(f"w_n_{i}")
        Xj = pybamm.Parameter(f"Xj_n_{i}")

        xj += Xj / (1 + pybamm.exp(f * (U - U0) / w))

    return xj


def dxdU_n(U):
    T = 298.15
    f = pybamm.constants.F / (pybamm.constants.R * T)
    dxj = 0
    for i in range(6):
        U0 = pybamm.Parameter(f"U0_n_{i}")
        w = pybamm.Parameter(f"w_n_{i}")
        Xj = pybamm.Parameter(f"Xj_n_{i}")

        e = pybamm.exp(f * (U - U0) / w)
        dxj += -(f / w) * (Xj * e) / (1 + e) ** 2

    return dxj


def x_p(U):
    T = 298.15
    f = pybamm.constants.F / (pybamm.constants.R * T)
    xj = 0
    for i in range(4):
        U0 = pybamm.Parameter(f"U0_p_{i}")
        w = pybamm.Parameter(f"w_p_{i}")
        Xj = pybamm.Parameter(f"Xj_p_{i}")

        xj += Xj / (1 + pybamm.exp(f * (U - U0) / w))

    return xj


def dxdU_p(U):
    T = 298.15
    f = pybamm.constants.F / (pybamm.constants.R * T)
    dxj = 0
    for i in range(4):
        U0 = pybamm.Parameter(f"U0_p_{i}")
        w = pybamm.Parameter(f"w_p_{i}")
        Xj = pybamm.Parameter(f"Xj_p_{i}")

        e = pybamm.exp(f * (U - U0) / w)
        dxj += -(f / w) * (Xj * e) / (1 + e) ** 2

    return dxj


def get_parameter_values():
    return {
        # cell
        "Negative electrode thickness [m]": 7.56e-05,
        "Separator thickness [m]": 1.2e-05,
        "Positive electrode thickness [m]": 7.56e-05,
        "Electrode height [m]": 0.065,
        "Electrode width [m]": 1.58,
        "Nominal cell capacity [A.h]": 5.0,
        "Current function [A]": 5.0,
        "Contact resistance [Ohm]": 0,
        # negative electrode
        "Negative electrode stoichiometry": x_n,
        "Negative electrode differential stoichiometry [V-1]": dxdU_n,
        "U0_n_0": 0.08843,
        "Xj_n_0": 0.43336,
        "w_n_0": 0.08611,
        "U0_n_1": 0.12799,
        "Xj_n_1": 0.23963,
        "w_n_1": 0.08009,
        "U0_n_2": 0.14331,
        "Xj_n_2": 0.15018,
        "w_n_2": 0.72469,
        "U0_n_3": 0.16984,
        "Xj_n_3": 0.05462,
        "w_n_3": 2.53277,
        "U0_n_4": 0.21446,
        "Xj_n_4": 0.06744,
        "w_n_4": 0.09470,
        "U0_n_5": 0.36325,
        "Xj_n_5": 0.05476,
        "w_n_5": 5.97354,
        "Negative electrode stoichiometry at 0% SOC": 0.03,
        "Negative electrode stoichiometry at 100% SOC": 0.9,
        "Negative electrode conductivity [S.m-1]": 215.0,
        "Maximum concentration in negative electrode [mol.m-3]": 33133.0,
        "Negative electrode diffusivity [m2.s-1]": 3.3e-14,
        "Negative electrode porosity": 0.25,
        "Negative electrode active material volume fraction": 0.75,
        "Negative particle radius [m]": 5.86e-06,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 0,
        "Negative electrode exchange-current density [A.m-2]" "": 2.7,
        "Negative electrode OCP entropic change [V.K-1]": 0.0,
        # positive electrode
        "Positive electrode stoichiometry": x_p,
        "Positive electrode differential stoichiometry [V-1]": dxdU_p,
        "U0_p_0": 3.62274,
        "Xj_p_0": 0.13442,
        "w_p_0": 0.96710,
        "U0_p_1": 3.72645,
        "Xj_p_1": 0.32460,
        "w_p_1": 1.39712,
        "U0_p_2": 3.90575,
        "Xj_p_2": 0.21118,
        "w_p_2": 3.50500,
        "U0_p_3": 4.22955,
        "Xj_p_3": 0.32980,
        "w_p_3": 5.52757,
        "Positive electrode stoichiometry at 0% SOC": 0.85,
        "Positive electrode stoichiometry at 100% SOC": 0.1,
        "Positive electrode conductivity [S.m-1]": 0.18,
        "Maximum concentration in positive electrode [mol.m-3]": 63104.0,
        "Positive electrode diffusivity [m2.s-1]": 4e-15,
        "Positive electrode porosity": 0.335,
        "Positive electrode active material volume fraction": 0.665,
        "Positive particle radius [m]": 5.22e-06,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 0,
        "Positive electrode exchange-current density [A.m-2]" "": 5,
        "Positive electrode OCP entropic change [V.K-1]": 0.0,
        # separator
        "Separator porosity": 0.47,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
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
        "Lower voltage cut-off [V]": 1,
        "Upper voltage cut-off [V]": 5,
        "Initial temperature [K]": 298.15,
        "Initial voltage in negative electrode [V]": 0.085,
        "Initial voltage in positive electrode [V]": 4.35,
        "Initial concentration in negative electrode [mol.m-3]": 29820,
        "Initial concentration in positive electrode [mol.m-3]": 6310,
    }
