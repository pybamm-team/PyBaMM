#
# A model to calculate electrode-specific SOH
#
import pybamm
import numpy as np


class ElectrodeSOHx100(pybamm.BaseModel):
    def __init__(self, name="ElectrodeSOHx100 model"):
        pybamm.citations.register("Mohtat2019")
        super().__init__(name)

        param = pybamm.LithiumIonParameters()

        Un = param.n.U_dimensional
        Up = param.p.U_dimensional
        T_ref = param.T_ref

        n_Li = pybamm.InputParameter("n_Li")
        V_max = pybamm.InputParameter("V_max")
        Cn = pybamm.InputParameter("C_n")
        Cp = pybamm.InputParameter("C_p")
        x_100_init = pybamm.InputParameter("x_100_init")

        x_100 = pybamm.Variable("x_100")

        y_100 = (n_Li * param.F / 3600 - x_100 * Cn) / Cp

        self.algebraic = {
            x_100: Up(y_100, T_ref) - Un(x_100, T_ref) - V_max,
        }

        self.initial_conditions = {x_100: x_100_init}

        self.variables = {"x_100": x_100, "y_100": y_100}

    @property
    def default_solver(self):
        # Use AlgebraicSolver as CasadiAlgebraicSolver gives unnecessary warnings
        return pybamm.AlgebraicSolver()


class ElectrodeSOHC(pybamm.BaseModel):
    def __init__(self, name="ElectrodeSOHC model"):
        pybamm.citations.register("Mohtat2019")
        super().__init__(name)

        param = pybamm.LithiumIonParameters()

        Un = param.n.U_dimensional
        Up = param.p.U_dimensional
        T_ref = param.T_ref

        n_Li = pybamm.InputParameter("n_Li")
        V_min = pybamm.InputParameter("V_min")
        Cn = pybamm.InputParameter("C_n")
        Cp = pybamm.InputParameter("C_p")
        x_100 = pybamm.InputParameter("x_100")
        y_100 = pybamm.InputParameter("y_100")

        C = pybamm.Variable("C")
        x_0 = x_100 - C / Cn
        y_0 = y_100 + C / Cp

        self.algebraic = {C: Up(y_0, T_ref) - Un(x_0, T_ref) - V_min}

        self.initial_conditions = {C: pybamm.minimum(Cn * x_100 - 0.1, param.Q)}

        self.variables = {
            "C": C,
            "x_0": x_0,
            "y_0": y_0,
            "Un(x_0)": Un(x_0, T_ref),
            "Up(y_0)": Up(y_0, T_ref),
            "Up(y_0) - Un(x_0)": Up(y_0, T_ref) - Un(x_0, T_ref),
            "Up(y_100) - Un(x_100)": Up(y_100, T_ref) - Un(x_100, T_ref),
            "n_Li_100": 3600 / param.F * (y_100 * Cp + x_100 * Cn),
            "n_Li_0": 3600 / param.F * (y_0 * Cp + x_0 * Cn),
            "n_Li": n_Li,
            "x_100": x_100,
            "y_100": y_100,
            "C_n": Cn,
            "C_p": Cp,
            "C_n * (x_100 - x_0)": Cn * (x_100 - x_0),
            "C_p * (y_100 - y_0)": Cp * (y_0 - y_100),
        }

    @property
    def default_solver(self):
        # Use AlgebraicSolver as CasadiAlgebraicSolver gives unnecessary warnings
        return pybamm.AlgebraicSolver()


def solve_electrode_soh(x100_sim, C_sim, inputs, parameter_values):
    param = pybamm.LithiumIonParameters()

    Vmax = inputs["V_max"]
    Vmin = inputs["V_min"]
    Cp = inputs["C_p"]
    Cn = inputs["C_n"]
    n_Li = inputs["n_Li"]

    y_100_min = 1e-6
    x_100_upper_limit = ((n_Li * param.F) / 3600 - y_100_min * Cp) / Cn

    OCPp_data = isinstance(parameter_values["Positive electrode OCP [V]"], tuple)
    OCPn_data = isinstance(parameter_values["Negative electrode OCP [V]"], tuple)

    if OCPp_data:
        y_100_min = np.min(parameter_values["Positive electrode OCP [V]"][1][0])
        y_100_max = np.max(parameter_values["Positive electrode OCP [V]"][1][0])

        x_100_upper_limit = (
            n_Li * pybamm.constants.F.value / 3600 - y_100_min * Cp
        ) / Cn

        x_100_lower_limit = (
            n_Li * pybamm.constants.F.value / 3600 - y_100_max * Cp
        ) / Cn

        if OCPn_data:
            V_lower_bound = min(
                parameter_values["Positive electrode OCP [V]"][1][1]
            ) - max(parameter_values["Negative electrode OCP [V]"][1][1])

            V_upper_bound = max(
                parameter_values["Positive electrode OCP [V]"][1][1]
            ) - min(parameter_values["Negative electrode OCP [V]"][1][1])
        else:

            V_lower_bound = (
                min(parameter_values["Positive electrode OCP [V]"][1][1])
                - parameter_values["Negative electrode OCP [V]"](
                    x_100_upper_limit
                ).evaluate()
            )

            V_upper_bound = (
                max(parameter_values["Positive electrode OCP [V]"][1][1])
                - parameter_values["Negative electrode OCP [V]"](
                    x_100_lower_limit
                ).evaluate()
            )

    elif OCPn_data:
        x_100_min = np.min(parameter_values["Negative electrode OCP [V]"][1][0])
        x_100_max = np.max(parameter_values["Negative electrode OCP [V]"][1][0])

        y_100_upper_limit = (
            n_Li * pybamm.constants.F.value / 3600 - x_100_min * Cp
        ) / Cn

        y_100_lower_limit = (
            n_Li * pybamm.constants.F.value / 3600 - x_100_max * Cp
        ) / Cn

        V_lower_bound = parameter_values["Positive electrode OCP [V]"](
            y_100_lower_limit
        ).evaluate() - max(parameter_values["Negative electrode OCP [V]"][1][1])

        V_upper_bound = parameter_values["Positive electrode OCP [V]"](
            y_100_upper_limit
        ).evaluate() - min(parameter_values["Negative electrode OCP [V]"][1][1])

    if OCPp_data or OCPn_data:

        if V_lower_bound > Vmin:
            raise (
                ValueError(
                    "Initial values are outside bounds of OCP data in parameters."
                )
            )

        if V_upper_bound < Vmax:
            raise (
                ValueError(
                    "Initial values are outside bounds of OCP data in parameters."
                )
            )

    if not isinstance(x_100_upper_limit, float):
        x_100_upper_limit = x_100_upper_limit.value

    inputs.update({"x_100_init": min(0.99 * x_100_upper_limit, 1 - 1e-6)})

    x100_sol = x100_sim.solve([0], inputs=inputs)
    inputs["x_100"] = x100_sol["x_100"].data[0]
    inputs["y_100"] = x100_sol["y_100"].data[0]
    C_sol = C_sim.solve([0], inputs=inputs)
    return C_sol


def get_initial_stoichiometries(initial_soc, parameter_values):
    """
    Calculate initial stoichiometries to start off the simulation at a particular
    state of charge, given voltage limits, open-circuit potentials, etc defined by
    parameter_values

    Parameters
    ----------
    initial_soc : float
        Target initial SOC. Must be between 0 and 1.
    parameter_values : :class:`pybamm.ParameterValues`
        The parameter values class that will be used for the simulation. Required for
        calculating appropriate initial stoichiometries.

    Returns
    -------
    x, y
        The initial stoichiometries that give the desired initial state of charge
    """
    if initial_soc < 0 or initial_soc > 1:
        raise ValueError("Initial SOC should be between 0 and 1")

    param = pybamm.LithiumIonParameters()

    V_min = parameter_values.evaluate(param.voltage_low_cut_dimensional)
    V_max = parameter_values.evaluate(param.voltage_high_cut_dimensional)
    C_n = parameter_values.evaluate(param.n.cap_init)
    C_p = parameter_values.evaluate(param.p.cap_init)
    n_Li = parameter_values.evaluate(param.n_Li_particles_init)

    model_x100 = ElectrodeSOHx100()
    model_C = ElectrodeSOHC()

    x100_sim = pybamm.Simulation(model_x100, parameter_values=parameter_values)
    C_sim = pybamm.Simulation(model_C, parameter_values=parameter_values)

    inputs = {
        "V_min": V_min,
        "V_max": V_max,
        "C_n": C_n,
        "C_p": C_p,
        "n_Li": n_Li,
    }

    # Solve the model and check outputs
    sol = solve_electrode_soh(x100_sim, C_sim, inputs, parameter_values)

    x_0 = sol["x_0"].data[0]
    y_0 = sol["y_0"].data[0]
    C = sol["C"].data[0]
    x = x_0 + initial_soc * C / C_n
    y = y_0 - initial_soc * C / C_p

    return x, y
