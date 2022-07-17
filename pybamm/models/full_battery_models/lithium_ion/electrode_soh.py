#
# A model to calculate electrode-specific SOH
#
from tabnanny import check
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


def create_electrode_soh_sims(parameter_values):
    x100_model = pybamm.lithium_ion.ElectrodeSOHx100()
    x100_sim = pybamm.Simulation(x100_model, parameter_values=parameter_values)
    C_model = pybamm.lithium_ion.ElectrodeSOHC()
    C_sim = pybamm.Simulation(C_model, parameter_values=parameter_values)
    return [x100_sim, C_sim]


def solve_electrode_soh(x100_sim, C_sim, inputs):
    x0_min, x100_max, _, _ = check_esoh_feasible(C_sim.parameter_values, inputs)

    x100_init = x100_max
    if x100_sim.solution is not None:
        x100_init_sol = x100_sim.solution["x_100"].data[0]
        # Update the initial condition if it is valid
        if x0_min < x100_init_sol < x0_min:
            x100_init = x100_init_sol

    inputs.update({"x_100_init": x100_init})

    x100_sol = x100_sim.solve([0], inputs=inputs)
    inputs["x_100"] = x100_sol["x_100"].data[0]
    inputs["y_100"] = x100_sol["y_100"].data[0]
    C_sol = C_sim.solve([0], inputs=inputs)

    # print(inputs)
    # print({k: C_sol[k].data[0] for k in ["x_0", "y_0", "x_100", "y_100", "C"]})
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

    model = pybamm.lithium_ion.ElectrodeSOH()

    param = pybamm.LithiumIonParameters()
    sim = pybamm.Simulation(model, parameter_values=parameter_values)

    V_min = parameter_values.evaluate(param.voltage_low_cut_dimensional)
    V_max = parameter_values.evaluate(param.voltage_high_cut_dimensional)
    C_n = parameter_values.evaluate(param.n.cap_init)
    C_p = parameter_values.evaluate(param.p.cap_init)
    n_Li = parameter_values.evaluate(param.n_Li_particles_init)

    # Solve the model and check outputs
    sol = sim.solve(
        [0],
        inputs={
            "V_min": V_min,
            "V_max": V_max,
            "C_n": C_n,
            "C_p": C_p,
            "n_Li": n_Li,
        },
    )

    x_0 = sol["x_0"].data[0]
    y_0 = sol["y_0"].data[0]
    C = sol["C"].data[0]
    x = x_0 + initial_soc * C / C_n
    y = y_0 - initial_soc * C / C_p

    return x, y


def check_esoh_feasible(parameter_values, inputs):
    param = pybamm.LithiumIonParameters()

    Vmax = inputs["V_max"]
    Vmin = inputs["V_min"]
    Cp = inputs["C_p"]
    Cn = inputs["C_n"]
    n_Li = inputs["n_Li"]

    OCPp_data = isinstance(parameter_values["Positive electrode OCP [V]"], tuple)
    OCPn_data = isinstance(parameter_values["Negative electrode OCP [V]"], tuple)

    if OCPp_data:
        y100_min = np.min(parameter_values["Positive electrode OCP [V]"][1][1])
        y0_max = np.max(parameter_values["Positive electrode OCP [V]"][1][1])
    else:
        y100_min = 1e-6
        y0_max = 1 - 1e-6

    if OCPn_data:
        x0_min = np.min(parameter_values["Negative electrode OCP [V]"][1][1])
        x100_max = np.max(parameter_values["Negative electrode OCP [V]"][1][1])
    else:
        x0_min = 1e-6
        x100_max = 1 - 1e-6

    F = pybamm.constants.F.value
    x100_max_from_y100_min = (n_Li * F / 3600 - y100_min * Cp) / Cn
    x0_min_from_y0_max = (n_Li * F / 3600 - y0_max * Cp) / Cn
    y100_min_from_x100_max = (n_Li * F / 3600 - x100_max * Cn) / Cp
    y0_max_from_x0_min = (n_Li * F / 3600 - x0_min * Cn) / Cp

    x100_max = min(x100_max_from_y100_min, x100_max)
    x0_min = max(x0_min_from_y0_max, x0_min)
    y100_min = max(y100_min_from_x100_max, y100_min)
    y0_max = min(y0_max_from_x0_min, y0_max)

    T = parameter_values["Reference temperature [K]"]
    V_lower_bound = parameter_values.evaluate(
        param.p.U_dimensional(y0_max, T) - param.n.U_dimensional(x0_min, T)
    )
    V_upper_bound = parameter_values.evaluate(
        param.p.U_dimensional(y100_min, T) - param.n.U_dimensional(x100_max, T)
    )

    if V_lower_bound > Vmin:
        raise (
            ValueError(
                f"The lower bound of the voltage, {V_lower_bound:.4f}V, "
                f"is greater than the target minimum voltage, {Vmin:.4f}V. "
                f"Stoichiometry limits are x:[{x0_min:.4f}, {x100_max:.4f}], "
                f"y:[{y100_min:.4f}, {y0_max:.4f}]."
            )
        )
    if V_upper_bound < Vmax:
        raise (
            ValueError(
                f"The upper bound of the voltage, {V_upper_bound:.4f}V, "
                f"is less than the target maximum voltage, {Vmax:.4f}V. "
                f"Stoichiometry limits are x:[{x0_min:.4f}, {x100_max:.4f}], "
                f"y:[{y100_min:.4f}, {y0_max:.4f}]."
            )
        )

    return (x0_min, x100_max, y100_min, y0_max)
