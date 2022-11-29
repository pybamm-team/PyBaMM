#
# A model to calculate electrode-specific SOH
#
import pybamm
import numpy as np
from functools import lru_cache
import warnings


class _ElectrodeSOH(pybamm.BaseModel):
    """Model to calculate electrode-specific SOH, from [1]_.
    This model is mainly for internal use, to calculate summary variables in a
    simulation.

    .. math::
        C_{Li} = y_{100}C_p + x_{100}C_n,
    .. math::
        V_{max} = U_p(y_{100}) - U_n(x_{100}),
    .. math::
        V_{min} = U_p(y_{0}) - U_n(x_{0}),
    .. math::
        x_0 = x_{100} - \\frac{C}{C_n},
    .. math::
        y_0 = y_{100} + \\frac{C}{C_p}.

    References
    ----------
    .. [1] Mohtat, P., Lee, S., Siegel, J. B., & Stefanopoulou, A. G. (2019). Towards
           better estimability of electrode-specific state of health: Decoding the cell
           expansion. Journal of Power Sources, 427, 101-111.

    **Extends:** :class:`pybamm.BaseModel`
    """

    def __init__(self, param=None, solve_for=None, known_value="C_Li"):
        pybamm.citations.register("Mohtat2019")
        name = "ElectrodeSOH model"
        super().__init__(name)

        param = param or pybamm.LithiumIonParameters()
        solve_for = solve_for or ["x_0", "x_100"]

        # Define parameters and input parameters
        Un = param.n.prim.U_dimensional
        Up = param.p.prim.U_dimensional
        T_ref = param.T_ref

        V_max = pybamm.InputParameter("V_max")
        V_min = pybamm.InputParameter("V_min")
        Cn = pybamm.InputParameter("C_n")
        Cp = pybamm.InputParameter("C_p")

        if known_value == "C_Li":
            C_Li = pybamm.InputParameter("C_Li")
        elif known_value == "C":
            C = pybamm.InputParameter("C")

        # Define variables for 100% state of charge
        if "x_100" in solve_for:
            x_100 = pybamm.Variable("x_100")
            if known_value == "C_Li":
                y_100 = (C_Li - x_100 * Cn) / Cp
            elif known_value == "C":
                y_100 = pybamm.Variable("y_100")
                C_Li = y_100 * Cp + x_100 * Cn
        else:
            x_100 = pybamm.InputParameter("x_100")
            y_100 = pybamm.InputParameter("y_100")
        Un_100 = Un(x_100, T_ref)
        Up_100 = Up(y_100, T_ref)

        # Define equations for 100% state of charge
        if "x_100" in solve_for:
            self.algebraic[x_100] = Up_100 - Un_100 - V_max
            self.initial_conditions[x_100] = pybamm.Scalar(0.9)

        # These variables are defined in all cases
        self.variables = {
            "x_100": x_100,
            "y_100": y_100,
            "Un(x_100)": Un_100,
            "Up(y_100)": Up_100,
            "Up(y_100) - Un(x_100)": Up_100 - Un_100,
            "C_Li": C_Li,
            "n_Li": C_Li * 3600 / param.F,
            "C_n": Cn,
            "C_p": Cp,
        }

        # Define variables and equations for 0% state of charge
        if "x_0" in solve_for:
            if known_value == "C_Li":
                x_0 = pybamm.Variable("x_0")
                C = Cn * (x_100 - x_0)
            elif known_value == "C":
                x_0 = x_100 - C / Cn
                C_Li = y_100 * Cp + x_0 * Cn
            y_0 = y_100 + C / Cp
            Un_0 = Un(x_0, T_ref)
            Up_0 = Up(y_0, T_ref)
            if known_value == "C_Li":
                # the variable we are solving for is x0, since y_100 is calculated
                # based on C_Li
                var = x_0
            elif known_value == "C":
                # the variable we are solving for is y_100, since x_0 is calculated
                # based on C
                var = y_100
            self.algebraic[var] = Up_0 - Un_0 - V_min
            self.initial_conditions[var] = pybamm.Scalar(0.1)

            # These variables are only defined if x_0 is solved for
            self.variables.update(
                {
                    "C": C,
                    "Capacity [A.h]": C,
                    "x_0": x_0,
                    "y_0": y_0,
                    "Un(x_0)": Un_0,
                    "Up(y_0)": Up_0,
                    "Up(y_0) - Un(x_0)": Up_0 - Un_0,
                    "x_100 - x_0": x_100 - x_0,
                    "y_0 - y_100": y_0 - y_100,
                    "C_n * (x_100 - x_0)": Cn * (x_100 - x_0),
                    "C_p * (y_0 - y_100)": Cp * (y_0 - y_100),
                    "Negative electrode excess capacity ratio": Cn / C,
                    "Positive electrode excess capacity ratio": Cp / C,
                }
            )

    @property
    def default_solver(self):
        # Use AlgebraicSolver as CasadiAlgebraicSolver gives unnecessary warnings
        return pybamm.AlgebraicSolver()


class ElectrodeSOHSolver:
    """
    Class used to check if the electrode SOH model is feasible, and solve it if it is.

    Parameters
    ----------
    parameter_values : :class:`pybamm.ParameterValues.Parameters`
        The parameters of the simulation
    param : :class:`pybamm.LithiumIonParameters`, optional
        Specific instance of the symbolic lithium-ion parameter class. If not provided,
        the default set of symbolic lithium-ion parameters will be used.
    known_value : str, optional
        The known value needed to complete the electrode SOH model.
        Can be "C_Li" (total lithium is known, e.g. from initial concentrations) or
        "C" (capacity is known, e.g. from nominal capacity). Default is "C_Li".

    """

    def __init__(self, parameter_values, param=None, known_value="C_Li"):
        self.parameter_values = parameter_values
        self.param = param or pybamm.LithiumIonParameters()
        self.known_value = known_value

        # Check whether each electrode OCP is a function (False) or data (True)
        OCPp_data = isinstance(parameter_values["Positive electrode OCP [V]"], tuple)
        OCPn_data = isinstance(parameter_values["Negative electrode OCP [V]"], tuple)

        # Calculate stoich limits for the open circuit potentials
        if OCPp_data:
            Up_sto = parameter_values["Positive electrode OCP [V]"][1][0]
            y100_min = max(np.min(Up_sto), 0) + 1e-6
            y0_max = min(np.max(Up_sto), 1) - 1e-6
        else:
            y100_min = 1e-6
            y0_max = 1 - 1e-6

        if OCPn_data:
            Un_sto = parameter_values["Negative electrode OCP [V]"][1][0]
            x0_min = max(np.min(Un_sto), 0) + 1e-6
            x100_max = min(np.max(Un_sto), 1) - 1e-6
        else:
            x0_min = 1e-6
            x100_max = 1 - 1e-6

        self.lims_ocp = (x0_min, x100_max, y100_min, y0_max)
        self.OCV_function = None

    @lru_cache
    def _get_electrode_soh_sims_full(self):
        full_model = _ElectrodeSOH(param=self.param, known_value=self.known_value)
        return pybamm.Simulation(full_model, parameter_values=self.parameter_values)

    @lru_cache
    def _get_electrode_soh_sims_split(self):
        x100_model = _ElectrodeSOH(
            param=self.param, solve_for=["x_100"], known_value=self.known_value
        )
        x100_sim = pybamm.Simulation(x100_model, parameter_values=self.parameter_values)
        x0_model = _ElectrodeSOH(
            param=self.param, solve_for=["x_0"], known_value=self.known_value
        )
        x0_sim = pybamm.Simulation(x0_model, parameter_values=self.parameter_values)
        return [x100_sim, x0_sim]

    def solve(self, inputs):
        if "n_Li" in inputs:
            warnings.warn(
                "Input 'n_Li' has been replaced by 'C_Li', which is 'n_Li * F / 3600'. "
                "This will be automatically calculated for now."
                "C_Li can be calculated from parameters as 'param.C_Li_particles_init'",
                DeprecationWarning,
            )
            n_Li = inputs.pop("n_Li")
            inputs["C_Li"] = n_Li * self.param.F.value / 3600
        ics = self._set_up_solve(inputs)
        try:
            sol = self._solve_full(inputs, ics)
        except pybamm.SolverError:
            # just in case solving one by one works better
            try:
                sol = self._solve_split(inputs, ics)
            except pybamm.SolverError as original_error:
                # check if the error is due to the simulation not being feasible
                self._check_esoh_feasible(inputs)
                # if that didn't raise an error, raise the original error instead
                raise original_error  # pragma: no cover (don't know how to get here)

        return sol

    def _set_up_solve(self, inputs):
        sim = self._get_electrode_soh_sims_full()
        if sim.solution is not None:
            return {
                var: sim.solution[var].data for var in ["x_100", "x_0", "y_100", "y_0"]
            }
        else:
            x0_init, x100_init, y100_init, y0_init = self._get_lims(inputs)
            return {
                "x_100": np.array(x100_init),
                "x_0": np.array(x0_init),
                "y_100": np.array(y100_init),
                "y_0": np.array(y0_init),
            }

    def _solve_full(self, inputs, ics):
        sim = self._get_electrode_soh_sims_full()
        sim.build()
        sim.built_model.set_initial_conditions_from(ics)
        sol = sim.solve([0], inputs=inputs)
        return sol

    def _solve_split(self, inputs, ics):
        x100_sim, x0_sim = self._get_electrode_soh_sims_split()
        x100_sim.build()
        x100_sim.built_model.set_initial_conditions_from(ics)
        x100_sol = x100_sim.solve([0], inputs=inputs)

        inputs["x_100"] = x100_sol["x_100"].data[0]
        inputs["y_100"] = x100_sol["y_100"].data[0]
        x0_sim.build()
        x0_sim.built_model.set_initial_conditions_from(ics)
        x0_sol = x0_sim.solve([0], inputs=inputs)

        return x0_sol

    def _get_lims(self, inputs):
        """
        Get stoichiometry limits based on C_Li, C_n, and C_p
        """
        Cp = inputs["C_p"]
        Cn = inputs["C_n"]

        x0_min, x100_max, y100_min, y0_max = self.lims_ocp

        if self.known_value == "C_Li":
            C_Li = inputs["C_Li"]
            C_Li_min = Cn * x0_min + Cp * y100_min
            C_Li_max = Cn * x100_max + Cp * y0_max
            if not C_Li_min <= C_Li <= C_Li_max:
                raise ValueError(
                    f"C_Li={C_Li:.4f} Ah is outside the range of possible values. "
                    f"C_Li_min = {C_Li_min:.4f} Ah, C_Li_max = {C_Li_max:.4f} Ah."
                )
            if C_Li > Cp:
                warnings.warn(f"C_Li={C_Li:.4f} Ah is greater than C_p={Cp:.4f} Ah.")

            # Update (tighten) stoich limits based on total lithium content and electrode
            # capacities
            x100_max_from_y100_min = (C_Li - y100_min * Cp) / Cn
            x0_min_from_y0_max = (C_Li - y0_max * Cp) / Cn
            y100_min_from_x100_max = (C_Li - x100_max * Cn) / Cp
            y0_max_from_x0_min = (C_Li - x0_min * Cn) / Cp

            x100_max = min(x100_max_from_y100_min, x100_max)
            x0_min = max(x0_min_from_y0_max, x0_min)
            y100_min = max(y100_min_from_x100_max, y100_min)
            y0_max = min(y0_max_from_x0_min, y0_max)
        elif self.known_value == "C":
            C = inputs["C"]
            C_max = min(Cn * (x100_max - x0_min), Cp * (y0_max - y100_min))
            if C > C_max:
                raise ValueError(
                    f"C={C:.4f} Ah is larger than the maximum possible capacity "
                    f"C_max={C_max:.4f} Ah."
                )

        # Check stoich limits are between 0 and 1
        if not (0 < x0_min < x100_max < 1 and 0 < y100_min < y0_max < 1):
            raise ValueError(
                "'0 < x0_min < x100_max < 1' is False for "
                f"x0_min={x0_min:.4f} and x100_max={x100_max:.4f} "
                "or '0 < y100_min < y0_max < 1' is False for "
                f"y100_min={y100_min:.4f} and y0_max={y0_max:.4f}"
            )

        return (x0_min, x100_max, y100_min, y0_max)

    def _check_esoh_feasible(self, inputs):
        """
        Check that the electrode SOH calculation is feasible, based on voltage limits
        """
        x0_min, x100_max, y100_min, y0_max = self._get_lims(inputs)
        Vmax = inputs["V_max"]
        Vmin = inputs["V_min"]

        # Parameterize the OCP functions
        if self.OCV_function is None:
            T = self.parameter_values["Reference temperature [K]"]
            x = pybamm.InputParameter("x")
            y = pybamm.InputParameter("y")
            self.OCV_function = self.parameter_values.process_symbol(
                self.param.p.prim.U_dimensional(y, T)
                - self.param.n.prim.U_dimensional(x, T)
            )

        # Check that the min and max achievable voltages span wider than the desired
        # voltage range
        V_lower_bound = float(
            self.OCV_function.evaluate(inputs={"x": x0_min, "y": y0_max})
        )
        V_upper_bound = float(
            self.OCV_function.evaluate(inputs={"x": x100_max, "y": y100_min})
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


def get_initial_stoichiometries(initial_soc, parameter_values, param=None):
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
    param : :class:`pybamm.LithiumIonParameters`, optional
        The symbolic parameter set to use for the simulation.
        If not provided, the default parameter set will be used.

    Returns
    -------
    x, y
        The initial stoichiometries that give the desired initial state of charge
    """
    if not 0 <= initial_soc <= 1:
        raise ValueError("Initial SOC should be between 0 and 1")

    param = param or pybamm.LithiumIonParameters()

    V_min = parameter_values.evaluate(param.voltage_low_cut_dimensional)
    V_max = parameter_values.evaluate(param.voltage_high_cut_dimensional)
    C_n = parameter_values.evaluate(param.n.cap_init)
    C_p = parameter_values.evaluate(param.p.cap_init)
    C_Li = parameter_values.evaluate(param.C_Li_particles_init)

    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values, param)

    inputs = {"V_min": V_min, "V_max": V_max, "C_n": C_n, "C_p": C_p, "C_Li": C_Li}

    # Solve the model and check outputs
    sol = esoh_solver.solve(inputs)

    x_0 = sol["x_0"].data[0]
    y_0 = sol["y_0"].data[0]
    C = sol["C"].data[0]
    x = x_0 + initial_soc * C / C_n
    y = y_0 - initial_soc * C / C_p

    return x, y
