#
# A model to calculate electrode-specific SOH
#
import pybamm
import numpy as np
from functools import lru_cache
import warnings
from .base_electrode_soh import _BaseElectrodeSOH, BaseElectrodeSOHSolver


class _ElectrodeSOHMSMR(_BaseElectrodeSOH):
    """
    Model to calculate electrode-specific SOH using the MSMR formulation, see
    :class:`_ElectrodeSOH`.
    """

    def __init__(
        self, param=None, solve_for=None, known_value="cyclable lithium capacity"
    ):
        super().__init__()

        param = param or pybamm.LithiumIonParameters({"open-circuit potential": "MSMR"})
        solve_for = solve_for or ["Un_0", "Un_100"]

        if known_value == "cell capacity" and solve_for != ["Un_0", "Un_100"]:
            raise ValueError(
                "If known_value is 'cell capacity', solve_for must be "
                "['Un_0', 'Un_100']"
            )

        # Define parameters and input parameters
        x_n = param.n.prim.x
        x_p = param.p.prim.x

        V_max = param.voltage_high_cut
        V_min = param.voltage_low_cut
        Q_n = pybamm.InputParameter("Q_n")
        Q_p = pybamm.InputParameter("Q_p")

        if known_value == "cyclable lithium capacity":
            Q_Li = pybamm.InputParameter("Q_Li")
        elif known_value == "cell capacity":
            Q = pybamm.InputParameter("Q")

        # Define variables for 0% state of charge
        # TODO: thermal effects (include dU/dT)
        if "Un_0" in solve_for:
            Un_0 = pybamm.Variable("Un(x_0)")
            Up_0 = V_min + Un_0
            x_0 = x_n(Un_0)
            y_0 = x_p(Up_0)

        # Define variables for 100% state of charge
        # TODO: thermal effects (include dU/dT)
        if "Un_100" in solve_for:
            Un_100 = pybamm.Variable("Un(x_100)")
            Up_100 = V_max + Un_100
            x_100 = x_n(Un_100)
            y_100 = x_p(Up_100)
        else:
            Un_100 = pybamm.InputParameter("Un(x_100)")
            Up_100 = pybamm.InputParameter("Up(y_100)")
            x_100 = x_n(Un_100)
            y_100 = x_p(Up_100)

        # Define equations for 100% state of charge
        if "Un_100" in solve_for:
            if known_value == "cyclable lithium capacity":
                Un_100_eqn = Q_Li - y_100 * Q_p - x_100 * Q_n
            elif known_value == "cell capacity":
                Un_100_eqn = x_100 - x_0 - Q / Q_n
                Q_Li = y_100 * Q_p + x_100 * Q_n
            self.algebraic[Un_100] = Un_100_eqn
            self.initial_conditions[Un_100] = pybamm.Scalar(0)  # better ic?

        # These variables are defined in all cases
        self.variables = self.get_100_soc_variables(
            x_100, y_100, Un_100, Up_100, Q_Li, Q_n, Q_p, param
        )

        # Define equation for 0% state of charge
        if "Un_0" in solve_for:
            if known_value == "cyclable lithium capacity":
                Q = Q_n * (x_100 - x_0)
            self.algebraic[Un_0] = y_100 - y_0 + Q / Q_p
            self.initial_conditions[Un_0] = pybamm.Scalar(1)  # better ic?

            # These variables are only defined if x_0 is solved for
            self.variables.update(
                self.get_0_soc_variables(
                    x_0, y_0, x_100, y_100, Un_0, Up_0, Q, Q_n, Q_p, param
                )
            )


class ElectrodeSOHMSMRSolver(BaseElectrodeSOHSolver):
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
        Can be "cyclable lithium capacity" (default) or "cell capacity".
    """

    def __init__(
        self,
        parameter_values,
        param=None,
        known_value="cyclable lithium capacity",
    ):
        super.__init__(parameter_values, param, known_value)

    def _get_lims_ocp(self):
        return (1e-6, 1 - 1e-6, 1e-6, 1 - 1e-6)

    def __get_electrode_soh_sims_full(self):
        full_model = _ElectrodeSOHMSMR(param=self.param, known_value=self.known_value)
        return pybamm.Simulation(full_model, parameter_values=self.parameter_values)

    def __get_electrode_soh_sims_split(self):
        Un100_model = _ElectrodeSOHMSMR(
            param=self.param, solve_for=["Un_100"], known_value=self.known_value
        )
        Un0_model = _ElectrodeSOHMSMR(
            param=self.param, solve_for=["Un_0"], known_value=self.known_value
        )
        Un100_sim = pybamm.Simulation(
            Un100_model, parameter_values=self.parameter_values
        )
        Un0_sim = pybamm.Simulation(Un0_model, parameter_values=self.parameter_values)
        return [Un100_sim, Un0_sim]

    def _set_up_solve(self, inputs):
        # Try with full sim
        sim = self._get_electrode_soh_sims_full()
        if sim.solution is not None:
            Un_100_sol = sim.solution["Un_100"].data
            Un_0_sol = sim.solution["Un_0"].data
            Up_100_sol = sim.solution["Up_100"].data
            Up_0_sol = sim.solution["Up_0"].data
            return {
                "Un(x_100)": Un_100_sol,
                "Un(x_0)": Un_0_sol,
                "Up(x_100)": Up_100_sol,
                "Up(x_0)": Up_0_sol,
            }

        # Try with split sims
        if self.known_value == "cyclable lithium capacity":
            x100_sim, x0_sim = self._get_electrode_soh_sims_split()
            if x100_sim.solution is not None and x0_sim.solution is not None:
                Un_100_sol = x100_sim.solution["Un_100"].data
                Un_0_sol = x0_sim.solution["Un_0"].data
                Up_100_sol = x100_sim.solution["Up_100"].data
                Up_0_sol = x0_sim.solution["Up_0"].data
                return {
                    "Un(x_100)": Un_100_sol,
                    "Un(x_0)": Un_0_sol,
                    "Up(x_100)": Up_100_sol,
                    "Up(x_0)": Up_0_sol,
                }

        # Fall back to initial conditions calculated from limits
        x0_min, x100_max, y100_min, y0_max = self._get_lims(inputs)
        if self.known_value == "cyclable lithium capacity":
            # trial and error suggests theses are good values
            x100_init = np.minimum(x100_max, 0.8)
            x0_init = np.maximum(x0_min, 0.2)
            y100_init = np.maximum(y100_min, 0.2)
            y0_init = np.minimum(y0_max, 0.8)
        elif self.known_value == "cell capacity":
            # Use stoich limits based on cell capacity and
            # electrode capacities
            Q = inputs["Q"]
            Q_n = inputs["Q_n"]
            Q_p = inputs["Q_p"]
            x0_min = np.maximum(x0_min, 0.1)
            x100_max = np.minimum(x100_max, 0.9)
            y100_min = np.maximum(y100_min, 0.1)
            y0_max = np.minimum(y0_max, 0.9)
            x100_init = np.minimum(x0_min + Q / Q_n, 0.9)
            x0_init = np.maximum(x100_max - Q / Q_n, 0.1)
            y100_init = np.maximum(y0_max - Q / Q_p, 0.1)
            y0_init = np.minimum(y100_min + Q / Q_p, 0.9)
        # Get ocps from stoich limits
        Un0, Un100, Up100, Up0 = self._get_ocp_msmr(
            x0_init, x100_init, y100_init, y0_init
        )
        return {
            "Un(x_100)": Un100,
            "Un(x_0)": Un0,
            "Up(y_100)": Up100,
            "Up(y_0)": Up0,
        }

    def _solve_split(self, inputs, ics):
        Un100_sim, Un0_sim = self._get_electrode_soh_sims_split()
        Un100_sim.build()
        Un100_sim.built_model.set_initial_conditions_from(ics)
        Un100_sol = Un100_sim.solve([0], inputs=inputs)
        inputs["Un(x_100)"] = Un100_sol["Un(x_100)"].data[0]
        inputs["Up(y_100)"] = Un100_sol["Up(y_100)"].data[0]
        Un0_sim.build()
        Un0_sim.built_model.set_initial_conditions_from(ics)
        Un0_sol = Un0_sim.solve([0], inputs=inputs)
        return Un0_sol

    def _check_esoh_feasible(self, inputs):
        """
        Check that the electrode SOH calculation is feasible, based on voltage limits
        """
        x0_min, x100_max, y100_min, y0_max = self._get_lims(inputs)

        Un0, Un100, Up100, Up0 = self._get_ocp_msmr(x0_min, x100_max, y100_min, y0_max)
        V_lower_bound = float(Up0 - Un0)
        V_upper_bound = float(Up100 - Un100)

        # Check that the min and max achievable voltages span wider than the desired
        # voltage range
        if V_lower_bound > self.V_min:
            raise (
                ValueError(
                    f"The lower bound of the voltage, {V_lower_bound:.4f}V, "
                    f"is greater than the target minimum voltage, {self.V_min:.4f}V. "
                    f"Stoichiometry limits are x:[{x0_min:.4f}, {x100_max:.4f}], "
                    f"y:[{y100_min:.4f}, {y0_max:.4f}]."
                )
            )
        if V_upper_bound < self.V_max:
            raise (
                ValueError(
                    f"The upper bound of the voltage, {V_upper_bound:.4f}V, "
                    f"is less than the target maximum voltage, {self.V_max:.4f}V. "
                    f"Stoichiometry limits are x:[{x0_min:.4f}, {x100_max:.4f}], "
                    f"y:[{y100_min:.4f}, {y0_max:.4f}]."
                )
            )

    def _get_ocp_msmr(self, x0, x100, y100, y0):
        """
        Get the open-circuit potentials of the electrodes at the given stoichiometries
        """
        V_max = self.param.voltage_high_cut
        V_min = self.param.voltage_low_cut
        x_n = self.param.n.prim.x
        x_p = self.param.p.prim.x
        model = pybamm.BaseModel()
        Un_0 = pybamm.Variable("Un(x_0)")
        Un_100 = pybamm.Variable("Un(x_100)")
        Up_0 = pybamm.Variable("Up(y_0)")
        Up_100 = pybamm.Variable("Up(y_100)")
        model.algebraic = {
            Un_0: x_n(Un_0) - x0,
            Un_100: x_n(Un_100) - x100,
            Up_0: x_p(Up_0) - y0,
            Up_100: x_p(Up_100) - y100,
        }
        model.initial_conditions = {
            Un_0: pybamm.Scalar(1),
            Un_100: pybamm.Scalar(0),
            Up_0: V_min * pybamm.Scalar(1),
            Up_100: V_max,
        }
        model.variables = {
            "Un(x_100)": Un_100,
            "Un(x_0)": Un_0,
            "Up(y_100)": Up_100,
            "Up(y_0)": Up_0,
        }
        self.parameter_values.process_model(model)
        sol = pybamm.AlgebraicSolver().solve(model)

        return (
            sol["Un(x_0)"].data,
            sol["Un(x_100)"].data,
            sol["Up(y_100)"].data,
            sol["Up(y_0)"].data,
        )

    def get_initial_stoichiometries(self, initial_value):
        """
        Calculate initial stoichiometries to start off the simulation at a particular
        state of charge, given voltage limits, open-circuit potentials, etc defined by
        parameter_values

        Parameters
        ----------
        initial_value : float
            Target initial value.
            If integer, interpreted as SOC, must be between 0 and 1.
            If string e.g. "4 V", interpreted as voltage,
            must be between V_min and V_max.

        Returns
        -------
        x, y
            The initial stoichiometries that give the desired initial state of charge
        """
        parameter_values = self.parameter_values
        param = self.param
        x_0, x_100, y_100, y_0 = self.get_min_max_stoichiometries()

        if isinstance(initial_value, str) and initial_value.endswith("V"):
            # TODO
            raise NotImplementedError
        elif isinstance(initial_value, (int, float)):
            initial_soc = initial_value
            if not 0 <= initial_soc <= 1:
                raise ValueError("Initial SOC should be between 0 and 1")

        else:
            raise ValueError(
                "Initial value must be a float between 0 and 1, "
                "or a string ending in 'V'"
            )

        x = x_0 + initial_soc * (x_100 - x_0)
        y = y_0 - initial_soc * (y_0 - y_100)

        return x, y
