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
        Q_{Li} = y_{100}Q_p + x_{100}Q_n,
    .. math::
        V_{max} = U_p(y_{100}) - U_n(x_{100}),
    .. math::
        V_{min} = U_p(y_{0}) - U_n(x_{0}),
    .. math::
        x_0 = x_{100} - \\frac{Q}{Q_n},
    .. math::
        y_0 = y_{100} + \\frac{Q}{Q_p}.

    References
    ----------
    .. [1] Mohtat, P., Lee, S., Siegel, J. B., & Stefanopoulou, A. G. (2019). Towards
           better estimability of electrode-specific state of health: Decoding the cell
           expansion. Journal of Power Sources, 427, 101-111.
    """

    def __init__(
        self, param=None, solve_for=None, known_value="cyclable lithium capacity"
    ):
        pybamm.citations.register("Mohtat2019")
        name = "ElectrodeSOH model"
        super().__init__(name)

        param = param or pybamm.LithiumIonParameters()
        solve_for = solve_for or ["x_0", "x_100"]

        if known_value == "cell capacity" and solve_for != ["x_0", "x_100"]:
            raise ValueError(
                "If known_value is 'cell capacity', solve_for must be ['x_0', 'x_100']"
            )

        # Define parameters and input parameters
        Un = param.n.prim.U
        Up = param.p.prim.U
        T_ref = param.T_ref

        V_max = param.voltage_high_cut
        V_min = param.voltage_low_cut
        Q_n = pybamm.InputParameter("Q_n")
        Q_p = pybamm.InputParameter("Q_p")

        if known_value == "cyclable lithium capacity":
            Q_Li = pybamm.InputParameter("Q_Li")
        elif known_value == "cell capacity":
            Q = pybamm.InputParameter("Q")

        # Define variables for 100% state of charge
        if "x_100" in solve_for:
            x_100 = pybamm.Variable("x_100")
            if known_value == "cyclable lithium capacity":
                y_100 = (Q_Li - x_100 * Q_n) / Q_p
            elif known_value == "cell capacity":
                y_100 = pybamm.Variable("y_100")
                Q_Li = y_100 * Q_p + x_100 * Q_n
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
            "Q_Li": Q_Li,
            "n_Li": Q_Li * 3600 / param.F,
            "Q_n": Q_n,
            "Q_p": Q_p,
        }

        # Define variables and equations for 0% state of charge
        if "x_0" in solve_for:
            if known_value == "cyclable lithium capacity":
                x_0 = pybamm.Variable("x_0")
                Q = Q_n * (x_100 - x_0)
                # the variable we are solving for is x0, since y_100 is calculated
                # based on Q_Li
                var = x_0
            elif known_value == "cell capacity":
                x_0 = x_100 - Q / Q_n
                Q_Li = y_100 * Q_p + x_0 * Q_n
                # the variable we are solving for is y_100, since x_0 is calculated
                # based on Q
                var = y_100
            y_0 = y_100 + Q / Q_p
            Un_0 = Un(x_0, T_ref)
            Up_0 = Up(y_0, T_ref)
            self.algebraic[var] = Up_0 - Un_0 - V_min
            self.initial_conditions[var] = pybamm.Scalar(0.1)

            # These variables are only defined if x_0 is solved for
            self.variables.update(
                {
                    "Q": Q,
                    "Capacity [A.h]": Q,
                    "x_0": x_0,
                    "y_0": y_0,
                    "Un(x_0)": Un_0,
                    "Up(y_0)": Up_0,
                    "Up(y_0) - Un(x_0)": Up_0 - Un_0,
                    "x_100 - x_0": x_100 - x_0,
                    "y_0 - y_100": y_0 - y_100,
                    "Q_n * (x_100 - x_0)": Q_n * (x_100 - x_0),
                    "Q_p * (y_0 - y_100)": Q_p * (y_0 - y_100),
                    "Negative electrode excess capacity ratio": Q_n / Q,
                    "Positive electrode excess capacity ratio": Q_p / Q,
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
        Can be "cyclable lithium capacity" (default) or "cell capacity".

    """

    def __init__(
        self, parameter_values, param=None, known_value="cyclable lithium capacity"
    ):
        self.parameter_values = parameter_values
        self.param = param or pybamm.LithiumIonParameters()
        self.known_value = known_value

        # Check whether each electrode OCP is a function (False) or data (True)
        OCPp_data = isinstance(parameter_values["Positive electrode OCP [V]"], tuple)
        OCPn_data = isinstance(parameter_values["Negative electrode OCP [V]"], tuple)

        # Calculate stoich limits for the open-circuit potentials
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
                "Input 'n_Li' has been replaced by 'Q_Li', which is 'n_Li * F / 3600'. "
                "This will be automatically calculated for now. "
                "Q_Li can be read from parameters as 'param.Q_Li_particles_init'",
                DeprecationWarning,
            )
            n_Li = inputs.pop("n_Li")
            inputs["Q_Li"] = n_Li * self.param.F.value / 3600
        if "C_n" in inputs:
            warnings.warn("Input 'C_n' has been renamed to 'Q_n'", DeprecationWarning)
            inputs["Q_n"] = inputs.pop("C_n")
        if "C_p" in inputs:
            warnings.warn("Input 'C_p' has been renamed to 'Q_p'", DeprecationWarning)
            inputs["Q_p"] = inputs.pop("C_p")
        if inputs.pop("V_min", None) is not None:
            warnings.warn(
                "V_min has been removed from the inputs. "
                "The 'Lower voltage cut-off [V]' parameter is now used automatically.",
                DeprecationWarning,
            )
        if inputs.pop("V_max", None) is not None:
            warnings.warn(
                "V_max has been removed from the inputs. "
                "The 'Upper voltage cut-off [V]' parameter is now used automatically.",
                DeprecationWarning,
            )
        ics = self._set_up_solve(inputs)
        try:
            sol = self._solve_full(inputs, ics)
        except pybamm.SolverError:
            # just in case solving one by one works better
            try:
                sol = self._solve_split(inputs, ics)
            except pybamm.SolverError as split_error:
                # check if the error is due to the simulation not being feasible
                self._check_esoh_feasible(inputs)
                # if that didn't raise an error, raise the original error instead
                raise split_error

        sol_dict = {key: sol[key].data[0] for key in sol.all_models[0].variables.keys()}
        return sol_dict

    def _set_up_solve(self, inputs):
        # Try with full sim
        sim = self._get_electrode_soh_sims_full()
        if sim.solution is not None:
            x100_sol = sim.solution["x_100"].data
            x0_sol = sim.solution["x_0"].data
            y100_sol = sim.solution["y_100"].data
            y0_sol = sim.solution["y_0"].data
            return {"x_100": x100_sol, "x_0": x0_sol, "y_100": y100_sol, "y_0": y0_sol}

        # Try with split sims
        if self.known_value == "cyclable lithium capacity":
            x100_sim, x0_sim = self._get_electrode_soh_sims_split()
            if x100_sim.solution is not None and x0_sim.solution is not None:
                x100_sol = x100_sim.solution["x_100"].data
                x0_sol = x0_sim.solution["x_0"].data
                y100_sol = x100_sim.solution["y_100"].data
                y0_sol = x0_sim.solution["y_0"].data
                return {
                    "x_100": x100_sol,
                    "x_0": x0_sol,
                    "y_100": y100_sol,
                    "y_0": y0_sol,
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
        return {"x_100": x100_init, "x_0": x0_init, "y_100": y100_init, "y_0": y0_init}

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
        Get stoichiometry limits based on Q_Li, Q_n, and Q_p
        """
        Q_p = inputs["Q_p"]
        Q_n = inputs["Q_n"]

        x0_min, x100_max, y100_min, y0_max = self.lims_ocp

        if self.known_value == "cyclable lithium capacity":
            Q_Li = inputs["Q_Li"]
            Q_Li_min = Q_n * x0_min + Q_p * y100_min
            Q_Li_max = Q_n * x100_max + Q_p * y0_max
            if not Q_Li_min <= Q_Li <= Q_Li_max:
                raise ValueError(
                    f"Q_Li={Q_Li:.4f} Ah is outside the range of possible values "
                    f"[{Q_Li_min:.4f}, {Q_Li_max:.4f}]."
                )
            if Q_Li > Q_p:
                warnings.warn(f"Q_Li={Q_Li:.4f} Ah is greater than Q_p={Q_p:.4f} Ah.")

            # Update (tighten) stoich limits based on total lithium content and
            # electrode capacities
            x100_max_from_y100_min = (Q_Li - y100_min * Q_p) / Q_n
            x0_min_from_y0_max = (Q_Li - y0_max * Q_p) / Q_n
            y100_min_from_x100_max = (Q_Li - x100_max * Q_n) / Q_p
            y0_max_from_x0_min = (Q_Li - x0_min * Q_n) / Q_p

            x100_max = min(x100_max_from_y100_min, x100_max)
            x0_min = max(x0_min_from_y0_max, x0_min)
            y100_min = max(y100_min_from_x100_max, y100_min)
            y0_max = min(y0_max_from_x0_min, y0_max)
        elif self.known_value == "cell capacity":
            Q = inputs["Q"]
            Q_max = min(Q_n * (x100_max - x0_min), Q_p * (y0_max - y100_min))
            if Q > Q_max:
                raise ValueError(
                    f"Q={Q:.4f} Ah is larger than the maximum possible capacity "
                    f"Q_max={Q_max:.4f} Ah."
                )

        # Check stoich limits are between 0 and 1
        if not (0 < x0_min < x100_max < 1 and 0 < y100_min < y0_max < 1):
            raise ValueError(
                "'0 < x0_min < x100_max < 1' is False for "
                f"x0_min={x0_min:.4f} and x100_max={x100_max:.4f} "
                "or '0 < y100_min < y0_max < 1' is False for "
                f"y100_min={y100_min:.4f} and y0_max={y0_max:.4f}"
            )  # pragma: no cover

        return (x0_min, x100_max, y100_min, y0_max)

    def _check_esoh_feasible(self, inputs):
        """
        Check that the electrode SOH calculation is feasible, based on voltage limits
        """
        x0_min, x100_max, y100_min, y0_max = self._get_lims(inputs)

        # Parameterize the OCP functions
        if self.OCV_function is None:
            T = self.parameter_values["Reference temperature [K]"]
            x = pybamm.InputParameter("x")
            y = pybamm.InputParameter("y")
            self.V_max = self.parameter_values.evaluate(self.param.voltage_high_cut)
            self.V_min = self.parameter_values.evaluate(self.param.voltage_low_cut)
            self.OCV_function = self.parameter_values.process_symbol(
                self.param.p.prim.U(y, T) - self.param.n.prim.U(x, T)
            )

        # Check that the min and max achievable voltages span wider than the desired
        # voltage range
        V_lower_bound = float(
            self.OCV_function.evaluate(inputs={"x": x0_min, "y": y0_max})
        )
        V_upper_bound = float(
            self.OCV_function.evaluate(inputs={"x": x100_max, "y": y100_min})
        )
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
            V_init = float(initial_value[:-1])
            V_min = parameter_values.evaluate(param.voltage_low_cut)
            V_max = parameter_values.evaluate(param.voltage_high_cut)

            if not V_min < V_init < V_max:
                raise ValueError(
                    f"Initial voltage {V_init}V is outside the voltage limits "
                    f"({V_min}, {V_max})"
                )

            # Solve simple model for initial soc based on target voltage
            soc_model = pybamm.BaseModel()
            soc = pybamm.Variable("soc")
            Up = param.p.prim.U
            Un = param.n.prim.U
            T_ref = parameter_values["Reference temperature [K]"]
            x = x_0 + soc * (x_100 - x_0)
            y = y_0 - soc * (y_0 - y_100)

            soc_model.algebraic[soc] = Up(y, T_ref) - Un(x, T_ref) - V_init
            # initial guess for soc linearly interpolates between 0 and 1
            # based on V linearly interpolating between V_max and V_min
            soc_model.initial_conditions[soc] = (V_init - V_min) / (V_max - V_min)
            soc_model.variables["soc"] = soc
            parameter_values.process_model(soc_model)
            initial_soc = pybamm.AlgebraicSolver().solve(soc_model, [0])["soc"].data[0]
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

    def get_min_max_stoichiometries(self):
        """
        Calculate min/max stoichiometries
        given voltage limits, open-circuit potentials, etc defined by parameter_values

        Returns
        -------
        x_0, x_100, y_100, y_0
            The min/max stoichiometries
        """
        parameter_values = self.parameter_values
        param = self.param

        Q_n = parameter_values.evaluate(param.n.Q_init)
        Q_p = parameter_values.evaluate(param.p.Q_init)

        if self.known_value == "cyclable lithium capacity":
            Q_Li = parameter_values.evaluate(param.Q_Li_particles_init)
            inputs = {"Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}
        elif self.known_value == "cell capacity":
            Q = parameter_values.evaluate(param.Q / param.n_electrodes_parallel)
            inputs = {"Q_n": Q_n, "Q_p": Q_p, "Q": Q}
        # Solve the model and check outputs
        sol = self.solve(inputs)
        return [sol["x_0"], sol["x_100"], sol["y_100"], sol["y_0"]]


def get_initial_stoichiometries(
    initial_value, parameter_values, param=None, known_value="cyclable lithium capacity"
):
    """
    Calculate initial stoichiometries to start off the simulation at a particular
    state of charge, given voltage limits, open-circuit potentials, etc defined by
    parameter_values

    Parameters
    ----------
    initial_value : float
        Target initial value.
        If integer, interpreted as SOC, must be between 0 and 1.
        If string e.g. "4 V", interpreted as voltage, must be between V_min and V_max.
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
    esoh_solver = ElectrodeSOHSolver(parameter_values, param, known_value)
    return esoh_solver.get_initial_stoichiometries(initial_value)


def get_min_max_stoichiometries(
    parameter_values, param=None, known_value="cyclable lithium capacity"
):
    """
    Calculate min/max stoichiometries
    given voltage limits, open-circuit potentials, etc defined by parameter_values

    Parameters
    ----------
    parameter_values : :class:`pybamm.ParameterValues`
        The parameter values class that will be used for the simulation. Required for
        calculating appropriate initial stoichiometries.
    param : :class:`pybamm.LithiumIonParameters`, optional
        The symbolic parameter set to use for the simulation.
        If not provided, the default parameter set will be used.

    Returns
    -------
    x_0, x_100, y_100, y_0
        The min/max stoichiometries
    """
    esoh_solver = ElectrodeSOHSolver(parameter_values, param, known_value)
    return esoh_solver.get_min_max_stoichiometries()


def calculate_theoretical_energy(
    parameter_values, initial_soc=1.0, final_soc=0.0, points=100
):
    """
    Calculate maximum energy possible from a cell given OCV, initial soc, and final soc
    given voltage limits, open-circuit potentials, etc defined by parameter_values

    Parameters
    ----------
    parameter_values : :class:`pybamm.ParameterValues`
        The parameter values class that will be used for the simulation.
    initial_soc : float
        The soc at begining of discharge, default 1.0
    final_soc : float
        The soc at end of discharge, default 1.0
    points : int
        The number of points at which to calculate voltage.

    Returns
    -------
    E
        The total energy of the cell in Wh
    """
    # Get initial and final stoichiometric values.
    n_i, p_i = get_initial_stoichiometries(initial_soc, parameter_values)
    n_f, p_f = get_initial_stoichiometries(final_soc, parameter_values)
    n_vals = np.linspace(n_i, n_f, num=points)
    p_vals = np.linspace(p_i, p_f, num=points)
    # Calculate OCV at each stoichiometry
    param = pybamm.LithiumIonParameters()
    T = param.T_amb(0)
    Vs = np.empty(n_vals.shape)
    for i in range(n_vals.size):
        Vs[i] = parameter_values.evaluate(
            param.p.prim.U(p_vals[i], T)
        ) - parameter_values.evaluate(param.n.prim.U(n_vals[i], T))
    # Calculate dQ
    Q_p = parameter_values.evaluate(param.p.prim.Q_init) * (p_f - p_i)
    dQ = Q_p / (points - 1)
    # Integrate and convert to W-h
    E = np.trapz(Vs, dx=dQ)
    return E
