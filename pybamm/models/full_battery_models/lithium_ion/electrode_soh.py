#
# A model to calculate electrode-specific SOH
#
import pybamm
import numpy as np


class ElectrodeSOH(pybamm.BaseModel):
    """Model to calculate electrode-specific SOH, from [1]_.
    This model is mainly for internal use, to calculate summary variables in a
    simulation.

    .. math::
        n_{Li} = \\frac{3600}{F}(y_{100}C_p + x_{100}C_n),
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

    def __init__(self, name="ElectrodeSOH model", param=None):
        pybamm.citations.register("Mohtat2019")
        super().__init__(name)

        param = param or pybamm.LithiumIonParameters()

        Un = param.n.prim.U_dimensional
        Up = param.p.prim.U_dimensional
        T_ref = param.T_ref

        n_Li = pybamm.InputParameter("n_Li")
        V_max = pybamm.InputParameter("V_max")
        V_min = pybamm.InputParameter("V_min")
        Cn = pybamm.InputParameter("C_n")
        Cp = pybamm.InputParameter("C_p")

        x_100 = pybamm.Variable("x_100")
        y_100 = (n_Li * param.F / 3600 - x_100 * Cn) / Cp

        x_0 = pybamm.Variable("x_0")
        C = Cn * (x_100 - x_0)
        y_0 = y_100 + C / Cp

        Un_0 = Un(x_0, T_ref)
        Up_0 = Up(y_0, T_ref)
        Un_100 = Un(x_100, T_ref)
        Up_100 = Up(y_100, T_ref)

        self.algebraic = {x_100: Up_100 - Un_100 - V_max, x_0: Up_0 - Un_0 - V_min}

        self.initial_conditions = {x_0: pybamm.Scalar(0.1), x_100: pybamm.Scalar(0.9)}

        self.variables = {
            "C": C,
            "Capacity [A.h]": C,
            "x_100": x_100,
            "y_100": y_100,
            "x_0": x_0,
            "y_0": y_0,
            "Un(x_100)": Un_100,
            "Up(y_100)": Up_100,
            "Un(x_0)": Un_0,
            "Up(y_0)": Up_0,
            "Up(y_0) - Un(x_0)": Up_0 - Un_0,
            "Up(y_100) - Un(x_100)": Up_100 - Un_100,
            "n_Li_100": 3600 / param.F * (y_100 * Cp + x_100 * Cn),
            "n_Li_0": 3600 / param.F * (y_0 * Cp + x_0 * Cn),
            "n_Li": n_Li,
            "C_n": Cn,
            "C_p": Cp,
            "C_n * (x_100 - x_0)": Cn * (x_100 - x_0),
            "C_p * (y_100 - y_0)": Cp * (y_0 - y_100),
        }

    @property
    def default_solver(self):
        # Use AlgebraicSolver as CasadiAlgebraicSolver gives unnecessary warnings
        return pybamm.AlgebraicSolver()


class ElectrodeSOHx100(pybamm.BaseModel):
    """Model to calculate electrode-specific SOH for x_100 and y_100, from [1]_.
    This model is mainly for internal use, to calculate summary variables in a
    simulation.

    .. math::
        n_{Li} = \\frac{3600}{F}(y_{100}C_p + x_{100}C_n),
    .. math::
        V_{max} = U_p(y_{100}) - U_n(x_{100}),

    **Extends:** :class:`pybamm.BaseModel`
    """

    def __init__(self, name="ElectrodeSOHx100 model", param=None):
        pybamm.citations.register("Mohtat2019")
        super().__init__(name)

        param = param or pybamm.LithiumIonParameters()

        Un = param.n.prim.U_dimensional
        Up = param.p.prim.U_dimensional
        T_ref = param.T_ref

        n_Li = pybamm.InputParameter("n_Li")
        V_max = pybamm.InputParameter("V_max")
        Cn = pybamm.InputParameter("C_n")
        Cp = pybamm.InputParameter("C_p")

        x_100 = pybamm.Variable("x_100")
        y_100 = (n_Li * param.F / 3600 - x_100 * Cn) / Cp

        Un_100 = Un(x_100, T_ref)
        Up_100 = Up(y_100, T_ref)

        self.algebraic = {x_100: Up_100 - Un_100 - V_max}

        self.initial_conditions = {x_100: pybamm.Scalar(0.9)}

        self.variables = {"x_100": x_100, "y_100": y_100}

    @property
    def default_solver(self):
        # Use AlgebraicSolver as CasadiAlgebraicSolver gives unnecessary warnings
        return pybamm.AlgebraicSolver()


class ElectrodeSOHx0(pybamm.BaseModel):
    """Model to calculate electrode-specific SOH for x_0 and y_0, from [1]_.
    This model is mainly for internal use, to calculate summary variables in a
    simulation.

    .. math::
        V_{min} = U_p(y_{0}) - U_n(x_{0}),
    .. math::
        x_0 = x_{100} - \\frac{C}{C_n},
    .. math::
        y_0 = y_{100} + \\frac{C}{C_p}.

    **Extends:** :class:`pybamm.BaseModel`
    """

    def __init__(self, name="ElectrodeSOHx0 model", param=None):
        pybamm.citations.register("Mohtat2019")
        super().__init__(name)

        param = param or pybamm.LithiumIonParameters()

        Un = param.n.prim.U_dimensional
        Up = param.p.prim.U_dimensional
        T_ref = param.T_ref

        n_Li = pybamm.InputParameter("n_Li")
        V_min = pybamm.InputParameter("V_min")
        Cn = pybamm.InputParameter("C_n")
        Cp = pybamm.InputParameter("C_p")
        x_100 = pybamm.InputParameter("x_100")
        y_100 = pybamm.InputParameter("y_100")

        x_0 = pybamm.Variable("x_0")
        C = Cn * (x_100 - x_0)
        y_0 = y_100 + C / Cp

        Un_0 = Un(x_0, T_ref)
        Up_0 = Up(y_0, T_ref)
        Un_100 = Un(x_100, T_ref)
        Up_100 = Up(y_100, T_ref)

        self.algebraic = {x_0: Up_0 - Un_0 - V_min}

        self.initial_conditions = {x_0: pybamm.Scalar(0.1)}

        self.variables = {
            "C": C,
            "Capacity [A.h]": C,
            "x_0": x_0,
            "y_0": y_0,
            "Un(x_100)": Un_100,
            "Up(y_100)": Up_100,
            "Un(x_0)": Un_0,
            "Up(y_0)": Up_0,
            "Up(y_0) - Un(x_0)": Up_0 - Un_0,
            "Up(y_100) - Un(x_100)": Up_100 - Un_100,
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


class ElectrodeSOHSolver:
    def __init__(self, parameter_values, param=None):
        self.parameter_values = parameter_values
        self.param = param or pybamm.LithiumIonParameters()

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

        # Parameterize the OCP functions
        T = parameter_values["Reference temperature [K]"]
        x = pybamm.InputParameter("x")
        y = pybamm.InputParameter("y")
        self.OCV_function = parameter_values.process_symbol(
            self.param.p.prim.U_dimensional(y, T)
            - self.param.n.prim.U_dimensional(x, T)
        )

    def _get_electrode_soh_sims_full(self):
        try:
            return self._full_sim
        except AttributeError:
            full_model = ElectrodeSOH(param=self.param)
            self._full_sim = pybamm.Simulation(
                full_model, parameter_values=self.parameter_values
            )
            return self._full_sim

    def _get_electrode_soh_sims_split(self):
        try:
            return self._split_sims
        except AttributeError:
            x100_model = ElectrodeSOHx100(param=self.param)
            x100_sim = pybamm.Simulation(
                x100_model, parameter_values=self.parameter_values
            )
            x0_model = ElectrodeSOHx0(param=self.param)
            x0_sim = pybamm.Simulation(x0_model, parameter_values=self.parameter_values)
            self._split_sims = [x100_sim, x0_sim]
            return self._split_sims

    def solve(self, inputs):
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
        x0_min, x100_max, _, _ = self._get_lims(inputs)

        x100_init = x100_max
        x0_init = x0_min
        if sim.solution is not None:
            # Update the initial conditions if they are valid
            x100_init_sol = sim.solution["x_100"].data[0]
            if x0_min < x100_init_sol < x100_max:
                x100_init = x100_init_sol
            x0_init_sol = sim.solution["x_0"].data[0]
            if x0_min < x0_init_sol < x100_max:
                x0_init = x0_init_sol
        return {"x_100": np.array(x100_init), "x_0": np.array(x0_init)}

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
        Get stoichiometry limits based on n_Li, C_n, and C_p
        """
        Cp = inputs["C_p"]
        Cn = inputs["C_n"]
        n_Li = inputs["n_Li"]

        x0_min, x100_max, y100_min, y0_max = self.lims_ocp

        # Update (tighten) stoich limits based on total lithium content and electrode
        # capacities
        F = pybamm.constants.F.value
        x100_max_from_y100_min = (n_Li * F / 3600 - y100_min * Cp) / Cn
        x0_min_from_y0_max = (n_Li * F / 3600 - y0_max * Cp) / Cn
        y100_min_from_x100_max = (n_Li * F / 3600 - x100_max * Cn) / Cp
        y0_max_from_x0_min = (n_Li * F / 3600 - x0_min * Cn) / Cp

        x100_max = min(x100_max_from_y100_min, x100_max)
        x0_min = max(x0_min_from_y0_max, x0_min)
        y100_min = max(y100_min_from_x100_max, y100_min)
        y0_max = min(y0_max_from_x0_min, y0_max)
        return (x0_min, x100_max, y100_min, y0_max)

    def _check_esoh_feasible(self, inputs):
        """
        Check that the electrode SOH calculation is feasible, based on voltage limits
        """
        x0_min, x100_max, y100_min, y0_max = self._get_lims(inputs)
        Vmax = inputs["V_max"]
        Vmin = inputs["V_min"]

        # Check stoich limits are between 0 and 1
        for x in ["x0_min", "x100_max", "y100_min", "y0_max"]:
            xval = eval(x)
            if not 0 < xval < 1:  # pragma: no cover
                raise ValueError(f"'{x}' should be between 0 and 1, but is {xval:.4f}")

        # Check that the min and max achievable voltages span wider than the desired
        # voltage range
        V_lower_bound = self.OCV_function.evaluate(inputs={"x": x0_min, "y": y0_max})
        V_upper_bound = self.OCV_function.evaluate(
            inputs={"x": x100_max, "y": y100_min}
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

    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values, param)

    inputs = {"V_min": V_min, "V_max": V_max, "C_n": C_n, "C_p": C_p, "n_Li": n_Li}

    # Solve the model and check outputs
    sol = esoh_solver.solve(inputs)

    x_0 = sol["x_0"].data[0]
    y_0 = sol["y_0"].data[0]
    C = sol["C"].data[0]
    x = x_0 + initial_soc * C / C_n
    y = y_0 - initial_soc * C / C_p

    return x, y
