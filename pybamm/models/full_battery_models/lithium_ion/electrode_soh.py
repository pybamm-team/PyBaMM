#
# A model to calculate electrode-specific SOH
#
import pybamm
import numpy as np
from functools import lru_cache


class _BaseElectrodeSOH(pybamm.BaseModel):
    def __init__(self):
        pybamm.citations.register("Mohtat2019")
        pybamm.citations.register("Weng2023")
        name = "ElectrodeSOH model"
        super().__init__(name)

    def get_100_soc_variables(
        self, x_100, y_100, Un_100, Up_100, Q_Li, Q_n, Q_p, param
    ):
        Acc_cm2 = param.A_cc * 1e4
        variables = {
            "x_100": x_100,
            "y_100": y_100,
            "Un(x_100)": Un_100,
            "Up(y_100)": Up_100,
            "Up(y_100) - Un(x_100)": Up_100 - Un_100,
            "Q_Li": Q_Li,
            "n_Li": Q_Li * 3600 / param.F,
            "Q_n": Q_n,
            "Q_p": Q_p,
            "Cyclable lithium capacity [A.h]": Q_Li,
            "Negative electrode capacity [A.h]": Q_n,
            "Positive electrode capacity [A.h]": Q_p,
            "Cyclable lithium capacity [mA.h.cm-2]": Q_Li * 1e3 / Acc_cm2,
            "Negative electrode capacity [mA.h.cm-2]": Q_n * 1e3 / Acc_cm2,
            "Positive electrode capacity [mA.h.cm-2]": Q_p * 1e3 / Acc_cm2,
            # eq 33 of Weng2023
            "Formation capacity loss [A.h]": Q_p - Q_Li,
            "Formation capacity loss [mA.h.cm-2]": (Q_p - Q_Li) * 1e3 / Acc_cm2,
            # eq 26 of Weng2024
            "Negative positive ratio": Q_n / Q_p,
            "NPR": Q_n / Q_p,
        }
        return variables

    def get_0_soc_variables(
        self, x_0, y_0, x_100, y_100, Un_0, Up_0, Q, Q_n, Q_p, param
    ):
        Acc_cm2 = param.A_cc * 1e4
        # eq 27 of Weng2023
        Q_n_excess = Q_n * (1 - x_100)
        NPR_practical = 1 + Q_n_excess / Q
        variables = {
            "Q": Q,
            "Capacity [A.h]": Q,
            "Capacity [mA.h.cm-2]": Q * 1e3 / Acc_cm2,
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
            "Practical negative positive ratio": NPR_practical,
            "Practical NPR": NPR_practical,
        }
        return variables

    @property
    def default_solver(self):
        # Use AlgebraicSolver as CasadiAlgebraicSolver gives unnecessary warnings
        return pybamm.AlgebraicSolver()


class _ElectrodeSOH(_BaseElectrodeSOH):
    """
    Model to calculate electrode-specific SOH, from :footcite:t:`Mohtat2019`. This
    model is mainly for internal use, to calculate summary variables in a simulation.
    Some of the output variables are defined in :footcite:t:`Weng2023`.

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

    """

    def __init__(
        self, param=None, solve_for=None, known_value="cyclable lithium capacity"
    ):
        super().__init__()

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

        V_max = param.ocp_soc_100
        V_min = param.ocp_soc_0
        Q_n = pybamm.InputParameter("Q_n")
        Q_p = pybamm.InputParameter("Q_p")

        if known_value == "cyclable lithium capacity":
            Q_Li = pybamm.InputParameter("Q_Li")
        elif known_value == "cell capacity":
            Q = pybamm.InputParameter("Q")
        else:
            raise ValueError(
                "Known value must be cell capacity or cyclable lithium capacity"
            )

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
        self.variables = self.get_100_soc_variables(
            x_100, y_100, Un_100, Up_100, Q_Li, Q_n, Q_p, param
        )

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
                self.get_0_soc_variables(
                    x_0, y_0, x_100, y_100, Un_0, Up_0, Q, Q_n, Q_p, param
                )
            )


class _ElectrodeSOHMSMR(_BaseElectrodeSOH):
    """
    Model to calculate electrode-specific SOH using the MSMR formulation from
    :footcite:t:`Baker2018`. See :class:`_ElectrodeSOH` for more details.
    """

    def __init__(
        self, param=None, solve_for=None, known_value="cyclable lithium capacity"
    ):
        pybamm.citations.register("Baker2018")
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

        T = param.T_ref
        V_max = param.voltage_high_cut
        V_min = param.voltage_low_cut
        Q_n = pybamm.InputParameter("Q_n")
        Q_p = pybamm.InputParameter("Q_p")

        if known_value == "cyclable lithium capacity":
            Q_Li = pybamm.InputParameter("Q_Li")
        elif known_value == "cell capacity":
            Q = pybamm.InputParameter("Q")
        else:
            raise ValueError(
                "Known value must be cell capacity or cyclable lithium capacity"
            )

        # Define variables for 0% state of charge
        # TODO: thermal effects (include dU/dT)
        if "Un_0" in solve_for:
            Un_0 = pybamm.Variable("Un(x_0)")
            Up_0 = V_min + Un_0
            x_0 = x_n(Un_0, T)
            y_0 = x_p(Up_0, T)

        # Define variables for 100% state of charge
        # TODO: thermal effects (include dU/dT)
        if "Un_100" in solve_for:
            Un_100 = pybamm.Variable("Un(x_100)")
            Up_100 = V_max + Un_100
            x_100 = x_n(Un_100, T)
            y_100 = x_p(Up_100, T)
        else:
            Un_100 = pybamm.InputParameter("Un(x_100)")
            Up_100 = pybamm.InputParameter("Up(y_100)")
            x_100 = x_n(Un_100, T)
            y_100 = x_p(Up_100, T)

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
    options : dict-like, optional
        A dictionary of options to be passed to the model, see
        :class:`pybamm.BatteryModelOptions`.
    """

    def __init__(
        self,
        parameter_values,
        param=None,
        known_value="cyclable lithium capacity",
        options=None,
    ):
        self.parameter_values = parameter_values
        self.param = param or pybamm.LithiumIonParameters(options)
        if known_value not in ["cell capacity", "cyclable lithium capacity"]:
            raise ValueError(
                "Known value must be cell capacity or cyclable lithium capacity"
            )
        self.known_value = known_value
        self.options = options or pybamm.BatteryModelOptions({})

        self.lims_ocp = self._get_lims_ocp()
        self.OCV_function = None
        self._get_electrode_soh_sims_full = lru_cache()(
            self.__get_electrode_soh_sims_full
        )
        self._get_electrode_soh_sims_split = lru_cache()(
            self.__get_electrode_soh_sims_split
        )

    def _get_lims_ocp(self):
        parameter_values = self.parameter_values

        # Check whether each electrode OCP is a function (False) or data (True)
        # Set to false for MSMR models
        if self.options["open-circuit potential"] == "MSMR":
            OCPp_data = False
            OCPn_data = False
        else:
            OCPp_data = isinstance(
                parameter_values["Positive electrode OCP [V]"], tuple
            )
            OCPn_data = isinstance(
                parameter_values["Negative electrode OCP [V]"], tuple
            )

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
        return (x0_min, x100_max, y100_min, y0_max)

    def __get_electrode_soh_sims_full(self):
        if self.options["open-circuit potential"] == "MSMR":
            full_model = _ElectrodeSOHMSMR(
                param=self.param, known_value=self.known_value
            )
        else:
            full_model = _ElectrodeSOH(param=self.param, known_value=self.known_value)
        return pybamm.Simulation(full_model, parameter_values=self.parameter_values)

    def __get_electrode_soh_sims_split(self):
        if self.options["open-circuit potential"] == "MSMR":
            x100_model = _ElectrodeSOHMSMR(
                param=self.param, solve_for=["Un_100"], known_value=self.known_value
            )
            x0_model = _ElectrodeSOHMSMR(
                param=self.param, solve_for=["Un_0"], known_value=self.known_value
            )
        else:
            x100_model = _ElectrodeSOH(
                param=self.param, solve_for=["x_100"], known_value=self.known_value
            )
            x0_model = _ElectrodeSOH(
                param=self.param, solve_for=["x_0"], known_value=self.known_value
            )
        x100_sim = pybamm.Simulation(x100_model, parameter_values=self.parameter_values)
        x0_sim = pybamm.Simulation(x0_model, parameter_values=self.parameter_values)
        return [x100_sim, x0_sim]

    def solve(self, inputs):
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

        # Calculate theoretical energy
        # TODO: energy calc for MSMR
        if self.options["open-circuit potential"] != "MSMR":
            energy_inputs = {**sol_dict, **inputs}
            energy = self.theoretical_energy_integral(energy_inputs)
            sol_dict.update({"Maximum theoretical energy [W.h]": energy})
        return sol_dict

    def _set_up_solve(self, inputs):
        # Try with full sim
        sim = self._get_electrode_soh_sims_full()
        if sim.solution is not None:
            if self.options["open-circuit potential"] == "MSMR":
                Un_100_sol = sim.solution["Un(x_100)"].data
                Un_0_sol = sim.solution["Un(x_0)"].data
                Up_100_sol = sim.solution["Up(y_100)"].data
                Up_0_sol = sim.solution["Up(y_0)"].data
                return {
                    "Un(x_100)": Un_100_sol,
                    "Un(x_0)": Un_0_sol,
                    "Up(x_100)": Up_100_sol,
                    "Up(x_0)": Up_0_sol,
                }
            else:
                x100_sol = sim.solution["x_100"].data
                x0_sol = sim.solution["x_0"].data
                y100_sol = sim.solution["y_100"].data
                y0_sol = sim.solution["y_0"].data
                return {
                    "x_100": x100_sol,
                    "x_0": x0_sol,
                    "y_100": y100_sol,
                    "y_0": y0_sol,
                }

        # Try with split sims
        if self.known_value == "cyclable lithium capacity":
            x100_sim, x0_sim = self._get_electrode_soh_sims_split()
            if x100_sim.solution is not None and x0_sim.solution is not None:
                if self.options["open-circuit potential"] == "MSMR":
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
                else:
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
        if self.options["open-circuit potential"] == "MSMR":
            msmr_pot_model = _get_msmr_potential_model(
                self.parameter_values, self.param
            )
            sol0 = pybamm.AlgebraicSolver().solve(
                msmr_pot_model, inputs={"x": x0_init, "y": y0_init}
            )
            sol100 = pybamm.AlgebraicSolver().solve(
                msmr_pot_model, inputs={"x": x100_init, "y": y100_init}
            )
            return {
                "Un(x_100)": sol100["Un"].data,
                "Un(x_0)": sol0["Un"].data,
                "Up(y_100)": sol100["Up"].data,
                "Up(y_0)": sol0["Up"].data,
            }
        else:
            return {
                "x_100": x100_init,
                "x_0": x0_init,
                "y_100": y100_init,
                "y_0": y0_init,
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
        if self.options["open-circuit potential"] == "MSMR":
            inputs["Un(x_100)"] = x100_sol["Un(x_100)"].data[0]
            inputs["Up(y_100)"] = x100_sol["Up(y_100)"].data[0]
        else:
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
            self.V_max = self.parameter_values.evaluate(self.param.ocp_soc_100)
            self.V_min = self.parameter_values.evaluate(self.param.ocp_soc_0)
            if self.options["open-circuit potential"] == "MSMR":
                # will solve for potentials at the sto limits, so no need
                # to store a function
                self.OCV_function = "MSMR"
            else:
                T = self.parameter_values["Reference temperature [K]"]
                x = pybamm.InputParameter("x")
                y = pybamm.InputParameter("y")
                self.OCV_function = self.parameter_values.process_symbol(
                    self.param.p.prim.U(y, T) - self.param.n.prim.U(x, T)
                )

        # Evaluate OCP function
        if self.options["open-circuit potential"] == "MSMR":
            msmr_pot_model = _get_msmr_potential_model(
                self.parameter_values, self.param
            )
            sol0 = pybamm.AlgebraicSolver(tol=1e-4).solve(
                msmr_pot_model, inputs={"x": x0_min, "y": y0_max}
            )
            sol100 = pybamm.AlgebraicSolver(tol=1e-4).solve(
                msmr_pot_model, inputs={"x": x100_max, "y": y100_min}
            )
            Up0 = sol0["Up"].data[0]
            Un0 = sol0["Un"].data[0]
            Up100 = sol100["Up"].data[0]
            Un100 = sol100["Un"].data[0]
            V_lower_bound = float(Up0 - Un0)
            V_upper_bound = float(Up100 - Un100)
        else:
            # address numpy 1.25 deprecation warning: array should have ndim=0
            # before conversion
            all_inputs = {**inputs, "x": x0_min, "y": y0_max}
            V_lower_bound = float(self.OCV_function.evaluate(inputs=all_inputs).item())
            all_inputs.update({"x": x100_max, "y": y100_min})
            V_upper_bound = float(self.OCV_function.evaluate(inputs=all_inputs).item())

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

    def get_initial_stoichiometries(self, initial_value, tol=1e-6, inputs=None):
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
        tol : float, optional
            The tolerance for the solver used to compute the initial stoichiometries.
            A lower value results in higher precision but may increase computation time.
            Default is 1e-6.

        Returns
        -------
        x, y
            The initial stoichiometries that give the desired initial state of charge
        """
        parameter_values = self.parameter_values
        param = self.param
        x_0, x_100, y_100, y_0 = self.get_min_max_stoichiometries(inputs=inputs)

        if isinstance(initial_value, str) and initial_value.endswith("V"):
            V_init = float(initial_value[:-1])
            V_min = parameter_values.evaluate(param.ocp_soc_0)
            V_max = parameter_values.evaluate(param.ocp_soc_100)

            if not V_min <= V_init <= V_max:
                raise ValueError(
                    f"Initial voltage {V_init}V is outside the voltage limits "
                    f"({V_min}, {V_max})"
                )

            # Solve simple model for initial soc based on target voltage
            soc_model = pybamm.BaseModel()
            soc = pybamm.Variable("soc")
            x = x_0 + soc * (x_100 - x_0)
            y = y_0 - soc * (y_0 - y_100)
            T_ref = parameter_values["Reference temperature [K]"]
            if self.options["open-circuit potential"] == "MSMR":
                xn = param.n.prim.x
                xp = param.p.prim.x
                Up = pybamm.Variable("Up")
                Un = pybamm.Variable("Un")
                soc_model.algebraic[Up] = x - xn(Un, T_ref)
                soc_model.algebraic[Un] = y - xp(Up, T_ref)
                soc_model.initial_conditions[Un] = 0
                soc_model.initial_conditions[Up] = V_max
                soc_model.algebraic[soc] = Up - Un - V_init
            else:
                Up = param.p.prim.U
                Un = param.n.prim.U
                soc_model.algebraic[soc] = Up(y, T_ref) - Un(x, T_ref) - V_init
            # initial guess for soc linearly interpolates between 0 and 1
            # based on V linearly interpolating between V_max and V_min
            soc_model.initial_conditions[soc] = (V_init - V_min) / (V_max - V_min)
            soc_model.variables["soc"] = soc
            parameter_values.process_model(soc_model)
            initial_soc = (
                pybamm.AlgebraicSolver(tol=tol).solve(soc_model, [0])["soc"].data[0]
            )
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

    def get_min_max_stoichiometries(self, inputs=None):
        """
        Calculate min/max stoichiometries
        given voltage limits, open-circuit potentials, etc defined by parameter_values

        Returns
        -------
        x_0, x_100, y_100, y_0
            The min/max stoichiometries
        """
        inputs = inputs or {}
        parameter_values = self.parameter_values
        param = self.param

        Q_n = parameter_values.evaluate(param.n.Q_init, inputs=inputs)
        Q_p = parameter_values.evaluate(param.p.Q_init, inputs=inputs)

        if self.known_value == "cyclable lithium capacity":
            Q_Li = parameter_values.evaluate(param.Q_Li_particles_init, inputs=inputs)
            all_inputs = {**inputs, "Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}
        elif self.known_value == "cell capacity":
            Q = parameter_values.evaluate(
                param.Q / param.n_electrodes_parallel, inputs=inputs
            )
            all_inputs = {**inputs, "Q_n": Q_n, "Q_p": Q_p, "Q": Q}
        # Solve the model and check outputs
        sol = self.solve(all_inputs)
        return [sol["x_0"], sol["x_100"], sol["y_100"], sol["y_0"]]

    def get_initial_ocps(self, initial_value, tol=1e-6):
        """
        Calculate initial open-circuit potentials to start off the simulation at a
        particular state of charge, given voltage limits, open-circuit potentials, etc
        defined by parameter_values

        Parameters
        ----------
        initial_value : float
            Target SOC, must be between 0 and 1.
        tol: float, optional
            Tolerance for the solver used in calculating initial stoichiometries.

        Returns
        -------
        Un, Up
            The initial open-circuit potentials at the desired initial state of charge
        """
        parameter_values = self.parameter_values
        param = self.param
        x, y = self.get_initial_stoichiometries(initial_value, tol)
        if self.options["open-circuit potential"] == "MSMR":
            msmr_pot_model = _get_msmr_potential_model(
                self.parameter_values, self.param
            )
            sol = pybamm.AlgebraicSolver().solve(
                msmr_pot_model, inputs={"x": x, "y": y}
            )
            Un = sol["Un"].data[0]
            Up = sol["Up"].data[0]
        else:
            T_ref = parameter_values["Reference temperature [K]"]
            Un = parameter_values.evaluate(param.n.prim.U(x, T_ref))
            Up = parameter_values.evaluate(param.p.prim.U(y, T_ref))
        return Un, Up

    def get_min_max_ocps(self):
        """
        Calculate min/max open-circuit potentials
        given voltage limits, open-circuit potentials, etc defined by parameter_values

        Returns
        -------
        Un_0, Un_100, Up_100, Up_0
            The min/max ocps
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
        return [sol["Un(x_0)"], sol["Un(x_100)"], sol["Up(y_100)"], sol["Up(y_0)"]]

    def theoretical_energy_integral(self, inputs, points=1000):
        x_0 = inputs["x_0"]
        y_0 = inputs["y_0"]
        x_100 = inputs["x_100"]
        y_100 = inputs["y_100"]
        Q_p = inputs["Q_p"]
        x_vals = np.linspace(x_100, x_0, num=points)
        y_vals = np.linspace(y_100, y_0, num=points)
        # Calculate OCV at each stoichiometry
        param = self.param
        T = param.T_amb_av(0)
        Vs = self.parameter_values.evaluate(
            param.p.prim.U(y_vals, T) - param.n.prim.U(x_vals, T), inputs=inputs
        ).flatten()
        # Calculate dQ
        Q = Q_p * (y_0 - y_100)
        dQ = Q / (points - 1)
        # Integrate and convert to W-h
        E = np.trapz(Vs, dx=dQ)
        return E


def get_initial_stoichiometries(
    initial_value,
    parameter_values,
    param=None,
    known_value="cyclable lithium capacity",
    options=None,
    tol=1e-6,
    inputs=None,
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
    known_value : str, optional
        The known value needed to complete the electrode SOH model.
        Can be "cyclable lithium capacity" (default) or "cell capacity".
    options : dict-like, optional
        A dictionary of options to be passed to the model, see
        :class:`pybamm.BatteryModelOptions`.
    tol : float, optional
        The tolerance for the solver used to compute the initial stoichiometries.
        A lower value results in higher precision but may increase computation time.
        Default is 1e-6.

    Returns
    -------
    x, y
        The initial stoichiometries that give the desired initial state of charge
    """
    esoh_solver = ElectrodeSOHSolver(parameter_values, param, known_value, options)
    return esoh_solver.get_initial_stoichiometries(initial_value, tol, inputs=inputs)


def get_min_max_stoichiometries(
    parameter_values, param=None, known_value="cyclable lithium capacity", options=None
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
    known_value : str, optional
        The known value needed to complete the electrode SOH model.
        Can be "cyclable lithium capacity" (default) or "cell capacity".
    options : dict-like, optional
        A dictionary of options to be passed to the model, see
        :class:`pybamm.BatteryModelOptions`.

    Returns
    -------
    x_0, x_100, y_100, y_0
        The min/max stoichiometries
    """
    esoh_solver = ElectrodeSOHSolver(parameter_values, param, known_value, options)
    return esoh_solver.get_min_max_stoichiometries()


def get_initial_ocps(
    initial_value,
    parameter_values,
    param=None,
    known_value="cyclable lithium capacity",
    options=None,
):
    """
    Calculate initial open-circuit potentials to start off the simulation at a
    particular state of charge, given voltage limits, open-circuit potentials, etc
    defined by parameter_values

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
    known_value : str, optional
        The known value needed to complete the electrode SOH model.
        Can be "cyclable lithium capacity" (default) or "cell capacity".
    options : dict-like, optional
        A dictionary of options to be passed to the model, see
        :class:`pybamm.BatteryModelOptions`.

    Returns
    -------
    Un, Up
        The initial electrode OCPs that give the desired initial state of charge
    """
    esoh_solver = ElectrodeSOHSolver(parameter_values, param, known_value, options)
    return esoh_solver.get_initial_ocps(initial_value)


def get_min_max_ocps(
    parameter_values, param=None, known_value="cyclable lithium capacity", options=None
):
    """
    Calculate min/max open-circuit potentials
    given voltage limits, open-circuit potentials, etc defined by parameter_values

    Parameters
    ----------
    parameter_values : :class:`pybamm.ParameterValues`
        The parameter values class that will be used for the simulation. Required for
        calculating appropriate initial open-circuit potentials.
    param : :class:`pybamm.LithiumIonParameters`, optional
        The symbolic parameter set to use for the simulation.
        If not provided, the default parameter set will be used.
    known_value : str, optional
        The known value needed to complete the electrode SOH model.
        Can be "cyclable lithium capacity" (default) or "cell capacity".
    options : dict-like, optional
        A dictionary of options to be passed to the model, see
        :class:`pybamm.BatteryModelOptions`.

    Returns
    -------
    Un_0, Un_100, Up_100, Up_0
        The min/max OCPs
    """
    esoh_solver = ElectrodeSOHSolver(parameter_values, param, known_value, options)
    return esoh_solver.get_min_max_ocps()


def theoretical_energy_integral(parameter_values, param, inputs, points=100):
    """
    Calculate maximum energy possible from a cell given OCV, initial soc, and final soc
    given voltage limits, open-circuit potentials, etc defined by parameter_values

    Parameters
    ----------
    parameter_values : :class:`pybamm.ParameterValues`
        The parameter values class that will be used for the simulation.
    n_i, n_f, p_i, p_f : float
        initial and final stoichiometries for the positive and negative
        electrodes, respectively
    points : int
        The number of points at which to calculate voltage.
    Returns
    -------
    E
        The total energy of the cell in Wh
    """
    esoh_solver = ElectrodeSOHSolver(parameter_values, param)
    return esoh_solver.theoretical_energy_integral(inputs, points=points)


def calculate_theoretical_energy(
    parameter_values, initial_soc=1.0, final_soc=0.0, points=100, tol=1e-6
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
        The soc at end of discharge, default 0.0
    points : int
        The number of points at which to calculate voltage.
    tol: float
        Tolerance for the solver used in calculating initial and final stoichiometries.
    Returns
    -------
    E
        The total energy of the cell in Wh
    """
    # Get initial and final stoichiometric values.
    x_100, y_100 = get_initial_stoichiometries(initial_soc, parameter_values, tol=tol)
    x_0, y_0 = get_initial_stoichiometries(final_soc, parameter_values, tol=tol)
    Q_p = parameter_values.evaluate(pybamm.LithiumIonParameters().p.prim.Q_init)
    E = theoretical_energy_integral(
        parameter_values,
        pybamm.LithiumIonParameters(),
        {"x_100": x_100, "x_0": x_0, "y_100": y_100, "y_0": y_0, "Q_p": Q_p},
        points=points,
    )
    return E


def _get_msmr_potential_model(parameter_values, param):
    """
    Returns a solver to calculate the open-circuit potentials of the individual
    electrodes at the given stoichiometries
    """
    V_max = param.voltage_high_cut
    V_min = param.voltage_low_cut
    x_n = param.n.prim.x
    x_p = param.p.prim.x
    T = param.T_ref
    model = pybamm.BaseModel()
    Un = pybamm.Variable("Un")
    Up = pybamm.Variable("Up")
    x = pybamm.InputParameter("x")
    y = pybamm.InputParameter("y")
    model.algebraic = {
        Un: x_n(Un, T) - x,
        Up: x_p(Up, T) - y,
    }
    model.initial_conditions = {
        Un: 1 - x,
        Up: V_max * (1 - y) + V_min * y,
    }
    model.variables = {
        "Un": Un,
        "Up": Up,
    }
    parameter_values.process_model(model)
    return model
