#
# A model to calculate electrode-specific SOH
#
import pybamm
import numpy as np
from functools import lru_cache
import warnings


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


class BaseElectrodeSOHSolver:
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
        self.parameter_values = parameter_values
        self.param = param or pybamm.LithiumIonParameters()
        self.known_value = known_value

        self.lims_ocp = self._get_lims_ocp()
        self.OCV_function = None
        self._get_electrode_soh_sims_full = lru_cache()(
            self.__get_electrode_soh_sims_full
        )
        self._get_electrode_soh_sims_split = lru_cache()(
            self.__get_electrode_soh_sims_split
        )

    def _get_lims_ocp(self):
        raise NotImplementedError

    def __get_electrode_soh_sims_full(self):
        raise NotImplementedError

    def __get_electrode_soh_sims_split(self):
        raise NotImplementedError

    def solve(self, inputs):
        if "n_Li" in inputs:
            warnings.warn(
                "Input 'n_Li' has been replaced by 'Q_Li', which is 'n_Li * F / 3600'. "
                "This will be automatically calculated for now. "
                "Q_Li can be read from parameters as 'param.Q_Li_particles_init'",
                DeprecationWarning,
            )
            n_Li = inputs.pop("n_Li")
            inputs["Q_Li"] = n_Li * pybamm.constants.F.value / 3600
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

        # Calculate theoretical energy
        # x_0 = sol_dict["x_0"]
        # y_0 = sol_dict["y_0"]
        # x_100 = sol_dict["x_100"]
        # y_100 = sol_dict["y_100"]
        # energy = pybamm.lithium_ion.electrode_soh.theoretical_energy_integral(
        #    self.parameter_values, x_100, x_0, y_100, y_0
        # )
        # sol_dict.update({"Maximum theoretical energy [W.h]": energy})
        return sol_dict

    def _set_up_solve(self, inputs):
        raise NotImplementedError

    def _solve_full(self, inputs, ics):
        sim = self._get_electrode_soh_sims_full()
        sim.build()
        sim.built_model.set_initial_conditions_from(ics)
        sol = sim.solve([0], inputs=inputs)
        return sol

    def _solve_split(self, inputs, ics):
        raise NotImplementedError

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
        raise NotImplementedError

    def get_initial_stoichiometries(self, initial_value):
        raise NotImplementedError

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
    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
        parameter_values, param, known_value
    )
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
    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
        parameter_values, param, known_value
    )
    return esoh_solver.get_min_max_stoichiometries()


def theoretical_energy_integral(parameter_values, n_i, n_f, p_i, p_f, points=100):
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
        The soc at end of discharge, default 0.0
    points : int
        The number of points at which to calculate voltage.

    Returns
    -------
    E
        The total energy of the cell in Wh
    """
    # Get initial and final stoichiometric values.
    x_100, y_100 = get_initial_stoichiometries(initial_soc, parameter_values)
    x_0, y_0 = get_initial_stoichiometries(final_soc, parameter_values)
    E = theoretical_energy_integral(
        parameter_values, x_100, x_0, y_100, y_0, points=points
    )
    return E
