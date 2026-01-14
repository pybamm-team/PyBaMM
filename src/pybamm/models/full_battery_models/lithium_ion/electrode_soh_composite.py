#
# A model to calculate electrode-specific SOH, adapted for composite electrodes
#
import pybamm

from .electrode_soh import _ElectrodeSOH
from .util import _get_lithiation_delithiation, check_if_composite


def _get_primary_only_options(options):
    """
    Create options dict with only primary phase OCP settings.

    When composite options have tuple OCP settings like
    (("single", "hysteresis"), "single"), the non-composite model needs only the
    primary (first) element for each electrode to avoid incorrectly detecting
    hysteresis from the secondary phase.

    Example transformation:
        Input: {"open-circuit potential": (("single", "hysteresis"), "single")}
        Output: {"open-circuit potential": ("single", "single")}
    """
    if options is None:
        return None

    if not isinstance(options, pybamm.BatteryModelOptions):
        options = pybamm.BatteryModelOptions(options)

    # Create a copy of the options as a dict
    options_dict = dict(options)

    # Get the OCP option
    ocp_option = options_dict.get("open-circuit potential")
    if ocp_option is None:
        return options_dict

    # Handle the different cases for OCP option structure
    # Case 1: Simple string - same for all electrodes/phases
    if isinstance(ocp_option, str):
        return options_dict

    # Case 2: 2-tuple for (negative, positive) electrodes
    if isinstance(ocp_option, tuple) and len(ocp_option) == 2:
        neg_ocp, pos_ocp = ocp_option

        # Extract primary phase OCP for negative electrode
        if isinstance(neg_ocp, tuple):
            neg_ocp = neg_ocp[0]  # Primary phase

        # Extract primary phase OCP for positive electrode
        if isinstance(pos_ocp, tuple):
            pos_ocp = pos_ocp[0]  # Primary phase

        options_dict["open-circuit potential"] = (neg_ocp, pos_ocp)

    return options_dict


def _get_stoich_variables(options):
    variables = {
        "x_100_1": pybamm.Variable("x_100_1", bounds=(0, 1)),
        "y_100_1": pybamm.Variable("y_100_1", bounds=(0, 1)),
        "x_0_1": pybamm.Variable("x_0_1", bounds=(0, 1)),
        "y_0_1": pybamm.Variable("y_0_1", bounds=(0, 1)),
        "x_init_1": pybamm.Variable("x_init_1", bounds=(0, 1)),
        "y_init_1": pybamm.Variable("y_init_1", bounds=(0, 1)),
    }
    is_positive_composite = check_if_composite(options, "positive")
    is_negative_composite = check_if_composite(options, "negative")
    if is_positive_composite:
        variables["y_100_2"] = pybamm.Variable("y_100_2", bounds=(0, 1))
        variables["y_0_2"] = pybamm.Variable("y_0_2", bounds=(0, 1))
        variables["y_init_2"] = pybamm.Variable("y_init_2", bounds=(0, 1))
    if is_negative_composite:
        variables["x_100_2"] = pybamm.Variable("x_100_2", bounds=(0, 1))
        variables["x_0_2"] = pybamm.Variable("x_0_2", bounds=(0, 1))
        variables["x_init_2"] = pybamm.Variable("x_init_2", bounds=(0, 1))
    return variables


def _get_initial_conditions(options, soc_init):
    variables = _get_stoich_variables(options)
    ics = {}
    # Avoid exact boundary values for better numerical stability
    eps = 0.01
    for name, var in variables.items():
        if "100" in name and "x" in name:
            ics[var] = 0.85
        elif "0" in name and "x" in name:
            ics[var] = 0.15
        elif "100" in name and "y" in name:
            ics[var] = 0.15
        elif "0" in name and "y" in name:
            ics[var] = 0.85
        elif "init" in name and "x" in name:
            ics[var] = pybamm.maximum(eps, pybamm.minimum(1 - eps, soc_init))
        elif "init" in name and "y" in name:
            ics[var] = pybamm.maximum(eps, pybamm.minimum(1 - eps, 1 - soc_init))
    return ics


def _get_direction(electrode):
    if electrode == "positive":
        return pybamm.Scalar(-1)
    else:
        return pybamm.Scalar(1)


def _get_prefix(electrode):
    if electrode == "positive":
        return "y"
    else:
        return "x"


def _get_electrode_capacity_equation(options, electrode):
    prefix = _get_prefix(electrode)
    e = electrode[0]
    i_am_composite = check_if_composite(options, electrode)
    stoich_variables = _get_stoich_variables(options)
    direction = _get_direction(electrode)
    Q_1 = pybamm.InputParameter(f"Q_{e}_1")
    Q = (
        direction
        * (stoich_variables[f"{prefix}_100_1"] - stoich_variables[f"{prefix}_0_1"])
        * Q_1
    )
    if i_am_composite:
        Q_2 = pybamm.InputParameter(f"Q_{e}_2")
        Q += (
            direction
            * (stoich_variables[f"{prefix}_100_2"] - stoich_variables[f"{prefix}_0_2"])
            * Q_2
        )
    return Q


def _get_cyclable_lithium_equation(options, soc="100"):
    x_soc_1 = pybamm.Variable(f"x_{soc}_1", bounds=(0, 1))
    y_soc_1 = pybamm.Variable(f"y_{soc}_1", bounds=(0, 1))
    Q_n_1 = pybamm.InputParameter("Q_n_1")
    Q_p_1 = pybamm.InputParameter("Q_p_1")
    lithium_primary_phases = Q_n_1 * x_soc_1 + Q_p_1 * y_soc_1
    lithium_secondary_phases = 0.0
    is_positive_composite = check_if_composite(options, "positive")
    is_negative_composite = check_if_composite(options, "negative")
    if is_positive_composite:
        Q_p_2 = pybamm.InputParameter("Q_p_2")
        y_soc_2 = pybamm.Variable(f"y_{soc}_2", bounds=(0, 1))
        lithium_secondary_phases += Q_p_2 * y_soc_2
    if is_negative_composite:
        Q_n_2 = pybamm.InputParameter("Q_n_2")
        x_soc_2 = pybamm.Variable(f"x_{soc}_2", bounds=(0, 1))
        lithium_secondary_phases += Q_n_2 * x_soc_2
    return lithium_primary_phases + lithium_secondary_phases


def _solve_secondary_stoichiometry(
    primary_stoich,
    parameter_values,
    param,
    electrode,
    direction,
    options,
    T,
    tol=1e-6,
):
    """
    Solve U_prim(z_1) = U_sec(z_2) to get z_2 given z_1.

    Parameters
    ----------
    primary_stoich : float
        The primary phase stoichiometry (x_1 or y_1)
    parameter_values : pybamm.ParameterValues
        The parameter values
    param : pybamm.LithiumIonParameters
        The parameter object
    electrode : str
        "negative" or "positive"
    direction : str
        "charge" or "discharge"
    options : dict
        Model options
    T : float
        Temperature
    tol : float
        Solver tolerance

    Returns
    -------
    float
        The secondary phase stoichiometry (x_2 or y_2)
    """
    # Create a simple model: U_prim(z_1) - U_sec(z_2) = 0
    model = pybamm.BaseModel()
    z_2 = pybamm.Variable("z_2", bounds=(0, 1))
    z_1 = pybamm.InputParameter("z_1")

    if electrode == "negative":
        lith_prim = _get_lithiation_delithiation(
            direction, "negative", options, phase="primary"
        )
        lith_sec = _get_lithiation_delithiation(
            direction, "negative", options, phase="secondary"
        )
        U_prim = param.n.prim.U(z_1, T, lith_prim)
        U_sec = param.n.sec.U(z_2, T, lith_sec)
    else:
        lith_prim = _get_lithiation_delithiation(
            direction, "positive", options, phase="primary"
        )
        lith_sec = _get_lithiation_delithiation(
            direction, "positive", options, phase="secondary"
        )
        U_prim = param.p.prim.U(z_1, T, lith_prim)
        U_sec = param.p.sec.U(z_2, T, lith_sec)

    model.algebraic[z_2] = U_prim - U_sec
    model.initial_conditions[z_2] = primary_stoich  # Start from primary stoich
    model.variables["z_2"] = z_2

    sim = pybamm.Simulation(
        model, parameter_values=parameter_values, solver=pybamm.AlgebraicSolver(tol=tol)
    )
    sol = sim.solve([0], inputs={"z_1": primary_stoich})
    return sol["z_2"].data[0]


class ElectrodeSOHComposite(pybamm.BaseModel):
    """Model to calculate electrode-specific SOH for a cell with composite electrodes,
    adapted from :footcite:t:`Mohtat2019`. This model is mainly for internal use, to
    calculate summary variables in a
    simulation.

    Subscript 1 indicates primary phase and subscript 2 indicates secondary phase.
    """

    def __init__(
        self,
        options,
        direction=None,
        name="ElectrodeSOH model",
        initialization_method="voltage",
    ):
        pybamm.citations.register("Mohtat2019")
        super().__init__(name)
        param = pybamm.LithiumIonParameters(options)

        # Start by just assuming known value is cyclable lithium (and solve all at once)
        Q_Li = pybamm.InputParameter("Q_Li")
        is_negative_composite = check_if_composite(options, "negative")
        is_positive_composite = check_if_composite(options, "positive")
        variables = _get_stoich_variables(options)
        x_100_1 = variables["x_100_1"]
        y_100_1 = variables["y_100_1"]
        x_0_1 = variables["x_0_1"]
        y_0_1 = variables["y_0_1"]
        V_max = param.voltage_high_cut
        V_min = param.voltage_low_cut
        # Here we use T_ref as the stoichiometry limits are defined using the reference
        # state
        if is_negative_composite:
            x_100_2 = variables["x_100_2"]
            x_0_2 = variables["x_0_2"]
            self.algebraic[x_100_2] = param.n.sec.U(
                x_100_2,
                param.T_ref,
                _get_lithiation_delithiation(
                    direction, "negative", options, phase="secondary"
                ),
            ) - param.n.prim.U(
                x_100_1,
                param.T_ref,
                _get_lithiation_delithiation(
                    direction, "negative", options, phase="primary"
                ),
            )
            self.algebraic[x_0_2] = param.n.sec.U(
                x_0_2,
                param.T_ref,
                _get_lithiation_delithiation(
                    direction, "negative", options, phase="secondary"
                ),
            ) - param.n.prim.U(
                x_0_1,
                param.T_ref,
                _get_lithiation_delithiation(
                    direction, "negative", options, phase="primary"
                ),
            )
        if is_positive_composite:
            y_100_2 = variables["y_100_2"]
            y_0_2 = variables["y_0_2"]
            self.algebraic[y_100_2] = param.p.sec.U(
                y_100_2,
                param.T_ref,
                _get_lithiation_delithiation(
                    direction, "positive", options, phase="secondary"
                ),
            ) - param.p.prim.U(
                y_100_1,
                param.T_ref,
                _get_lithiation_delithiation(
                    direction, "positive", options, phase="primary"
                ),
            )
            self.algebraic[y_0_2] = param.p.prim.U(
                y_0_1,
                param.T_ref,
                _get_lithiation_delithiation(
                    direction, "positive", options, phase="primary"
                ),
            ) - param.p.sec.U(
                y_0_2,
                param.T_ref,
                _get_lithiation_delithiation(
                    direction, "positive", options, phase="secondary"
                ),
            )
        self.algebraic[x_100_1] = (
            param.p.prim.U(
                y_100_1,
                param.T_ref,
                _get_lithiation_delithiation(
                    direction, "positive", options, phase="primary"
                ),
            )
            - param.n.prim.U(
                x_100_1,
                param.T_ref,
                _get_lithiation_delithiation(
                    direction, "negative", options, phase="primary"
                ),
            )
            - V_max
        )
        self.algebraic[x_0_1] = (
            param.p.prim.U(
                y_0_1,
                param.T_ref,
                _get_lithiation_delithiation(
                    direction, "positive", options, phase="primary"
                ),
            )
            - param.n.prim.U(
                x_0_1,
                param.T_ref,
                _get_lithiation_delithiation(
                    direction, "negative", options, phase="primary"
                ),
            )
            - V_min
        )
        # arbitrary choice: use y_0_1 for the capacity equation
        self.algebraic[y_0_1] = _get_electrode_capacity_equation(
            options, "positive"
        ) - _get_electrode_capacity_equation(options, "negative")
        self.algebraic[y_100_1] = Q_Li - _get_cyclable_lithium_equation(options)

        x_init_1 = variables["x_init_1"]
        y_init_1 = variables["y_init_1"]
        if initialization_method == "voltage":
            # Here we use T_init so that the initial voltage is correct, including the
            # contribution from the entropic change
            V_init = pybamm.InputParameter("V_init")
            self.algebraic[x_init_1] = (
                param.p.prim.U(
                    y_init_1,
                    param.T_init,
                    _get_lithiation_delithiation(
                        direction, "positive", options, phase="primary"
                    ),
                )
                - param.n.prim.U(
                    x_init_1,
                    param.T_init,
                    _get_lithiation_delithiation(
                        direction, "negative", options, phase="primary"
                    ),
                )
                - V_init
            )
            self.algebraic[y_init_1] = (
                _get_cyclable_lithium_equation(options, "init") - Q_Li
            )
        elif initialization_method == "SOC":
            soc_init = pybamm.InputParameter("SOC_init")
            negative_soc = x_init_1 * pybamm.InputParameter("Q_n_1")
            if is_negative_composite:
                x_init_2 = variables["x_init_2"]
                negative_soc += x_init_2 * pybamm.InputParameter("Q_n_2")

            negative_0_soc = x_0_1 * pybamm.InputParameter("Q_n_1")
            if is_negative_composite:
                negative_0_soc += x_0_2 * pybamm.InputParameter("Q_n_2")

            negative_100_soc = x_100_1 * pybamm.InputParameter("Q_n_1")
            if is_negative_composite:
                negative_100_soc += x_100_2 * pybamm.InputParameter("Q_n_2")
            self.algebraic[x_init_1] = (
                (negative_soc - negative_0_soc) / (negative_100_soc - negative_0_soc)
            ) - soc_init
            self.algebraic[y_init_1] = (
                _get_cyclable_lithium_equation(options, "init") - Q_Li
            )
        else:
            raise ValueError("Invalid initialization method")
        # Add voltage equations for secondary phases (init), we use T_ref if setting
        # based on SOC since the stoichiometry limits are defined using the reference
        # state, and T_init if setting based on voltage since the entropic change is
        # included in the voltage equation
        T = param.T_init if initialization_method == "voltage" else param.T_ref
        if is_positive_composite:
            y_init_2 = variables["y_init_2"]
            self.algebraic[y_init_2] = param.p.prim.U(
                y_init_1,
                T,
                _get_lithiation_delithiation(
                    direction, "positive", options, phase="primary"
                ),
            ) - param.p.sec.U(
                y_init_2,
                T,
                _get_lithiation_delithiation(
                    direction, "positive", options, phase="secondary"
                ),
            )
        if is_negative_composite:
            x_init_2 = variables["x_init_2"]
            self.algebraic[x_init_2] = param.n.prim.U(
                x_init_1,
                T,
                _get_lithiation_delithiation(
                    direction, "negative", options, phase="primary"
                ),
            ) - param.n.sec.U(
                x_init_2,
                T,
                _get_lithiation_delithiation(
                    direction, "negative", options, phase="secondary"
                ),
            )

        self.variables.update(variables)
        if initialization_method == "SOC":
            soc_init = pybamm.InputParameter("SOC_init")
        else:
            soc_init = (V_init - V_min) / (V_max - V_min)
        self.initial_conditions.update(_get_initial_conditions(options, soc_init))

    @property
    def default_solver(self):
        return pybamm.AlgebraicSolver(method="lsq")

    @staticmethod
    def solve_split(
        initial_value,
        parameter_values,
        direction=None,
        param=None,
        options=None,
        tol=1e-6,
        inputs=None,
    ):
        """
        Split solve approach for composite electrode SOH.

        Step 1: Solve for primary stoichiometries using non-composite model
        Step 2: Solve U_prim(z_1) = U_sec(z_2) for secondary stoichiometries

        Parameters
        ----------
        initial_value : float or str
            Target initial value. If float (0-1), interpreted as SOC.
            If string ending in 'V', interpreted as voltage.
        parameter_values : pybamm.ParameterValues
            Parameter values for the simulation
        direction : str, optional
            "charge" or "discharge" for hysteresis direction
        param : pybamm.LithiumIonParameters, optional
            Parameter object
        options : dict, optional
            Model options
        tol : float, optional
            Solver tolerance (default 1e-6)
        inputs : dict, optional
            Additional inputs

        Returns
        -------
        dict
            Dictionary of stoichiometry values
        """
        inputs = inputs or {}
        param = param or pybamm.LithiumIonParameters(options)

        is_positive_composite = check_if_composite(options, "positive")
        is_negative_composite = check_if_composite(options, "negative")

        # Get electrode capacities
        Q_n_1 = parameter_values.evaluate(param.n.prim.Q_init, inputs=inputs)
        Q_p_1 = parameter_values.evaluate(param.p.prim.Q_init, inputs=inputs)
        Qs = {"Q_n_1": Q_n_1, "Q_p_1": Q_p_1}
        if is_positive_composite:
            Q_p_2 = parameter_values.evaluate(param.p.sec.Q_init, inputs=inputs)
            Qs["Q_p_2"] = Q_p_2
        if is_negative_composite:
            Q_n_2 = parameter_values.evaluate(param.n.sec.Q_init, inputs=inputs)
            Qs["Q_n_2"] = Q_n_2

        Q_Li = parameter_values.evaluate(param.Q_Li_particles_init, inputs=inputs)

        # Determine initialization method
        if isinstance(initial_value, str) and initial_value.endswith("V"):
            V_init = float(initial_value[:-1])
            initialization_method = "voltage"
        elif isinstance(initial_value, float) and 0 <= initial_value <= 1:
            initialization_method = "SOC"
        else:
            raise ValueError("Invalid initial value")

        # Step 1: Solve for primary stoichiometries
        Q_n_total = Q_n_1 + (Qs.get("Q_n_2", 0))
        Q_p_total = Q_p_1 + (Qs.get("Q_p_2", 0))

        primary_options = _get_primary_only_options(options)
        primary_model = _ElectrodeSOH(
            direction=direction,
            param=param,
            solve_for=["x_0", "x_100"],
            known_value="cyclable lithium capacity",
            options=primary_options,
        )

        primary_inputs = {**inputs, "Q_n": Q_n_total, "Q_p": Q_p_total, "Q_Li": Q_Li}
        primary_sim = pybamm.Simulation(
            primary_model,
            parameter_values=parameter_values,
            solver=pybamm.AlgebraicSolver(tol=tol),
        )

        primary_sol = primary_sim.solve([0], inputs=primary_inputs)
        x_100_1 = primary_sol["x_100"].data[0]
        x_0_1 = primary_sol["x_0"].data[0]
        y_100_1 = primary_sol["y_100"].data[0]
        y_0_1 = primary_sol["y_0"].data[0]

        # Step 2: Solve for secondary stoichiometries
        T_ref = parameter_values["Reference temperature [K]"]
        result = {
            "x_100_1": x_100_1,
            "x_0_1": x_0_1,
            "y_100_1": y_100_1,
            "y_0_1": y_0_1,
        }

        # Equilibrium stoichiometries (100%, 0% SOC) use direction=None
        if is_negative_composite:
            result["x_100_2"] = _solve_secondary_stoichiometry(
                x_100_1, parameter_values, param, "negative", None, options, T_ref, tol
            )
            result["x_0_2"] = _solve_secondary_stoichiometry(
                x_0_1, parameter_values, param, "negative", None, options, T_ref, tol
            )

        if is_positive_composite:
            result["y_100_2"] = _solve_secondary_stoichiometry(
                y_100_1, parameter_values, param, "positive", None, options, T_ref, tol
            )
            result["y_0_2"] = _solve_secondary_stoichiometry(
                y_0_1, parameter_values, param, "positive", None, options, T_ref, tol
            )

        # Solve for initial stoichiometries
        if initialization_method == "voltage":
            T_init = parameter_values["Initial temperature [K]"]
            soc_model = pybamm.BaseModel()
            x_init = pybamm.Variable("x_init", bounds=(0, 1))
            y_init = y_0_1 + (x_init - x_0_1) / (x_100_1 - x_0_1) * (y_100_1 - y_0_1)
            lith_pos = _get_lithiation_delithiation(
                direction, "positive", options, phase="primary"
            )
            lith_neg = _get_lithiation_delithiation(
                direction, "negative", options, phase="primary"
            )
            Up = param.p.prim.U(y_init, T_init, lith_pos)
            Un = param.n.prim.U(x_init, T_init, lith_neg)
            soc_model.algebraic[x_init] = Up - Un - V_init
            soc_model.initial_conditions[x_init] = (x_0_1 + x_100_1) / 2
            soc_model.variables["x_init"] = x_init
            soc_model.variables["y_init"] = y_init
            soc_sim = pybamm.Simulation(
                soc_model,
                parameter_values=parameter_values,
                solver=pybamm.AlgebraicSolver(tol=tol),
            )
            soc_sol = soc_sim.solve([0], inputs=inputs)
            x_init_1 = soc_sol["x_init"].data[0]
            y_init_1 = soc_sol["y_init"].data[0]
        else:  # SOC initialization
            soc = initial_value
            x_init_1 = x_0_1 + soc * (x_100_1 - x_0_1)
            y_init_1 = y_0_1 + soc * (y_100_1 - y_0_1)

        result["x_init_1"] = x_init_1
        result["y_init_1"] = y_init_1

        # Solve for secondary init stoichiometries (use direction for hysteresis)
        T_for_init = (
            parameter_values["Initial temperature [K]"]
            if initialization_method == "voltage"
            else T_ref
        )
        if is_negative_composite:
            result["x_init_2"] = _solve_secondary_stoichiometry(
                x_init_1,
                parameter_values,
                param,
                "negative",
                direction,
                options,
                T_for_init,
                tol,
            )
        if is_positive_composite:
            result["y_init_2"] = _solve_secondary_stoichiometry(
                y_init_1,
                parameter_values,
                param,
                "positive",
                direction,
                options,
                T_for_init,
                tol,
            )

        return result

    @staticmethod
    def solve_full(
        initial_value,
        parameter_values,
        direction=None,
        param=None,
        options=None,
        tol=1e-6,
        inputs=None,
    ):
        """
        Full solve approach: solve all stoichiometries simultaneously.

        Parameters
        ----------
        initial_value : float or str
            Target initial value. If float (0-1), interpreted as SOC.
            If string ending in 'V', interpreted as voltage.
        parameter_values : pybamm.ParameterValues
            Parameter values for the simulation
        direction : str, optional
            "charge" or "discharge" for hysteresis direction
        param : pybamm.LithiumIonParameters, optional
            Parameter object
        options : dict, optional
            Model options
        tol : float, optional
            Solver tolerance (default 1e-6)
        inputs : dict, optional
            Additional inputs

        Returns
        -------
        dict
            Dictionary of stoichiometry values
        """
        inputs = inputs or {}
        param = param or pybamm.LithiumIonParameters(options)

        is_positive_composite = check_if_composite(options, "positive")
        is_negative_composite = check_if_composite(options, "negative")

        # Get electrode capacities
        Q_n_1 = parameter_values.evaluate(param.n.prim.Q_init, inputs=inputs)
        Q_p_1 = parameter_values.evaluate(param.p.prim.Q_init, inputs=inputs)
        Qs = {"Q_n_1": Q_n_1, "Q_p_1": Q_p_1}
        if is_positive_composite:
            Q_p_2 = parameter_values.evaluate(param.p.sec.Q_init, inputs=inputs)
            Qs["Q_p_2"] = Q_p_2
        if is_negative_composite:
            Q_n_2 = parameter_values.evaluate(param.n.sec.Q_init, inputs=inputs)
            Qs["Q_n_2"] = Q_n_2

        Q_Li = parameter_values.evaluate(param.Q_Li_particles_init, inputs=inputs)

        # Determine initialization method
        if isinstance(initial_value, str) and initial_value.endswith("V"):
            V_init = float(initial_value[:-1])
            initialization_method = "voltage"
        elif isinstance(initial_value, float) and 0 <= initial_value <= 1:
            initialization_method = "SOC"
        else:
            raise ValueError("Invalid initial value")

        # Solve all stoichiometries simultaneously using the algebraic model
        all_inputs = {**inputs, **Qs, "Q_Li": Q_Li}
        if initialization_method == "voltage":
            all_inputs["V_init"] = V_init
        else:
            all_inputs["SOC_init"] = initial_value

        model = ElectrodeSOHComposite(
            options, direction, initialization_method=initialization_method
        )
        sim = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            solver=pybamm.AlgebraicSolver(tol=tol),
        )
        sol = sim.solve([0, 1], inputs=all_inputs)
        return {var: sol[var].entries[0] for var in model.variables.keys()}


def get_initial_stoichiometries_composite(
    initial_value,
    parameter_values,
    direction=None,
    param=None,
    options=None,
    tol=1e-6,
    inputs=None,
    known_value="cyclable lithium capacity",
    solve_method="split",
    **kwargs,
):
    """
    Get the minimum and maximum stoichiometries from the parameter values.

    Parameters
    ----------
    initial_value : float
        Target initial value.
        If integer, interpreted as SOC, must be between 0 and 1.
        If string e.g. "4 V", interpreted as voltage,
        must be between V_min and V_max.
    direction : str, optional
        The OCV branch to use in the electrode SOH model. Can be "charge" or
        "discharge".
    tol : float, optional
        The tolerance for the solver used to compute the initial stoichiometries.
        A lower value results in higher precision but may increase computation time.
        Default is 1e-6.
    inputs : dict, optional
        A dictionary of input parameters passed to the model.
    known_value : str, optional
        The known value needed to complete the electrode SOH model.
        Can be "cyclable lithium capacity".
    solve_method : str, optional
        The solve method to use. Can be "split" (default) or "full".
        - "split": Solve primary stoichiometries first, then secondary
        - "full": Solve all stoichiometries simultaneously
    """
    inputs = inputs or {}
    param = param or pybamm.LithiumIonParameters(options)

    if known_value != "cyclable lithium capacity":
        raise ValueError(
            "Only `cyclable lithium capacity` is supported for composite electrodes"
        )

    if solve_method == "full":
        return ElectrodeSOHComposite.solve_full(
            initial_value,
            parameter_values,
            direction=direction,
            param=param,
            options=options,
            tol=tol,
            inputs=inputs,
        )
    elif solve_method == "split":
        return ElectrodeSOHComposite.solve_split(
            initial_value,
            parameter_values,
            direction=direction,
            param=param,
            options=options,
            tol=tol,
            inputs=inputs,
        )
    else:
        raise ValueError(f"Invalid solve method: {solve_method}")
