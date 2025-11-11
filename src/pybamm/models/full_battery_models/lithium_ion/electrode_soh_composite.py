#
# A model to calculate electrode-specific SOH, adapted for composite electrodes
#
import pybamm

from .util import _get_lithiation_delithiation, check_if_composite


def _get_stoich_variables(options):
    variables = {
        "x_100_1": pybamm.Variable("x_100_1"),
        "y_100_1": pybamm.Variable("y_100_1"),
        "x_0_1": pybamm.Variable("x_0_1"),
        "y_0_1": pybamm.Variable("y_0_1"),
        "x_init_1": pybamm.Variable("x_init_1"),
        "y_init_1": pybamm.Variable("y_init_1"),
    }
    is_positive_composite = check_if_composite(options, "positive")
    is_negative_composite = check_if_composite(options, "negative")
    if is_positive_composite:
        variables["y_100_2"] = pybamm.Variable("y_100_2")
        variables["y_0_2"] = pybamm.Variable("y_0_2")
        variables["y_init_2"] = pybamm.Variable("y_init_2")
    if is_negative_composite:
        variables["x_100_2"] = pybamm.Variable("x_100_2")
        variables["x_0_2"] = pybamm.Variable("x_0_2")
        variables["x_init_2"] = pybamm.Variable("x_init_2")
    return variables


def _get_initial_conditions(options, soc_init):
    variables = _get_stoich_variables(options)
    ics = {}
    for name, var in variables.items():
        if "100" in name and "x" in name:
            ics[var] = 0.8
        elif "0" in name and "x" in name:
            ics[var] = 0.2
        elif "100" in name and "y" in name:
            ics[var] = 0.8
        elif "0" in name and "y" in name:
            ics[var] = 0.2
        elif "init" in name and "x" in name:
            ics[var] = soc_init
        elif "init" in name and "y" in name:
            ics[var] = 1 - soc_init
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
    x_soc_1 = pybamm.Variable(f"x_{soc}_1")
    y_soc_1 = pybamm.Variable(f"y_{soc}_1")
    Q_n_1 = pybamm.InputParameter("Q_n_1")
    Q_p_1 = pybamm.InputParameter("Q_p_1")
    lithium_primary_phases = Q_n_1 * x_soc_1 + Q_p_1 * y_soc_1
    lithium_secondary_phases = 0.0
    is_positive_composite = check_if_composite(options, "positive")
    is_negative_composite = check_if_composite(options, "negative")
    if is_positive_composite:
        Q_p_2 = pybamm.InputParameter("Q_p_2")
        y_soc_2 = pybamm.Variable(f"y_{soc}_2")
        lithium_secondary_phases += Q_p_2 * y_soc_2
    if is_negative_composite:
        Q_n_2 = pybamm.InputParameter("Q_n_2")
        x_soc_2 = pybamm.Variable(f"x_{soc}_2")
        lithium_secondary_phases += Q_n_2 * x_soc_2
    return lithium_primary_phases + lithium_secondary_phases


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
            soc_init = 0.5
        self.initial_conditions.update(_get_initial_conditions(options, soc_init))

    @property
    def default_solver(self):
        # Use AlgebraicSolver as CasadiAlgebraicSolver gives unnecessary warnings
        return pybamm.AlgebraicSolver()


def get_initial_stoichiometries_composite(
    initial_value,
    parameter_values,
    direction=None,
    param=None,
    options=None,
    tol=1e-6,
    inputs=None,
    known_value="cyclable lithium capacity",
    **kwargs,
):
    """
    Get the minimum and maximum stoichiometries from the parameter values

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
    """
    inputs = inputs or {}
    if known_value != "cyclable lithium capacity":
        raise ValueError(
            "Only `cyclable lithium capacity` is supported for composite electrodes"
        )

    Q_n_1 = parameter_values.evaluate(param.n.prim.Q_init, inputs=inputs)
    Q_p_1 = parameter_values.evaluate(param.p.prim.Q_init, inputs=inputs)
    is_positive_composite = check_if_composite(options, "positive")
    is_negative_composite = check_if_composite(options, "negative")
    Qs = {
        "Q_n_1": Q_n_1,
        "Q_p_1": Q_p_1,
    }
    if is_positive_composite:
        Q_p_2 = parameter_values.evaluate(param.p.sec.Q_init, inputs=inputs)
        Qs["Q_p_2"] = Q_p_2
    if is_negative_composite:
        Q_n_2 = parameter_values.evaluate(param.n.sec.Q_init, inputs=inputs)
        Qs["Q_n_2"] = Q_n_2

    Q_Li = parameter_values.evaluate(param.Q_Li_particles_init, inputs=inputs)
    all_inputs = {**inputs, **Qs, "Q_Li": Q_Li}
    # Solve the model and check outputs
    if isinstance(initial_value, str) and initial_value.endswith("V"):
        all_inputs["V_init"] = float(initial_value[:-1])
        initialization_method = "voltage"
    elif isinstance(initial_value, float) and initial_value >= 0 and initial_value <= 1:
        initialization_method = "SOC"
        all_inputs["SOC_init"] = initial_value
    else:
        raise ValueError("Invalid initial value")
    model = ElectrodeSOHComposite(
        options, direction, initialization_method=initialization_method
    )
    sim = pybamm.Simulation(
        model, parameter_values=parameter_values, solver=pybamm.AlgebraicSolver(tol=tol)
    )
    sol = sim.solve([0, 1], inputs=all_inputs)
    return {var: sol[var].entries[0] for var in model.variables.keys()}
