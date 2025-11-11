#
# A model to calculate electrode-specific SOH, adapted to a half-cell
#
import pybamm

from .util import _get_lithiation_delithiation, check_if_composite


class ElectrodeSOHHalfCell(pybamm.BaseModel):
    """
    Model to calculate electrode-specific SOH for a half-cell, adapted from
    :footcite:t:`Mohtat2019`.
    This model is mainly for internal use, to calculate summary variables in a
    simulation.

    .. math::
        V_{max} = U_w(x_{100}),
    .. math::
        V_{min} = U_w(x_{0}),
    .. math::
        x_0 = x_{100} - \\frac{C}{C_w}.

    Subscript w indicates working electrode and subscript c indicates counter electrode.

    """

    def __init__(self, name="ElectrodeSOH model", direction=None, options=None):
        pybamm.citations.register("Mohtat2019")
        super().__init__(name)
        options = options or {"working electrode": "positive"}
        param = pybamm.LithiumIonParameters(options)
        is_composite = check_if_composite(options, "positive")

        # Primary phase
        x_100 = pybamm.Variable("x_100", bounds=(0, 1))
        x_0 = pybamm.Variable("x_0", bounds=(0, 1))
        Q_w = pybamm.InputParameter("Q_w")
        T_ref = param.T_ref
        U_w = param.p.prim.U
        lith_delith_primary = _get_lithiation_delithiation(
            direction, "positive", options, phase="primary"
        )

        # Secondary phase
        Q_2 = pybamm.Scalar(0)
        if is_composite:
            x_100_2 = pybamm.Variable("x_100_2", bounds=(0, 1))
            x_0_2 = pybamm.Variable("x_0_2", bounds=(0, 1))
            Q_w_2 = pybamm.InputParameter("Q_w_2")
            U_w_2 = param.p.sec.U
            Q_2 = Q_w_2 * (x_0_2 - x_100_2)
            lith_delith_secondary = _get_lithiation_delithiation(
                direction, "positive", options, phase="secondary"
            )

        Q_1 = Q_w * (x_0 - x_100)
        Q = Q_1 + Q_2

        V_max = param.ocp_soc_100
        V_min = param.ocp_soc_0

        self.algebraic = {
            x_100: U_w(x_100, T_ref, lith_delith_primary) - V_max,
            x_0: U_w(x_0, T_ref, lith_delith_primary) - V_min,
        }
        self.initial_conditions = {x_100: 0.8, x_0: 0.2}
        if is_composite:
            self.algebraic[x_100_2] = (
                U_w_2(x_100_2, T_ref, lith_delith_secondary) - V_max
            )
            self.algebraic[x_0_2] = U_w_2(x_0_2, T_ref, lith_delith_secondary) - V_min
            self.initial_conditions[x_100_2] = 0.8
            self.initial_conditions[x_0_2] = 0.2

        self.variables = {
            "x_100": x_100,
            "x_0": x_0,
            "Q": Q,
            "Uw(x_100)": U_w(x_100, T_ref, lith_delith_primary),
            "Uw(x_0)": U_w(x_0, T_ref, lith_delith_primary),
            "Q_w": Q_w,
        }
        if is_composite:
            self.variables.update(
                {
                    "x_100_2": x_100_2,
                    "x_0_2": x_0_2,
                    "Q_w_2": Q_w_2,
                    "Uw(x_100_2)": U_w_2(x_100_2, T_ref, lith_delith_secondary),
                    "Uw(x_0_2)": U_w_2(x_0_2, T_ref, lith_delith_secondary),
                }
            )

    @property
    def default_solver(self):
        # Use AlgebraicSolver as CasadiAlgebraicSolver gives unnecessary warnings
        return pybamm.AlgebraicSolver(method="lsq__trf", tol=1e-7)


def get_initial_stoichiometry_half_cell(
    initial_value,
    parameter_values,
    param=None,
    options=None,
    tol=1e-6,
    inputs=None,
    direction=None,
    **kwargs,
):
    """
    Calculate initial stoichiometry to start off the simulation at a particular
    state of charge, given voltage limits, open-circuit potential, etc defined by
    parameter_values

    Parameters
    ----------
    initial_value : float
        Target initial value.
        If integer, interpreted as SOC, must be between 0 and 1.
        If string e.g. "4 V", interpreted as voltage,
        must be between V_min and V_max.
    parameter_values : pybamm.ParameterValues
        The parameter values to use in the calculation
    param : :class:`pybamm.LithiumIonParameters`, optional
        The symbolic parameter set to use for the simulation.
        If not provided, the default parameter set will be used.
    tol : float, optional
        The tolerance for the solver used to compute the initial stoichiometries.
        Default is 1e-6.
    inputs : dict, optional
        A dictionary of input parameters passed to the model.

    Returns
    -------
    x
        The initial stoichiometry that give the desired initial state of charge
    """
    param = pybamm.LithiumIonParameters(options)
    x_dict = get_min_max_stoichiometries(
        parameter_values, inputs=inputs, direction=direction, options=options
    )
    x_0, x_100 = x_dict["x_0"], x_dict["x_100"]
    is_composite = check_if_composite(options, "positive")
    lith_delith_primary = _get_lithiation_delithiation(
        direction, "positive", options, phase="primary"
    )
    if is_composite:
        x_0_2, x_100_2 = x_dict["x_0_2"], x_dict["x_100_2"]
        lith_delith_secondary = _get_lithiation_delithiation(
            direction, "positive", options, phase="secondary"
        )

    if isinstance(initial_value, str) and initial_value.endswith("V"):
        V_init = float(initial_value[:-1])
        V_min = parameter_values.evaluate(param.voltage_low_cut, inputs=inputs)
        V_max = parameter_values.evaluate(param.voltage_high_cut, inputs=inputs)

        if not V_min - tol <= V_init <= V_max + tol:
            raise ValueError(
                f"Initial voltage {V_init}V is outside the voltage limits "
                f"({V_min}, {V_max})"
            )
        # Solve simple model for initial soc based on target voltage
        model = pybamm.BaseModel()
        x = pybamm.Variable("x")
        Up = param.p.prim.U
        T_init = param.T_init
        model.variables["x"] = x
        model.algebraic[x] = Up(x, T_init, lith_delith_primary) - V_init
        # initial guess for x linearly interpolates between 0 and 1
        # based on V linearly interpolating between V_max and V_min
        soc_initial_guess = (V_init - V_min) / (V_max - V_min)
        model.initial_conditions[x] = 1 - soc_initial_guess
        if is_composite:
            Up_2 = param.p.sec.U
            x_2 = pybamm.Variable("x_2")
            model.algebraic[x_2] = Up_2(x_2, T_init, lith_delith_secondary) - V_init
            model.variables["x_2"] = x_2
            model.initial_conditions[x_2] = 1 - soc_initial_guess

        parameter_values.process_model(model)
        sol = pybamm.AlgebraicSolver("lsq__trf", tol=tol).solve(
            model, [0], inputs=inputs
        )
        x = sol["x"].data[0]
        if is_composite:
            x_2 = sol["x_2"].data[0]
    elif isinstance(initial_value, int | float):
        if not 0 <= initial_value <= 1:
            raise ValueError("Initial SOC should be between 0 and 1")
        if not is_composite:
            x = x_0 + initial_value * (x_100 - x_0)
        else:
            model = pybamm.BaseModel()
            x = pybamm.Variable("x")
            x_2 = pybamm.Variable("x_2")
            U_p = param.p.prim.U
            U_p_2 = param.p.sec.U
            # here we use T_ref and SOC 0 and 1 are defined using the reference state
            T_ref = param.T_ref
            model.algebraic[x] = U_p(x, T_ref, lith_delith_primary) - U_p_2(
                x_2, T_ref, lith_delith_secondary
            )
            model.initial_conditions[x] = x_0 + initial_value * (x_100 - x_0)
            model.initial_conditions[x_2] = x_0_2 + initial_value * (x_100_2 - x_0_2)
            Q_w = parameter_values.evaluate(param.p.prim.Q_init, inputs=inputs)
            Q_w_2 = parameter_values.evaluate(param.p.sec.Q_init, inputs=inputs)
            Q_min = x_100 * Q_w + x_100_2 * Q_w_2
            Q_max = x_0 * Q_w + x_0_2 * Q_w_2
            Q_now = Q_w * x + Q_w_2 * x_2
            soc = (Q_now - Q_min) / (Q_max - Q_min)
            model.algebraic[x_2] = soc - initial_value
            model.variables["x"] = x
            model.variables["x_2"] = x_2
            parameter_values.process_model(model)
            sol = pybamm.AlgebraicSolver(tol=tol).solve(model, [0], inputs=inputs)
            x = sol["x"].data[0]
            x_2 = sol["x_2"].data[0]
    else:
        raise ValueError(
            "Initial value must be a float between 0 and 1, or a string ending in 'V'"
        )
    ret_dict = {"x": x}
    if is_composite:
        ret_dict["x_2"] = x_2
    return ret_dict


def get_min_max_stoichiometries(
    parameter_values, options=None, inputs=None, direction=None
):
    """
    Get the minimum and maximum stoichiometries from the parameter values

    Parameters
    ----------
    parameter_values : pybamm.ParameterValues
        The parameter values to use in the calculation
    options : dict, optional
        A dictionary of options to be passed to the parameters, see
        :class:`pybamm.BatteryModelOptions`.
        If None, the default is used: {"working electrode": "positive"}
    """
    inputs = inputs or {}
    if options is None:
        options = {"working electrode": "positive"}
    esoh_model = pybamm.lithium_ion.ElectrodeSOHHalfCell(
        "ElectrodeSOH", direction=direction, options=options
    )
    param = pybamm.LithiumIonParameters(options)
    is_composite = check_if_composite(options, "positive")
    if is_composite:
        Q_w = parameter_values.evaluate(param.p.prim.Q_init, inputs=inputs)
        Q_w_2 = parameter_values.evaluate(param.p.sec.Q_init, inputs=inputs)
        Q_inputs = {"Q_w": Q_w, "Q_w_2": Q_w_2}
    else:
        Q_w = parameter_values.evaluate(param.p.prim.Q_init, inputs=inputs)
        Q_inputs = {"Q_w": Q_w}
    # Add Q_w to input parameters
    all_inputs = {**inputs, **Q_inputs}
    esoh_sim = pybamm.Simulation(esoh_model, parameter_values=parameter_values)
    esoh_sol = esoh_sim.solve([0], inputs=all_inputs)
    x_0, x_100 = esoh_sol["x_0"].data[0], esoh_sol["x_100"].data[0]
    if is_composite:
        x_0_2, x_100_2 = esoh_sol["x_0_2"].data[0], esoh_sol["x_100_2"].data[0]
        ret_dict = {"x_0": x_0, "x_100": x_100, "x_0_2": x_0_2, "x_100_2": x_100_2}
    else:
        ret_dict = {"x_0": x_0, "x_100": x_100}
    return ret_dict
