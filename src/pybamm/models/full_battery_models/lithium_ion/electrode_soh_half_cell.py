#
# A model to calculate electrode-specific SOH, adapted to a half-cell
#
import pybamm


class ElectrodeSOHHalfCell(pybamm.BaseModel):
    """Model to calculate electrode-specific SOH for a half-cell, adapted from
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

    def __init__(self, name="ElectrodeSOH model"):
        pybamm.citations.register("Mohtat2019")
        super().__init__(name)
        param = pybamm.LithiumIonParameters({"working electrode": "positive"})

        x_100 = pybamm.Variable("x_100", bounds=(0, 1))
        x_0 = pybamm.Variable("x_0", bounds=(0, 1))
        Q_w = pybamm.InputParameter("Q_w")
        T_ref = param.T_ref
        U_w = param.p.prim.U
        Q = Q_w * (x_100 - x_0)

        V_max = param.ocp_soc_100
        V_min = param.ocp_soc_0

        self.algebraic = {
            x_100: U_w(x_100, T_ref) - V_max,
            x_0: U_w(x_0, T_ref) - V_min,
        }
        self.initial_conditions = {x_100: 0.8, x_0: 0.2}

        self.variables = {
            "x_100": x_100,
            "x_0": x_0,
            "Q": Q,
            "Uw(x_100)": U_w(x_100, T_ref),
            "Uw(x_0)": U_w(x_0, T_ref),
            "Q_w": Q_w,
            "Q_w * (x_100 - x_0)": Q_w * (x_100 - x_0),
        }

    @property
    def default_solver(self):
        # Use AlgebraicSolver as CasadiAlgebraicSolver gives unnecessary warnings
        return pybamm.AlgebraicSolver()


def get_initial_stoichiometry_half_cell(
    initial_value,
    parameter_values,
    param=None,
    options=None,
    tol=1e-6,
    inputs=None,
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
    x_0, x_100 = get_min_max_stoichiometries(parameter_values, inputs=inputs)

    if isinstance(initial_value, str) and initial_value.endswith("V"):
        V_init = float(initial_value[:-1])
        V_min = parameter_values.evaluate(param.voltage_low_cut)
        V_max = parameter_values.evaluate(param.voltage_high_cut)

        if not V_min - tol <= V_init <= V_max + tol:
            raise ValueError(
                f"Initial voltage {V_init}V is outside the voltage limits "
                f"({V_min}, {V_max})"
            )

        # Solve simple model for initial soc based on target voltage
        soc_model = pybamm.BaseModel()
        soc = pybamm.Variable("soc")
        Up = param.p.prim.U
        T_ref = parameter_values["Reference temperature [K]"]
        x = x_0 + soc * (x_100 - x_0)

        soc_model.algebraic[soc] = Up(x, T_ref) - V_init
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

    return x


def get_min_max_stoichiometries(parameter_values, options=None, inputs=None):
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
    esoh_model = pybamm.lithium_ion.ElectrodeSOHHalfCell("ElectrodeSOH")
    param = pybamm.LithiumIonParameters(options)
    Q_w = parameter_values.evaluate(param.p.Q_init, inputs=inputs)
    # Add Q_w to input parameters
    all_inputs = {**inputs, "Q_w": Q_w}
    esoh_sim = pybamm.Simulation(esoh_model, parameter_values=parameter_values)
    esoh_sol = esoh_sim.solve([0], inputs=all_inputs)
    x_0, x_100 = esoh_sol["x_0"].data[0], esoh_sol["x_100"].data[0]
    return x_0, x_100
