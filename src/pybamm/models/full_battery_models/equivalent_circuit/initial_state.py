import numpy as np

import pybamm


def set_initial_state(
    initial_value,
    parameter_values,
    direction=None,
    param=None,
    inplace=True,
    options=None,
    inputs=None,
    tol=1e-6,
):
    """
    Set the value of the initial state of charge.

    Parameters
    ----------
    initial_value : float
        Target initial value.
        If float, interpreted as SOC, must be between 0 and 1.
        If string e.g. "4 V", interpreted as voltage, must be between V_min and V_max.
    parameter_values : :class:`pybamm.ParameterValues`
        Parameters and their corresponding values.
    param : :class:`pybamm.LithiumIonParameters`, optional
        The symbolic parameter set to use for the simulation.
        If not provided, the default parameter set will be used.
    inplace: bool, optional
        If True, replace the parameters values in place. Otherwise, return a new set of
        parameter values. Default is True.
    options : dict-like, optional
        A dictionary of options to be passed to the model, see
        :class:`pybamm.BatteryModelOptions`.
    inputs : dict, optional
        A dictionary of input parameters to pass to the model when solving.
    tol : float, optional
        The tolerance for the solver used to compute the initial stoichiometries.
        A lower value results in higher precision but may increase computation time.
        Default is 1e-6.
    """
    parameter_values = parameter_values if inplace else parameter_values.copy()
    param = param or pybamm.EcmParameters()

    if isinstance(initial_value, str) and initial_value.endswith("V"):
        V_init = float(initial_value[:-1])
        V_min = parameter_values.evaluate(param.voltage_low_cut, inputs=inputs)
        V_max = parameter_values.evaluate(param.voltage_high_cut, inputs=inputs)

        if not V_min <= V_init <= V_max:
            raise ValueError(
                f"Initial voltage {V_init}V is outside the voltage limits "
                f"({V_min}, {V_max})"
            )

        # Solve simple model for initial soc based on target voltage
        soc_model = pybamm.BaseModel()
        soc = pybamm.Variable("soc")
        soc_model.algebraic[soc] = param.ocv(soc) - V_init

        # initial guess for soc linearly interpolates between 0 and 1
        # based on V linearly interpolating between V_max and V_min
        soc_model.initial_conditions[soc] = (V_init - V_min) / (V_max - V_min)
        soc_model.variables["soc"] = soc
        parameter_values.process_model(soc_model)
        initial_soc = (
            pybamm.AlgebraicSolver(tol=tol)
            .solve(soc_model, [0], inputs=inputs)["soc"]
            .data[0]
        )

        # Ensure that the result lies between 0 and 1
        parameter_values["Initial SoC"] = np.minimum(np.maximum(initial_soc, 0.0), 1.0)

    elif isinstance(initial_value, int | float):
        if not 0 <= initial_value <= 1:
            raise ValueError("Initial SOC should be between 0 and 1")
        parameter_values["Initial SoC"] = initial_value

    else:
        raise ValueError(
            "Initial value must be a float between 0 and 1, or a string ending in 'V'"
        )

    return parameter_values
