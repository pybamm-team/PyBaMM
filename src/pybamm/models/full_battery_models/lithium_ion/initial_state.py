import pybamm

from .util import _has_hysteresis, check_if_composite


def _set_hysteresis_branch(parameter_values, electrode, direction, options, phase=None):
    phase = phase or ""
    if phase != "":
        phase_prefactor = phase.capitalize() + ": "
    else:
        phase_prefactor = ""
    if direction is None:
        initial_hysteresis_branch = 0
    else:
        if (direction == "discharge" and electrode == "negative") or (
            direction == "charge" and electrode == "positive"
        ):
            initial_hysteresis_branch = 1
        elif (direction == "charge" and electrode == "negative") or (
            direction == "discharge" and electrode == "positive"
        ):
            initial_hysteresis_branch = -1
        else:
            raise ValueError(f"Invalid direction: {direction}")
    parameter_values.update(
        {
            f"{phase_prefactor}Initial hysteresis state in {electrode} electrode": initial_hysteresis_branch,
        }
    )
    return parameter_values


def set_initial_state(
    initial_value,
    parameter_values,
    direction=None,
    param=None,
    known_value="cyclable lithium capacity",
    inplace=True,
    options=None,
    inputs=None,
    tol=1e-6,
):
    """
    Set the values of the parameters representing the initial model states.

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
    known_value : str, optional
        The known value needed to complete the electrode SOH model.
        Can be "cyclable lithium capacity" (default) or "cell capacity".
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
    options = options or {}
    parameter_values = parameter_values if inplace else parameter_values.copy()
    param = param or pybamm.LithiumIonParameters(options)

    for electrode in ["negative", "positive"]:
        if check_if_composite(options, electrode):
            for phase in ["primary", "secondary"]:
                if _has_hysteresis(electrode, options, phase):
                    parameter_values = _set_hysteresis_branch(
                        parameter_values, electrode, direction, options, phase
                    )
        else:
            if _has_hysteresis(electrode, options):
                parameter_values = _set_hysteresis_branch(
                    parameter_values, electrode, direction, options
                )

    if options is not None and options.get("open-circuit potential", None) == "MSMR":
        """
        Set the initial OCP of each electrode, based on the initial SOC or voltage.
        """
        Un, Up = pybamm.lithium_ion.get_initial_ocps(
            initial_value,
            parameter_values,
            direction=direction,
            param=param,
            known_value=known_value,
            options=options,
            tol=tol,
            inputs=inputs,
        )
        parameter_values.update(
            {
                "Initial voltage in negative electrode [V]": Un,
                "Initial voltage in positive electrode [V]": Up,
            }
        )
    elif options is not None and options.get("working electrode", None) == "positive":
        """
        Set the initial stoichiometry of the working electrode, based on the initial
        SOC or voltage.
        """
        results = pybamm.lithium_ion.get_initial_stoichiometry_half_cell(
            initial_value,
            parameter_values,
            param=param,
            options=options,
            tol=tol,
            inputs=inputs,
            direction=direction,
        )
        _set_concentration_from_stoich(
            parameter_values,
            param,
            "positive",
            "primary",
            results["x"],
            inputs,
            options,
        )
        if check_if_composite(options, "positive"):
            _set_concentration_from_stoich(
                parameter_values,
                param,
                "positive",
                "secondary",
                results["x_2"],
                inputs,
                options,
            )
    elif options is not None and (
        check_if_composite(options, "positive")
        or check_if_composite(options, "negative")
    ):
        """
        Set the initial stoichiometry of each electrode, based on the initial SOC or
        voltage.
        """
        initial_stoichs = pybamm.lithium_ion.get_initial_stoichiometries_composite(
            initial_value,
            parameter_values,
            direction=direction,
            param=param,
            options=options,
            tol=tol,
            inputs=inputs,
            known_value=known_value,
        )
        _set_concentration_from_stoich(
            parameter_values,
            param,
            "positive",
            "primary",
            initial_stoichs["y_init_1"],
            inputs,
            options,
        )
        _set_concentration_from_stoich(
            parameter_values,
            param,
            "negative",
            "primary",
            initial_stoichs["x_init_1"],
            inputs,
            options,
        )
        if check_if_composite(options, "positive"):
            _set_concentration_from_stoich(
                parameter_values,
                param,
                "positive",
                "secondary",
                initial_stoichs["y_init_2"],
                inputs,
                options,
            )
        if check_if_composite(options, "negative"):
            _set_concentration_from_stoich(
                parameter_values,
                param,
                "negative",
                "secondary",
                initial_stoichs["x_init_2"],
                inputs,
                options,
            )
    else:
        """
        Set the initial stoichiometry of each electrode, based on the initial SOC or
        voltage.
        """
        x, y = pybamm.lithium_ion.get_initial_stoichiometries(
            initial_value,
            parameter_values,
            direction=direction,
            param=param,
            known_value=known_value,
            options=options,
            tol=tol,
            inputs=inputs,
        )
        _set_concentration_from_stoich(
            parameter_values, param, "negative", "primary", x, inputs, options
        )
        _set_concentration_from_stoich(
            parameter_values, param, "positive", "primary", y, inputs, options
        )

    return parameter_values


def _set_concentration_from_stoich(
    parameter_values, param, electrode, phase, stoich, inputs, options
):
    if electrode == "positive":
        electrode_param = param.p
    else:
        electrode_param = param.n
    if phase == "primary":
        phase_param = electrode_param.prim
    elif phase == "secondary":
        phase_param = electrode_param.sec
    else:
        raise ValueError(f"Invalid phase: {phase}")
    if check_if_composite(options, electrode):
        phase_prefactor = phase.capitalize() + ": "
    else:
        phase_prefactor = ""
    parameter_values.update(
        {
            f"{phase_prefactor}Initial concentration in {electrode} electrode [mol.m-3]": stoich
            * parameter_values.evaluate(phase_param.c_max, inputs=inputs)
        }
    )
