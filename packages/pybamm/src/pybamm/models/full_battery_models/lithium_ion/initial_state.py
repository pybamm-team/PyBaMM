import pybamm

from .initial_state_linearisation import get_linearised_initial_state
from .util import _has_hysteresis, _initial_hysteresis_state, check_if_composite


def _set_hysteresis_branch(parameter_values, electrode, direction, options, phase=None):
    phase = phase or ""
    if phase != "":
        phase_prefactor = phase.capitalize() + ": "
    else:
        phase_prefactor = ""
    name = f"{phase_prefactor}Initial hysteresis state in {electrode} electrode"
    # Values written by earlier calls with a direction, so a later call without one
    # resets them instead of keeping them as if the user had set them
    from_direction = parameter_values._hysteresis_states_from_direction
    if direction is None:
        if name in parameter_values and parameter_values[name] is not (
            from_direction.pop(name, None)
        ):
            # Keep the user's initial state; the initial stoichiometries are
            # solved on the OCP at that state (see `get_initial_ocp`)
            return parameter_values
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
        # A new float object, so a value the user sets later is never this one
        initial_hysteresis_branch = float(initial_hysteresis_branch)
        from_direction[name] = initial_hysteresis_branch
    parameter_values.update({name: initial_hysteresis_branch})
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
    esoh_solver=None,
    differentiable=False,
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
    direction : str, optional
        "charge" or "discharge" starts each hysteresis electrode on that OCP branch.
        If None, a voltage is solved on the OCP at the initial hysteresis state (0
        if unset). A spatially varying initial hysteresis state is evaluated at the
        electrode midpoint, so the initial voltage is then approximate.
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
    esoh_solver : optional
        A pre-built electrode SOH solver or simulation to reuse.
    differentiable : bool, optional
        If True, set the initial states as expressions in the input parameters that
        match the solution, and its first derivative with respect to the inputs, at
        ``inputs``. Required for forward sensitivities with respect to inputs that
        the initial state depends on. Default is False.
    """
    options = options or {}
    original_parameter_values = parameter_values
    parameter_values = parameter_values if inplace else parameter_values.copy()
    param = param or pybamm.LithiumIonParameters(options)

    has_hysteresis = False
    for electrode in ["negative", "positive"]:
        if check_if_composite(options, electrode):
            for phase in ["primary", "secondary"]:
                if _has_hysteresis(electrode, options, phase):
                    has_hysteresis = True
                    parameter_values = _set_hysteresis_branch(
                        parameter_values, electrode, direction, options, phase
                    )
        else:
            if _has_hysteresis(electrode, options):
                has_hysteresis = True
                parameter_values = _set_hysteresis_branch(
                    parameter_values, electrode, direction, options
                )

    if differentiable:
        # The hysteresis states set above enter a voltage start's OCP; otherwise the
        # original values keep their processing cache across calls
        initial_state = get_linearised_initial_state(
            initial_value,
            parameter_values if has_hysteresis else original_parameter_values,
            param,
            options,
            direction=direction,
            known_value=known_value,
            tol=tol,
            inputs=inputs,
            esoh_solver=esoh_solver,
        )

    def set_concentration(electrode, phase, stoich):
        _set_concentration_from_stoich(
            parameter_values,
            param,
            electrode,
            phase,
            stoich,
            inputs,
            options,
            differentiable=differentiable,
        )

    if options is not None and options.get("open-circuit potential", None) == "MSMR":
        """
        Set the initial OCP of each electrode, based on the initial SOC or voltage.
        """
        if differentiable:
            Un, Up = initial_state["Un"], initial_state["Up"]
        else:
            Un, Up = pybamm.lithium_ion.get_initial_ocps(
                initial_value,
                parameter_values,
                direction=direction,
                param=param,
                known_value=known_value,
                options=options,
                tol=tol,
                inputs=inputs,
                esoh_solver=esoh_solver,
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
        if differentiable:
            results = initial_state
        else:
            results = pybamm.lithium_ion.get_initial_stoichiometry_half_cell(
                initial_value,
                parameter_values,
                param=param,
                options=options,
                tol=tol,
                inputs=inputs,
                direction=direction,
                esoh_sim=esoh_solver,
            )
        set_concentration("positive", "primary", results["x"])
        if check_if_composite(options, "positive"):
            set_concentration("positive", "secondary", results["x_2"])
    elif options is not None and (
        check_if_composite(options, "positive")
        or check_if_composite(options, "negative")
    ):
        """
        Set the initial stoichiometry of each electrode, based on the initial SOC or
        voltage.
        """
        if differentiable:
            initial_stoichs = initial_state
        else:
            initial_stoichs = pybamm.lithium_ion.get_initial_stoichiometries_composite(
                initial_value,
                parameter_values,
                direction=direction,
                param=param,
                options=options,
                tol=tol,
                inputs=inputs,
                known_value=known_value,
                esoh_sim=esoh_solver,
            )
        set_concentration("positive", "primary", initial_stoichs["y_init_1"])
        set_concentration("negative", "primary", initial_stoichs["x_init_1"])
        if check_if_composite(options, "positive"):
            set_concentration("positive", "secondary", initial_stoichs["y_init_2"])
        if check_if_composite(options, "negative"):
            set_concentration("negative", "secondary", initial_stoichs["x_init_2"])
    else:
        """
        Set the initial stoichiometry of each electrode, based on the initial SOC or
        voltage.
        """
        if differentiable:
            x, y = initial_state["x"], initial_state["y"]
        else:
            x, y = pybamm.lithium_ion.get_initial_stoichiometries(
                initial_value,
                parameter_values,
                direction=direction,
                param=param,
                known_value=known_value,
                options=options,
                tol=tol,
                inputs=inputs,
                esoh_solver=esoh_solver,
            )
        set_concentration("negative", "primary", x)
        set_concentration("positive", "primary", y)

    return parameter_values


def _set_concentration_from_stoich(
    parameter_values,
    param,
    electrode,
    phase,
    stoich,
    inputs,
    options,
    differentiable=False,
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
    if differentiable:
        concentration = stoich * parameter_values.process_symbol(phase_param.c_max)
        if not concentration.has_symbol_of_classes(pybamm.InputParameter):
            concentration = float(concentration.evaluate())
    else:
        concentration = stoich * parameter_values.evaluate(
            phase_param.c_max, inputs=inputs
        )
    parameter_values.update(
        {
            f"{phase_prefactor}Initial concentration in {electrode} electrode [mol.m-3]": concentration
        }
    )


def _fingerprint_standard(pv, param, inputs):
    return (
        "standard",
        float(pv.evaluate(param.n.Q_init, inputs=inputs)),
        float(pv.evaluate(param.p.Q_init, inputs=inputs)),
        float(pv.evaluate(param.Q_Li_particles_init, inputs=inputs)),
        float(pv.evaluate(param.ocp_soc_100, inputs=inputs)),
        float(pv.evaluate(param.ocp_soc_0, inputs=inputs)),
    )


def _fingerprint_half_cell(pv, param, options, inputs):
    parts = [
        "half_cell",
        float(pv.evaluate(param.p.prim.Q_init, inputs=inputs)),
        float(pv.evaluate(param.ocp_soc_100, inputs=inputs)),
        float(pv.evaluate(param.ocp_soc_0, inputs=inputs)),
    ]
    if check_if_composite(options, "positive"):
        parts.append(float(pv.evaluate(param.p.sec.Q_init, inputs=inputs)))
    return tuple(parts)


def _fingerprint_composite(pv, param, options, inputs):
    parts = [
        "composite",
        float(pv.evaluate(param.n.prim.Q_init, inputs=inputs)),
        float(pv.evaluate(param.p.prim.Q_init, inputs=inputs)),
        float(pv.evaluate(param.Q_Li_particles_init, inputs=inputs)),
        float(pv.evaluate(param.ocp_soc_100, inputs=inputs)),
        float(pv.evaluate(param.ocp_soc_0, inputs=inputs)),
    ]
    if check_if_composite(options, "negative"):
        parts.append(float(pv.evaluate(param.n.sec.Q_init, inputs=inputs)))
    if check_if_composite(options, "positive"):
        parts.append(float(pv.evaluate(param.p.sec.Q_init, inputs=inputs)))
    return tuple(parts)


def _fingerprint_hysteresis(pv, param, options, inputs):
    """Initial hysteresis states that set the initial OCP when no direction is given."""
    parts = []
    if options.get("working electrode") == "positive":
        electrodes = ["positive"]
    else:
        electrodes = ["negative", "positive"]
    for electrode in electrodes:
        domain_param = param.domain_params[electrode]
        phases = ["primary"]
        if check_if_composite(options, electrode):
            phases.append("secondary")
        for phase in phases:
            phase_param = domain_param.phase_params[phase]
            name = (
                f"{phase_param.phase_prefactor}Initial hysteresis state in "
                f"{electrode} electrode"
            )
            if _has_hysteresis(electrode, options, phase) and name in pv:
                h0 = _initial_hysteresis_state(phase_param, electrode)
                parts.append(float(pv.evaluate(h0, inputs=inputs)))
    return tuple(parts)


def compute_esoh_fingerprint(parameter_values, param, options, inputs):
    """Evaluate the quantities that determine the eSOH result for this model type.

    Routes to a model-specific helper based on the model options, mirroring
    the branching in :func:`set_initial_state`.

    Parameters
    ----------
    parameter_values : :class:`pybamm.ParameterValues`
        The (unprocessed) parameter values.
    param : :class:`pybamm.LithiumIonParameters`
        The symbolic parameter set from the model.
    options : dict-like
        Model options (e.g. from ``model.options``).
    inputs : dict or None
        Runtime input parameters.

    Returns
    -------
    tuple
        Hashable tuple of evaluated scalar quantities.
    """
    options = options or {}
    inputs = inputs or {}

    if options.get("open-circuit potential") == "MSMR":
        return _fingerprint_standard(parameter_values, param, inputs)
    elif options.get("working electrode") == "positive":
        fingerprint = _fingerprint_half_cell(parameter_values, param, options, inputs)
    elif check_if_composite(options, "positive") or check_if_composite(
        options, "negative"
    ):
        fingerprint = _fingerprint_composite(parameter_values, param, options, inputs)
    else:
        fingerprint = _fingerprint_standard(parameter_values, param, inputs)
    return fingerprint + _fingerprint_hysteresis(
        parameter_values, param, options, inputs
    )
