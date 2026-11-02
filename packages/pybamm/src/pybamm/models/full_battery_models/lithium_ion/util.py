import pybamm


def check_if_composite(options, electrode):
    if not isinstance(options, pybamm.BatteryModelOptions):
        options = pybamm.BatteryModelOptions(options)
    domain_options = getattr(options, electrode)
    particle_phases = domain_options["particle phases"]
    return particle_phases == "2"


def _initial_state_variant(options):
    """Name the electrode SOH formulation that sets the initial state."""
    if options.get("open-circuit potential") == "MSMR":
        return "msmr"
    if options.get("working electrode") == "positive":
        return "half-cell"
    if check_if_composite(options, "positive") or check_if_composite(
        options, "negative"
    ):
        return "composite"
    return "standard"


def _initial_state_parameters(param, options):
    """
    The parameters that set the initial state, each with the name of the solved
    value that sets it and the symbol that value is scaled by.
    """
    variant = _initial_state_variant(options)
    if variant == "msmr":
        return {
            "Initial voltage in negative electrode [V]": ("Un", pybamm.Scalar(1)),
            "Initial voltage in positive electrode [V]": ("Up", pybamm.Scalar(1)),
        }
    if variant == "half-cell":
        stoichiometries = {"positive": ["x", "x_2"]}
    elif variant == "composite":
        stoichiometries = {
            "negative": ["x_init_1", "x_init_2"],
            "positive": ["y_init_1", "y_init_2"],
        }
    else:
        stoichiometries = {"negative": ["x"], "positive": ["y"]}
    parameters = {}
    for electrode, keys in stoichiometries.items():
        electrode_param = param.domain_params[electrode]
        if check_if_composite(options, electrode):
            phases = [
                ("Primary: ", electrode_param.prim, keys[0]),
                ("Secondary: ", electrode_param.sec, keys[1]),
            ]
        else:
            phases = [("", electrode_param.prim, keys[0])]
        for prefix, phase_param, key in phases:
            name = f"{prefix}Initial concentration in {electrode} electrode [mol.m-3]"
            parameters[name] = (key, phase_param.c_max)
    return parameters


def _has_hysteresis(electrode, options, phase=None):
    if not isinstance(options, pybamm.BatteryModelOptions):
        options = pybamm.BatteryModelOptions(options)
    hysteresis_options = [
        "current sigmoid",
        "one-state hysteresis",
        "one-state differential capacity hysteresis",
        # Also catch old names
        "Axen",
        "Wycisk",
    ]
    domain_options = getattr(options, electrode)
    if isinstance(domain_options["open-circuit potential"], str):
        return domain_options["open-circuit potential"] in hysteresis_options
    elif isinstance(domain_options["open-circuit potential"], tuple):
        ocp_option = domain_options["open-circuit potential"]
        if phase == "primary":
            return ocp_option[0] in hysteresis_options
        elif phase == "secondary":
            return ocp_option[1] in hysteresis_options
        else:
            return any(
                isinstance(item, str) and item in hysteresis_options
                for item in ocp_option
            )
    else:
        ocp_opt = domain_options["open-circuit potential"]
        raise ValueError(f"Invalid open-circuit potential option: {ocp_opt}")


def get_lithiation_delithiation(direction, electrode, options, phase=None):
    """
    Get the lithiation/delithiation direction for OCP evaluation.

    Parameters
    ----------
    direction : str or None
        The cell-level direction: "charge", "discharge", or None (equilibrium).
    electrode : str
        "negative" or "positive"
    options : dict or BatteryModelOptions
        Model options containing OCP settings
    phase : str, optional
        "primary" or "secondary" for composite electrodes

    Returns
    -------
    str or None
        "lithiation", "delithiation", or None (for equilibrium/no hysteresis)
    """
    if direction is None or not _has_hysteresis(electrode, options, phase):
        return None
    elif (direction == "charge" and electrode == "negative") or (
        direction == "discharge" and electrode == "positive"
    ):
        return "lithiation"
    elif (direction == "discharge" and electrode == "negative") or (
        direction == "charge" and electrode == "positive"
    ):
        return "delithiation"
    else:
        raise ValueError()


def _initial_hysteresis_state(phase_param, electrode):
    """Initial hysteresis state of an electrode phase at the electrode midpoint."""
    L = phase_param.domain_param.L
    x_mid = L / 2 if electrode == "negative" else phase_param.main_param.L_x - L / 2
    return pybamm.FunctionParameter(
        f"{phase_param.phase_prefactor}Initial hysteresis state in {electrode} "
        "electrode",
        {"Through-cell distance (x) [m]": x_mid},
    )


def get_initial_ocp(phase_param, sto, T, direction, electrode, options, phase=None):
    """
    Open-circuit potential [V] an electrode phase starts on.

    With a direction, or without hysteresis, this is the OCP branch given by
    :func:`get_lithiation_delithiation`. For a hysteresis phase with no direction,
    it is the OCP at the initial hysteresis state :math:`h_0`,
    :math:`(1 + h_0) / 2 \\, U_{delith} + (1 - h_0) / 2 \\, U_{lith}`, with
    :math:`h_0` evaluated at the electrode midpoint.

    Parameters
    ----------
    phase_param : :class:`pybamm.LithiumIonParameters` phase
        The phase parameters, e.g. ``param.n.prim``
    sto : :class:`pybamm.Symbol`
        The stoichiometry
    T : :class:`pybamm.Symbol`
        The temperature [K]
    direction : str or None
        The cell-level direction: "charge", "discharge", or None.
    electrode : str
        "negative" or "positive"
    options : dict or BatteryModelOptions
        Model options containing OCP settings
    phase : str, optional
        "primary" or "secondary" for composite electrodes
    """
    if direction is not None or not _has_hysteresis(electrode, options, phase):
        lithiation = get_lithiation_delithiation(direction, electrode, options, phase)
        return phase_param.U(sto, T, lithiation)
    h0 = _initial_hysteresis_state(phase_param, electrode)
    U_lith = phase_param.U(sto, T, "lithiation")
    U_delith = phase_param.U(sto, T, "delithiation")
    return (1 + h0) / 2 * U_delith + (1 - h0) / 2 * U_lith


def get_equilibrium_direction(soc_state, electrode, options, phase=None):
    """
    Get the appropriate cell direction for equilibrium stoichiometry calculation.

    For electrodes with hysteresis, equilibrium stoichiometries should be calculated
    on the OCP branch corresponding to how the cell reached that SOC:
    - 100% SOC: reached via charging → use "charge" direction
    - 0% SOC: reached via discharging → use "discharge" direction

    For electrodes without hysteresis, returns None (use equilibrium OCP).

    Parameters
    ----------
    soc_state : str
        "100" for 100% SOC or "0" for 0% SOC
    electrode : str
        "negative" or "positive"
    options : dict or BatteryModelOptions
        Model options containing OCP settings
    phase : str, optional
        "primary" or "secondary" for composite electrodes

    Returns
    -------
    str or None
        "charge" (for 100% SOC with hysteresis), "discharge" (for 0% SOC with
        hysteresis), or None (no hysteresis)
    """
    if not _has_hysteresis(electrode, options, phase):
        return None

    if soc_state == "100":
        # 100% SOC is reached by charging
        return "charge"
    elif soc_state == "0":
        # 0% SOC is reached by discharging
        return "discharge"
    else:
        raise ValueError(f"soc_state must be '100' or '0', got '{soc_state}'")
