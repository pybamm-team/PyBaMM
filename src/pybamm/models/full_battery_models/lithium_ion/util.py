import pybamm


def check_if_composite(options, electrode):
    if not isinstance(options, pybamm.BatteryModelOptions):
        options = pybamm.BatteryModelOptions(options)
    domain_options = getattr(options, electrode)
    particle_phases = domain_options["particle phases"]
    if particle_phases == "2":
        return True
    else:
        return False


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
