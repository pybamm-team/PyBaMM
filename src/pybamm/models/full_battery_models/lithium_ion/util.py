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
        raise ValueError(
            f"Invalid open-circuit potential option: {domain_options['open-circuit potential']}"
        )


def _get_lithiation_delithiation(direction, electrode, options, phase=None):
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
