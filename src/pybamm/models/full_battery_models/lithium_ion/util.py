def check_if_composite(options, electrode):
    options = options or {}
    particle_phases = options.get("particle phases", None)
    if particle_phases is None:
        return False
    if particle_phases == "2":
        return True
    if isinstance(particle_phases, tuple) and particle_phases[0] == "2":
        if electrode == "negative":
            return True
    if isinstance(particle_phases, tuple) and particle_phases[1] == "2":
        if electrode == "positive":
            return True
    return False


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
        raise ValueError


def _has_hysteresis(electrode, options, phase=None):
    hysteresis_options = [
        "current sigmoid",
        "one-state hysteresis",
        "one-state differential capacity hysteresis",
    ]
    phase = phase or ""
    if options.get("open-circuit potential") is None:
        return False
    if isinstance(options["open-circuit potential"], str):
        if options["open-circuit potential"] in hysteresis_options:
            return True
        else:
            return False
    elif isinstance(options["open-circuit potential"], tuple):
        if electrode == "negative":
            my_ocp_options = options["open-circuit potential"][0]

            if check_if_composite(options, electrode) and isinstance(
                my_ocp_options, tuple
            ):
                if phase == "primary":
                    if my_ocp_options[0] in hysteresis_options:
                        return True
                    else:
                        return False
                elif phase == "secondary":
                    if my_ocp_options[1] in hysteresis_options:
                        return True
                    else:
                        return False
                else:
                    raise ValueError
            else:
                if my_ocp_options in hysteresis_options:
                    return True
                else:
                    return False

        elif electrode == "positive":
            if options["open-circuit potential"][1] in hysteresis_options:
                return True
            else:
                return False
    else:
        raise ValueError
