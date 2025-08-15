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


def _get_lithiation_delithiation(direction, electrode, options):
    if direction is None or not _has_hysteresis(electrode, options):
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


def _has_hysteresis(electrode, options):
    hysteresis_options = [
        "current sigmoid",
        "one-state hysteresis",
        "one-state differential capacity hysteresis",
    ]
    if isinstance(options["open-circuit potential"], str):
        if options["open-circuit potential"] in hysteresis_options:
            return True
        else:
            return False
    elif isinstance(options["open-circuit potential"], tuple):
        if electrode == "negative":
            if options["open-circuit potential"][0] in hysteresis_options:
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
