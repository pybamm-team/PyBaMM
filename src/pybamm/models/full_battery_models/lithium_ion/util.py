def check_if_composite(options, electrode):
    options = options or {}
    particle_phases = options.get("particle phases", None)
    if particle_phases is None:
        return False
    if particle_phases == "2":
        return True
    elif isinstance(particle_phases, tuple) and particle_phases[0] == "2":
        if electrode == "positive":
            return False
        else:
            return True
    elif isinstance(particle_phases, tuple) and particle_phases[1] == "2":
        if electrode == "positive":
            return True
        else:
            return False
    else:
        return False
