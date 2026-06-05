"""
Best-effort conversion of legacy BPX v0.x objects to the BPX v1.x schema.

BPX v1.0 added a top-level ``State`` block and moved the initial/ambient
temperature and initial electrolyte concentration out of ``Parameterisation``,
so v0.x files no longer validate. These helpers detect a v0.x object and repack
it into the v1.x layout so older files keep loading.

Adapted from the v0.x -> v1.x converter shared by @ejfdickinson in
https://github.com/pybamm-team/PyBaMM/issues/5571.
"""

from __future__ import annotations

import copy
import re

# Electrode prefixes for the synthesised hysteresis state.
_ELECTRODES = ("Negative", "Positive")


def _bpx_major_version(bpx_obj: dict) -> int:
    """
    Return the major version of a raw BPX object's ``Header.BPX`` field.

    The version may be encoded as a number (e.g. ``0.4`` or ``1.0`` in older
    files) or as a string (e.g. ``"0.4.0"`` or ``"1.1.0"``). Both forms are
    accepted; ``bpx`` itself coerces a float version to a string for backward
    compatibility, so a float does not by itself indicate a v0.x file.
    """
    try:
        version = bpx_obj["Header"]["BPX"]
    except (KeyError, TypeError) as err:
        raise ValueError(
            "Invalid BPX object: missing 'Header' -> 'BPX' version field."
        ) from err

    if isinstance(version, str):
        match = re.match(r"^\s*(\d+)", version)
        if match is None:
            raise ValueError(f"Invalid BPX version field: {version!r}.")
        return int(match.group(1))
    # bool is a subclass of int but is never a valid version
    if isinstance(version, (int, float)) and not isinstance(version, bool):
        return int(version)

    raise ValueError(f"Invalid BPX version field: {version!r}.")


def is_legacy_bpx(bpx_obj: dict) -> bool:
    """
    Return ``True`` if ``bpx_obj`` is a legacy BPX v0.x object (major version
    ``< 1``), and ``False`` for v1.x and later.

    Raises ``ValueError`` if the version field is missing or malformed.
    """
    return _bpx_major_version(bpx_obj) < 1


def convert_v0_to_v1(bpx_obj: dict) -> dict:
    """
    Return a new dict repacking a legacy BPX v0.x object into the v1.x schema
    (the input is not mutated).

    The v1.x ``State`` block is synthesised from the v0.x ``Parameterisation``
    entries, with placeholders for the required fields v0.x lacks: initial SOC
    set to ``1``, heat transfer coefficient and initial hysteresis state set to
    ``0`` (the lumped ``Thermal conductivity`` has no v1.x equivalent and is
    dropped), and ``Initial temperature`` falling back to the ambient (then
    reference) temperature when absent. Cross-version semantic changes are not
    adjusted.
    """
    params = copy.deepcopy(bpx_obj)
    parameterisation = params.get("Parameterisation", {})
    cell = parameterisation.get("Cell", {})
    electrolyte = parameterisation.get("Electrolyte", {})

    # Reference temperature stays in Cell under v1.x, so read it without popping.
    ambient_temperature = cell.pop("Ambient temperature [K]", None)
    reference_temperature = cell.get("Reference temperature [K]")
    initial_temperature = cell.pop("Initial temperature [K]", None)
    if initial_temperature is None:
        initial_temperature = (
            ambient_temperature
            if ambient_temperature is not None
            else reference_temperature
        )

    # Drop the deprecated lumped thermal conductivity (no v1.x equivalent).
    cell.pop("Thermal conductivity [W.m-1.K-1]", None)

    initial_conditions: dict = {"Initial state-of-charge": 1}
    if initial_temperature is not None:
        initial_conditions["Initial temperature [K]"] = initial_temperature

    initial_electrolyte_concentration = electrolyte.pop(
        "Initial concentration [mol.m-3]", None
    )
    if initial_electrolyte_concentration is not None:
        initial_conditions["Initial electrolyte concentration [mol.m-3]"] = (
            initial_electrolyte_concentration
        )

    # Placeholder hysteresis state; blended electrodes need a per-phase dict.
    for electrode in _ELECTRODES:
        electrode_params = parameterisation.get(f"{electrode} electrode", {})
        key = f"Initial hysteresis state: {electrode} electrode"
        if "Particle" in electrode_params:
            initial_conditions[key] = {
                phase: 0 for phase in electrode_params["Particle"]
            }
        else:
            initial_conditions[key] = 0

    thermal_environment: dict = {"Heat transfer coefficient [W.m-2.K-1]": 0}
    if ambient_temperature is not None:
        thermal_environment["Ambient temperature [K]"] = ambient_temperature

    params["State"] = {
        "Initial conditions": initial_conditions,
        "Thermal environment": thermal_environment,
    }

    params.setdefault("Header", {})["BPX"] = "1.0.0"

    return params
