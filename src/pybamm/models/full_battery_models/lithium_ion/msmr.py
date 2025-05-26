import pybamm
import re
from .dfn import DFN


class MSMR(DFN):
    def __init__(self, options=None, name="MSMR", build=True):
        # Necessary/default options
        options = options or {}
        if "number of MSMR reactions" not in options:
            raise pybamm.OptionError(
                "number of MSMR reactions must be specified for MSMR"
            )
        if (
            "open-circuit potential" in options
            and options["open-circuit potential"] != "MSMR"
        ):
            raise pybamm.OptionError(
                "'open-circuit potential' must be 'MSMR' for MSMR not '{}'".format(
                    options["open-circuit potential"]
                )
            )
        elif "particle" in options and options["particle"] != "MSMR":
            raise pybamm.OptionError(
                "'particle' must be 'MSMR' for MSMR not '{}'".format(
                    options["particle"]
                )
            )
        elif (
            "intercalation kinetics" in options
            and options["intercalation kinetics"] != "MSMR"
        ):
            raise pybamm.OptionError(
                "'intercalation kinetics' must be 'MSMR' for MSMR not '{}'".format(
                    options["intercalation kinetics"]
                )
            )
        else:
            options.update(
                {
                    "open-circuit potential": "MSMR",
                    "particle": "MSMR",
                    "intercalation kinetics": "MSMR",
                }
            )
        super().__init__(options=options, name=name)

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues("MSMR_Example")


# Replace the deprecated MSMR parameter names with the new ones
# matches e.g. "X_p_3", "X_n_l_300", "Q_n_d_0", "U0_p_42", "a_n_5", "j0_ref_p_d_12", etc.
_VALID_NAME_RE = re.compile(
    r"^(?P<base>X|Q|w|U0|a|j0_ref)"  # base (now includes a and j0_ref)
    r"_(?P<elec>n|p)"  # electrode
    r"(?:_(?P<qual>[ld]))?"  # optional qualifier
    r"_(?P<idx>[0-9]+)$"  # non-negative integer index
)


def is_deprecated_msmr_name(key: str) -> bool:
    """
    Return True if `key` follows the (legacy) MSMR naming convention:
      BASE ∈ {X, Q, w, U0, a, j0_ref}
      electrode ∈ {n, p}
      optional qualifier ∈ {l, d}
      index ∈ non-negative integer
    """
    return bool(_VALID_NAME_RE.fullmatch(key))


_BASE_DESC = {
    "X": "host site occupancy fraction",
    "Q": "host site occupancy capacity",
    "w": "host site ideality factor",
    "U0": "host site standard potential",
    "a": "host site charge transfer coefficient",
    "j0_ref": "host site reference exchange-current density",
}

# only Q, U0, and j0_ref have units
_UNITS = {
    "Q": " [A.h]",
    "U0": " [V]",
    "j0_ref": " [A.m-2]",
}

_ELECTRODE = {
    "n": "Negative",
    "p": "Positive",
}

_QUALIFIER = {
    "l": "lithiation",
    "d": "delithiation",
}


def replace_deprecated_msmr_name(key: str) -> str:
    """
    Convert e.g. "X_n_d_3" →
        "Negative electrode host site occupancy fraction (delithiation) (3)"
    and likewise for a/U0/Q/j0_ref.
    """
    m = _VALID_NAME_RE.fullmatch(key)
    if not m:
        raise ValueError(f"Invalid MSMR name: {key!r}")

    base = m.group("base")
    elec = m.group("elec")
    qual = m.group("qual")
    idx = m.group("idx")

    # start constructing the description
    desc = f"{_ELECTRODE[elec]} electrode {_BASE_DESC[base]}"
    if qual:
        desc += f" ({_QUALIFIER[qual]})"
    desc += f" ({idx})"

    # tack on units if this base has them
    if base in _UNITS:
        desc += _UNITS[base]

    return desc
