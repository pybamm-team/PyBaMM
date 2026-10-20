"""Options specific to the dual-continuum (DC) model."""

from __future__ import annotations

import pybamm

#: Default number of DC1t exponential modes.
DEFAULT_MODES = 8


class DCModelOptions(pybamm.FuzzyDict):
    """Dual-continuum options; the first value of each list is the default.

    The options are "model type" ("DC1", "DC1t", "DC0", "Yang"), "transient
    closure modes" (number of DC1t modes, default 8), "calculate surface
    concentration a priori" (galvanostatic only), "closure variable"
    ("dimensional", "dimensionless", "isolated sphere"), "surface area" ("from
    image", "spherical") and "lithium foil surface porosity" (half cells). The
    model's README describes each one.
    """

    possible_options = {
        "model type": ["DC1", "DC1t", "DC0", "Yang"],
        "calculate surface concentration a priori": ["false", "true"],
        "closure variable": ["dimensional", "dimensionless", "isolated sphere"],
        "surface area": ["from image", "spherical"],
        "lithium foil surface porosity": ["false", "true"],
    }

    def __init__(self, extra_options=None):
        extra_options = dict(extra_options or {})
        options = {k: v[0] for k, v in self.possible_options.items()}
        options["transient closure modes"] = DEFAULT_MODES

        for key, value in extra_options.items():
            if key == "transient closure modes":
                options[key] = self._number_of_modes(value)
                continue
            if key not in self.possible_options:
                best = pybamm.FuzzyDict([*self.possible_options, "transient closure modes"]).get_best_matches(key)
                hint = (
                    f"Best matches are {best}"
                    if best
                    else f"Possible options are {list(self.possible_options)}"
                )
                raise pybamm.OptionError(
                    f"DC option '{key}' not recognised. {hint}"
                )
            if isinstance(value, bool):
                value = "true" if value else "false"
            if value not in self.possible_options[key]:
                best = pybamm.FuzzyDict(self.possible_options[key]).get_best_matches(value)
                hint = (
                    f"Best matches are {best}"
                    if best
                    else f"Possible options are {list(self.possible_options)}"
                )
                raise pybamm.OptionError(
                    f"Invalid value '{value}' for DC option '{key}'. {hint}"
                )
            options[key] = value

        if (
            options["model type"] == "Yang"
            and options["calculate surface concentration a priori"] == "true"
        ):
            raise pybamm.OptionError(
                "Cannot calculate the surface concentration a priori for Yang's model"
            )

        if (
            "transient closure modes" in extra_options
            and options["model type"] != "DC1t"
        ):
            raise pybamm.OptionError(
                "'transient closure modes' only applies to model type 'DC1t'"
            )

        super().__init__(options)

    @staticmethod
    def _number_of_modes(value):
        """Validate the number of transient closure modes."""
        if isinstance(value, bool) or not str(value).strip().isdigit():
            raise pybamm.OptionError(
                f"Invalid value '{value}' for DC option 'transient closure modes'. "
                "It must be a positive integer"
            )
        number = int(str(value).strip())
        if number < 1:
            raise pybamm.OptionError(
                "DC option 'transient closure modes' must be at least 1"
            )
        return number

    @property
    def implicit_surface_concentration(self):
        """Whether c_surf is an extra algebraic unknown."""
        if self["model type"] == "DC0":
            return False
        if self["model type"] == "Yang":
            return True
        return self["calculate surface concentration a priori"] == "false"
