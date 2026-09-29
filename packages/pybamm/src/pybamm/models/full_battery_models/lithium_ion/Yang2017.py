import pybamm
from pybamm.models.full_battery_models.base_battery_model import option_values_match

from .dfn import DFN

_YANG2017_OPTIONS = {
    "SEI": ("ec reaction limited", "none"),
    "SEI film resistance": "distributed",
    "SEI porosity change": "true",
    "lithium plating": ("irreversible", "none"),
    "lithium plating porosity change": "true",
    "total interfacial current density as a state": "true",
}


class Yang2017(DFN):
    def _model_default_options(self, supplied):
        working_electrode = supplied.get("working electrode", "both")
        if working_electrode != "both":
            raise pybamm.OptionError(
                "Yang2017 requires 'working electrode' to be 'both'."
            )
        for key, value in _YANG2017_OPTIONS.items():
            if key in supplied and not option_values_match(
                key, supplied[key], value, working_electrode
            ):
                raise pybamm.OptionError(f"Yang2017 requires '{key}' to be {value!r}.")
        return dict(_YANG2017_OPTIONS)

    def __init__(self, options=None, name="Yang2017", build=True):
        super().__init__(options=options, name=name, build=build)
        pybamm.citations.register("Yang2017")
