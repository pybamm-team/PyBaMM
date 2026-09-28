import pybamm

from .dfn import DFN

_YANG2017_OPTIONS = {
    "SEI": ("ec reaction limited", "none"),
    "SEI film resistance": "distributed",
    "SEI porosity change": "true",
    "lithium plating": ("irreversible", "none"),
    "lithium plating porosity change": "true",
}


class Yang2017(DFN):
    def _model_default_options(self, supplied):
        for key, value in _YANG2017_OPTIONS.items():
            if key in supplied and supplied[key] != value:
                raise pybamm.OptionError(f"Yang2017 requires '{key}' to be {value!r}.")
        return dict(_YANG2017_OPTIONS)

    def __init__(self, options=None, name="Yang2017", build=True):
        super().__init__(options=options, name=name, build=build)
        pybamm.citations.register("Yang2017")
