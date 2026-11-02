import pybamm

from .dfn import DFN


class Yang2017(DFN):
    fixed_options = {
        "working electrode": "both",
        "SEI": ("ec reaction limited", "none"),
        "SEI film resistance": "distributed",
        "SEI porosity change": "true",
        "lithium plating": ("irreversible", "none"),
        "lithium plating porosity change": "true",
    }

    def __init__(self, options=None, name="Yang2017", build=True):
        super().__init__(options=options, name=name, build=build)
        pybamm.citations.register("Yang2017")
