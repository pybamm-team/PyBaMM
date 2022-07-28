import pybamm
from .dfn import DFN


class Yang2017(DFN):
    def __init__(self, options=None, name="Yang2017", build=True):
        options = {
            "SEI": "ec reaction limited",
            "SEI film resistance": "distributed",
            "SEI porosity change": "true",
            "lithium plating": "irreversible",
            "lithium plating porosity change": "true",
        }
        super().__init__(options=options, name=name)
        pybamm.citations.register("Yang2017")

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues({
            "chemistry": "lithium_ion",
            "cell": "LGM50_Chen2020",
            "negative electrode": "graphite_Chen2020",
            "separator": "separator_Chen2020",
            "positive electrode": "nmc_Chen2020",
            "electrolyte": "lipf6_Nyman2008",
            "experiment": "1C_discharge_from_full_Chen2020",
            "sei": "example",
            "lithium plating": "okane2020_Li_plating",
        })
