
import pybamm
from .dfn import DFN

class Yang2017(DFN):
   def __init__(self):
        options = {"sei": "ec reaction limited", "sei film resistance": "distributed", "sei porosity change": "true", "lithium plating": "irreversible", "lithium plating porosity change": "true" }
        super().__init__(options=options, name="Yang2017")
        pybamm.citations.register("Yang2017")
   @property
   def default_parameter_values(self):
       return pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Yang2017)
   
 