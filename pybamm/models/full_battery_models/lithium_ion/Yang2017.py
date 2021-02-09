
import pybamm
from .dfn import DFN

class Yang2017(DFN):
   def __init__(self):
        options = options = {"sei": "ec reaction limited", "sei film resistance": "distributed", "sei porosity change": "true", "lithium plating": "irreversible"}
        super().__init__(self, options=options, name="Yang 2017")
  
   @property
   def default_parameter_values(self):
       return pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Yang2017)
