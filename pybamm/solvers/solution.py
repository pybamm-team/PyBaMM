#
# Solution interface class
#
import pybamm


class Solution(pybamm.SolutionBase):
    def __new__(cls, *args, **kwargs):
        if kwargs.get("output_variables", []):
            # Solution contains the full state vector 'y'
            return pybamm.SolutionVars(*args, **kwargs)
        else:
            # Solution contains only the requested variables
            return pybamm.SolutionFull(*args, **kwargs)
