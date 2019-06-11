#
# Class for full thermal submode
#
import pybamm


class ThermalFull(pybamm.BaseThermal):
    """Class for full thermal submodel

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseThermal`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        T = pybamm.standard_variables.T

        variables = {"Cell temperature": T}

        return variables

    def get_derived_variables(self):

        
