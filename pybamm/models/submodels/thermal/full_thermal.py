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
        T_av = pybamm.average(T)

        variables = self.get_standard_fundamental_variables(T, T_av)
        return variables

    def get_derived_variables(self, variables):
        variables.update(self.get_standard_derived_variables(variables))
        return variables

    def _flux_law(self, T):
        """Fourier's law for heat transfer"""
        q = -self.param.lambda_k * pybamm.grad(T)
        return q

