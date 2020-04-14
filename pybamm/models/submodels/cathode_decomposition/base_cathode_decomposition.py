#
# Class for cathode decomposition in Li-ion batteries 
#
import pybamm
from scipy import constants 

class CathodeDecomposition(pybamm.BaseSubModel):
    """Base class for cathode decomposition in Li-ion batteries.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict, optional
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, reactions=None):
        super().__init__(param, reactions=reactions)

    def get_fundamental_variables(self):
        param = self.param
        k_b = pybamm.Scalar(constants.k) 
        T_av = pybamm.standard_variables.T_av
        alpha = pybamm.Variable("Degree of conversion of cathode decomposition")
        variables = {
            "Degree of conversion of cathode decomposition": alpha,
            "Cathode decomposition reaction rate": alpha*(1-alpha) * param.A_ca * pybamm.exp(-param.E_cn/(k_b*T_av))
        }
        return variables

    def set_rhs(self, variables):
        param = self.param
        k_b = pybamm.Scalar(constants.k) 
        T_av = pybamm.standard_variables.T_av
        alpha = variables["Degree of conversion of cathode decomposition"]
        self.rhs = {
            alpha: alpha*(1-alpha) * param.A_ca * pybamm.exp(-param.E_cn/(k_b*T_av))
        }