#
# Class for graphite anode decomposition in Li-ion batteries 
#
import pybamm
from scipy import constants 

class AnodeDecomposition(pybamm.BaseSubModel):
    """Base class for graphite anode decomposition in Li-ion batteries.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict, optional
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        param = self.param
        k_b = pybamm.Scalar(constants.k) 
        T_av = pybamm.standard_variables.T_av
        c_s_n = pybamm.standard_variables.c_s_n
        x_an = c_s_n /param.c_n_max
        z = pybamm.Variable("Relative SEI thickness")

        variables = {
            "Relative SEI thickness": z,
            "Anode decomposition reaction rate": -param.A_an * x_an * pybamm.exp(-param.E_an/(k_b*T_av)) * pybamm.exp(-z/param.z_0),
        }
        return variables

    def set_rhs(self, variables):
        param = self.param
        k_b = pybamm.Scalar(constants.k) 
        T_av = pybamm.standard_variables.T_av
        c_s_n = pybamm.standard_variables.c_s_n
        x_an = c_s_n /param.c_n_max
        z = variables["Relative SEI thickness"]

        self.rhs = {
            z: param.A_an * x_an * pybamm.exp(-param.E_an/(k_b*T_av)) * pybamm.exp(-z/param.z_0), 
        }
        
    def set_initial_conditions(self, variables):
        z = variables["Relative SEI thickness"]
        self.initial_conditions = {z: self.param.z_0}