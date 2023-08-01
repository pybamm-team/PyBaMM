#
# Class for cathode decomposition in Li-ion batteries 
#
import pybamm
from scipy import constants 

class ThreeStateCathodeDecomposition(pybamm.BaseSubModel):
    """Base class for cathode decomposition in Li-ion batteries.

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
        alpha = pybamm.Variable("Degree of conversion of cathode decomposition", domain="current collector")
        alpha_core = pybamm.Variable("Degree of conversion of cathode decomposition in the core section", domain="current collector")
        alpha_mid = pybamm.Variable("Degree of conversion of cathode decomposition in the middle section", domain="current collector")
        alpha_outer = pybamm.Variable("Degree of conversion of cathode decomposition in the outer section", domain="current collector")
        
        variables = {
            "Degree of conversion of cathode decomposition": alpha,
            "Degree of conversion of cathode decomposition in the core section": alpha_core,
            "Degree of conversion of cathode decomposition in the middle section": alpha_mid,
            "Degree of conversion of cathode decomposition in the outer section": alpha_outer,}
        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        k_b = pybamm.Scalar(constants.k) 
        T_av = variables["X-averaged positive electrode temperature"]
        T_av_dim = param.Delta_T * T_av + param.T_ref

        T_outer = param.Delta_T * variables["Outer cell temperature"] + param.T_ref
        T_mid = param.Delta_T * variables["Middle cell temperature"] + param.T_ref
        T_core = param.Delta_T * variables["Core cell temperature"] + param.T_ref
        
        alpha = variables["Degree of conversion of cathode decomposition"]
        alpha_core = variables["Degree of conversion of cathode decomposition in the core section"]
        alpha_mid = variables["Degree of conversion of cathode decomposition in the middle section"]
        alpha_outer = variables["Degree of conversion of cathode decomposition in the outer section"]

        rho_p_dim = pybamm.Parameter("Positive electrode density [kg.m-3]")

        # r_ca_dim = (
        #     alpha
        #     * (1-alpha) 
        #     * param.therm.A_ca 
        #     * pybamm.exp(-param.therm.E_ca/(k_b*T_av_dim))
        # ) # units 1/s
        def r_ca_dim(alpha, T): 
            return alpha* (1-alpha)* param.therm.A_ca* pybamm.exp(-param.therm.E_ca/(k_b*T))

        def Q_exo_ca (alpha, T):
            return rho_p_dim * param.therm.h_ca * r_ca_dim(alpha, T) / Q_scale
        
        r_ca_dim_av = r_ca_dim(alpha, T_av_dim)
        r_ca_dim_core = r_ca_dim(alpha_core, T_core)
        r_ca_dim_mid = r_ca_dim(alpha_mid, T_mid)
        r_ca_dim_outer = r_ca_dim(alpha_outer, T_outer)

        gamma_core = self.param.therm.gamma_core
        gamma_mid = self.param.therm.gamma_mid
        gamma_outer = self.param.therm.gamma_outer

        Q_scale = param.i_typ * param.potential_scale / param.L_x
        Q_exo_ca_av = Q_exo_ca(alpha, T_av_dim)
        Q_exo_ca_core = Q_exo_ca(alpha_core, T_core)#*gamma_core
        Q_exo_ca_mid = Q_exo_ca(alpha_mid, T_mid)#*gamma_mid
        Q_exo_ca_outer = Q_exo_ca(alpha_outer, T_outer)#*gamma_outer

        variables = {
            "Cathode decomposition reaction rate [s-1]": r_ca_dim_av,
            "Cathode decomposition reaction rate": r_ca_dim_av * param.timescale,
            "Cathode decomposition heating": Q_exo_ca_av,
            "Cathode decomposition heating [W.m-3]": Q_exo_ca_av * Q_scale,

            "Core section cathode decomposition reaction rate [s-1]": r_ca_dim_core,
            "Core section cathode decomposition reaction rate": r_ca_dim_core * param.timescale,
            "Core section cathode decomposition heating": Q_exo_ca_core,
            "Core section cathode decomposition heating [W.m-3]": Q_exo_ca_core * Q_scale,

            "Middle section cathode decomposition reaction rate [s-1]": r_ca_dim_mid,
            "Middle section cathode decomposition reaction rate": r_ca_dim_mid * param.timescale,
            "Middle section cathode decomposition heating": Q_exo_ca_mid,
            "Middle section cathode decomposition heating [W.m-3]": Q_exo_ca_mid * Q_scale,

            "Outer section cathode decomposition reaction rate [s-1]": r_ca_dim_outer,
            "Outer section cathode decomposition reaction rate": r_ca_dim_outer * param.timescale,
            "Outer section cathode decomposition heating": Q_exo_ca_outer,
            "Outer section cathode decomposition heating [W.m-3]": Q_exo_ca_outer * Q_scale,
        }


         
        # units 1/s
        return variables

    def set_rhs(self, variables):
        decomp_rate = variables["Cathode decomposition reaction rate"]
        decomp_rate_core = variables["Core section cathode decomposition reaction rate"]
        decomp_rate_mid = variables["Middle section cathode decomposition reaction rate"]
        decomp_rate_outer = variables["Outer section cathode decomposition reaction rate"]

        alpha = variables["Degree of conversion of cathode decomposition"]
        alpha_core = variables["Degree of conversion of cathode decomposition in the core section"]
        alpha_mid = variables["Degree of conversion of cathode decomposition in the middle section"]
        alpha_outer = variables["Degree of conversion of cathode decomposition in the outer section"]
        self.rhs = {
            alpha: decomp_rate,
            alpha_core: decomp_rate_core,
            alpha_mid: decomp_rate_mid,
            alpha_outer: decomp_rate_outer,
            }

    def set_initial_conditions(self, variables):
        alpha = variables["Degree of conversion of cathode decomposition"]
        alpha_core = variables["Degree of conversion of cathode decomposition in the core section"]
        alpha_mid = variables["Degree of conversion of cathode decomposition in the middle section"]
        alpha_outer = variables["Degree of conversion of cathode decomposition in the outer section"]
        
        self.initial_conditions = {
            alpha: self.param.therm.alpha_0,
            alpha_core: self.param.therm.alpha_0,
            alpha_mid: self.param.therm.alpha_0,
            alpha_outer: self.param.therm.alpha_0
            }