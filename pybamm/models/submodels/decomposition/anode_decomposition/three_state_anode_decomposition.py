#
# Class for graphite anode decomposition in Li-ion batteries
#
import pybamm
from scipy import constants


class ThreeStateAnodeDecomposition(pybamm.BaseSubModel):
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
        z = pybamm.Variable("Relative SEI thickness", domain="current collector")
        z_core = pybamm.Variable("Relative SEI thickness in the core section", domain="current collector")
        z_mid = pybamm.Variable("Relative SEI thickness in the middle section", domain="current collector")
        z_outer = pybamm.Variable("Relative SEI thickness in the outer section", domain="current collector")
 
        variables = {
            "Relative SEI thickness": z,
            "Relative SEI thickness in the core section": z_core,
            "Relative SEI thickness in the middle section": z_mid,
            "Relative SEI thickness in the outer section": z_outer,
        }
        return variables

    def get_coupled_variables(self, variables):
        
        param = self.param
        k_b = pybamm.Scalar(constants.k)
        c_s_n_surf = variables["X-averaged negative particle surface concentration"]
        x_an = c_s_n_surf
        rho_n_dim = pybamm.Parameter("Negative electrode density [kg.m-3]")

        T_av = variables["X-averaged negative electrode temperature"]
        T_av_dim = param.Delta_T * T_av + param.T_ref
        T_outer = param.Delta_T * variables["Outer cell temperature"] + param.T_ref
        T_mid = param.Delta_T * variables["Middle cell temperature"] + param.T_ref
        T_core = param.Delta_T * variables["Core cell temperature"] + param.T_ref
        
        z = variables["Relative SEI thickness"]
        z_core = variables["Relative SEI thickness in the core section"]
        z_mid = variables["Relative SEI thickness in the middle section"]
        z_outer = variables["Relative SEI thickness in the outer section"]

        def r_an_dim(z, x_an, T):
            return -param.therm.A_an * x_an * pybamm.exp(-param.therm.E_an / (k_b * T)) * pybamm.exp(-z / param.therm.z_0)  # units 1/s
        
        def Q_exo_an(z, x_an, T):
            return -rho_n_dim * param.therm.h_an * r_an_dim(z, x_an, T) / Q_scale
        
        r_an_dim_av = r_an_dim(z, x_an, T_av_dim)
        r_an_dim_core = r_an_dim(z_core, x_an, T_core)
        r_an_dim_mid = r_an_dim(z_mid, x_an, T_mid)
        r_an_dim_outer = r_an_dim(z_outer, x_an, T_outer)

        gamma_core = self.param.therm.gamma_core
        gamma_mid = self.param.therm.gamma_mid
        gamma_outer = self.param.therm.gamma_outer

        Q_scale = param.i_typ * param.potential_scale / param.L_x
        Q_exo_an_av = Q_exo_an(z, x_an, T_av_dim)
        Q_exo_an_core = Q_exo_an(z_core, x_an, T_core)#*gamma_core
        Q_exo_an_mid = Q_exo_an(z_mid, x_an, T_mid)#*gamma_mid
        Q_exo_an_outer = Q_exo_an(z_outer, x_an, T_outer)#*gamma_outer

        Q_exo_an = -rho_n_dim * param.therm.h_an * r_an_dim(z, x_an, T_av_dim) / Q_scale # original units W.m-3

        variables = {
            "Anode decomposition reaction rate [s-1]": r_an_dim_av,
            "Anode decomposition reaction rate": r_an_dim_av * param.timescale,
            "Anode decomposition heating": Q_exo_an_av,
            "Anode decomposition heating [W.m-3]": Q_exo_an_av * Q_scale,

            "Core section anode decomposition reaction rate [s-1]": r_an_dim_core,
            "Core section anode decomposition reaction rate": r_an_dim_core * param.timescale,
            "Core section anode decomposition heating": Q_exo_an_core,
            "Core section anode decomposition heating [W.m-3]": Q_exo_an_core * Q_scale,

            "Middle section anode decomposition reaction rate [s-1]": r_an_dim_mid,
            "Middle section anode decomposition reaction rate": r_an_dim_mid * param.timescale,
            "Middle section anode decomposition heating": Q_exo_an_mid,
            "Middle section anode decomposition heating [W.m-3]": Q_exo_an_mid * Q_scale,

            "Outer section anode decomposition reaction rate [s-1]": r_an_dim_outer,
            "Outer section anode decomposition reaction rate": r_an_dim_outer * param.timescale,
            "Outer section anode decomposition heating": Q_exo_an_outer,
            "Outer section anode decomposition heating [W.m-3]": Q_exo_an_outer * Q_scale,
        }
        
        return variables

    def set_rhs(self, variables):
        decomp_rate = variables["Anode decomposition reaction rate"]
        decomp_rate_core = variables["Core section anode decomposition reaction rate"]
        decomp_rate_mid = variables["Middle section anode decomposition reaction rate"]
        decomp_rate_outer = variables["Outer section anode decomposition reaction rate"]

        z = variables["Relative SEI thickness"]
        z_core = variables["Relative SEI thickness in the core section"]
        z_mid = variables["Relative SEI thickness in the middle section"]
        z_outer = variables["Relative SEI thickness in the outer section"]
        self.rhs = {
            z: -decomp_rate*(z>0),
            z_core: -decomp_rate_core*(z>0),
            z_mid: -decomp_rate_mid*(z>0),
            z_outer: -decomp_rate_outer*(z>0),
            }

    def set_initial_conditions(self, variables):
        z = variables["Relative SEI thickness"]
        z_core = variables["Relative SEI thickness in the core section"]
        z_mid = variables["Relative SEI thickness in the middle section"]
        z_outer = variables["Relative SEI thickness in the outer section"]
        
        self.initial_conditions = {
            z: self.param.therm.z_0,
            z_core: self.param.therm.z_0,
            z_mid: self.param.therm.z_0,
            z_outer: self.param.therm.z_0
            }
        
