#
# Class for SEI decomposition in Li-ion batteries
#
import pybamm
from scipy import constants


class ThreeStateSeiDecomposition(pybamm.BaseSubModel):
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
        x_sei = pybamm.Variable("Fraction of Li in SEI", domain="current collector")
        x_sei_core = pybamm.Variable("Fraction of Li in SEI in the core section", domain="current collector")
        x_sei_mid = pybamm.Variable("Fraction of Li in SEI in the middle section", domain="current collector")
        x_sei_outer = pybamm.Variable("Fraction of Li in SEI in the outer section", domain="current collector")

        variables = {
            "Fraction of Li in SEI": x_sei,
            "Fraction of Li in SEI in the core section": x_sei_core,
            "Fraction of Li in SEI in the middle section": x_sei_mid,
            "Fraction of Li in SEI in the outer section": x_sei_outer}
        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        k_b = pybamm.Scalar(constants.k)
        T_av = variables["X-averaged negative electrode temperature"]
        T_av_dim = param.Delta_T * T_av + param.T_ref
        rho_n_dim = pybamm.Parameter("Negative electrode density [kg.m-3]")
        
        T_outer = param.Delta_T * variables["Outer cell temperature"] + param.T_ref
        T_mid = param.Delta_T * variables["Middle cell temperature"] + param.T_ref
        T_core = param.Delta_T * variables["Core cell temperature"] + param.T_ref

        x_sei = variables["Fraction of Li in SEI"]
        x_sei_core = variables["Fraction of Li in SEI in the core section"]
        x_sei_mid = variables["Fraction of Li in SEI in the middle section"]
        x_sei_outer = variables["Fraction of Li in SEI in the outer section"]

        # r_sei_dim = (
        #     -param.therm.A_sei
        #     * x_sei
        #     * pybamm.exp(-param.therm.E_sei / (k_b * T_av_dim))
        # )  # units 1/s
        def r_sei_dim (x_sei, T):
            r = -param.therm.A_sei * x_sei * pybamm.exp(-param.therm.E_sei / (k_b * T))
            return  r  # units 1/s
        
        def Q_exo_sei(x_sei, T):
            return -rho_n_dim * param.therm.h_sei * r_sei_dim(x_sei, T) / Q_scale
        
        r_sei_dim_av = r_sei_dim(x_sei, T_av_dim)
        r_sei_dim_core = r_sei_dim(x_sei_core, T_core)
        r_sei_dim_mid = r_sei_dim(x_sei_mid, T_mid)
        r_sei_dim_outer = r_sei_dim(x_sei_outer, T_outer)

        gamma_core = self.param.therm.gamma_core
        gamma_mid = self.param.therm.gamma_mid
        gamma_outer = self.param.therm.gamma_outer

        Q_scale = param.i_typ * param.potential_scale / param.L_x
        Q_exo_sei_av = Q_exo_sei(x_sei, T_av_dim)
        Q_exo_sei_core = Q_exo_sei(x_sei_core, T_core)#*gamma_core
        Q_exo_sei_mid = Q_exo_sei(x_sei_mid, T_mid)#*gamma_mid
        Q_exo_sei_outer = Q_exo_sei(x_sei_outer, T_outer)#*gamma_outer

        variables = {
            "SEI decomposition reaction rate [s-1]": r_sei_dim_av,
            "SEI decomposition reaction rate": r_sei_dim_av * param.timescale,
            "SEI decomposition heating": Q_exo_sei_av,
            "SEI decomposition heating [W.m-3]": Q_exo_sei_av * Q_scale,

            "Core section SEI decomposition reaction rate [s-1]": r_sei_dim_core,
            "Core section SEI decomposition reaction rate": r_sei_dim_core * param.timescale,
            "Core section SEI decomposition heating": Q_exo_sei_core,
            "Core section SEI decomposition heating [W.m-3]": Q_exo_sei_core * Q_scale,

            "Middle section SEI decomposition reaction rate [s-1]": r_sei_dim_mid,
            "Middle section SEI decomposition reaction rate": r_sei_dim_mid * param.timescale,
            "Middle section SEI decomposition heating": Q_exo_sei_mid,
            "Middle section SEI decomposition heating [W.m-3]": Q_exo_sei_mid * Q_scale,

            "Outer section SEI decomposition reaction rate [s-1]": r_sei_dim_outer,
            "Outer section SEI decomposition reaction rate": r_sei_dim_outer * param.timescale,
            "Outer section SEI decomposition heating": Q_exo_sei_outer,
            "Outer section SEI decomposition heating [W.m-3]": Q_exo_sei_outer * Q_scale,
        }

        return variables

    def set_rhs(self, variables):
        decomp_rate = variables["SEI decomposition reaction rate"]
        decomp_rate_core = variables["Core section SEI decomposition reaction rate"]
        decomp_rate_mid = variables["Middle section SEI decomposition reaction rate"]
        decomp_rate_outer = variables["Outer section SEI decomposition reaction rate"]

        x_sei = variables["Fraction of Li in SEI"]
        x_sei_core = variables["Fraction of Li in SEI in the core section"]
        x_sei_mid = variables["Fraction of Li in SEI in the middle section"]
        x_sei_outer = variables["Fraction of Li in SEI in the outer section"]
        self.rhs = {
            x_sei: decomp_rate,
            x_sei_core: decomp_rate_core,
            x_sei_mid: decomp_rate_mid,
            x_sei_outer: decomp_rate_outer,
            }

    def set_initial_conditions(self, variables):
        x_sei = variables["Fraction of Li in SEI"]
        x_sei_core = variables["Fraction of Li in SEI in the core section"]
        x_sei_mid = variables["Fraction of Li in SEI in the middle section"]
        x_sei_outer = variables["Fraction of Li in SEI in the outer section"]
        
        self.initial_conditions = {
            x_sei: self.param.therm.x_sei_0,
            x_sei_core: self.param.therm.x_sei_0,
            x_sei_mid: self.param.therm.x_sei_0,
            x_sei_outer: self.param.therm.x_sei_0
            }
