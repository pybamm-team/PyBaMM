#
# Class for SEI decomposition in Li-ion batteries
#
import pybamm
from scipy import constants


class SeiDecomposition(pybamm.BaseSubModel):
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
        x_sei = pybamm.Variable("Fraction of Li in SEI", domain="negative electrode", auxiliary_domains={"secondary": "current collector"},)

        variables = {"Fraction of Li in SEI": x_sei}
        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        k_b = pybamm.Scalar(constants.k)
        T = variables["Negative electrode temperature"]
        T_dimensional = param.Delta_T * T + param.T_ref
        x_sei = variables["Fraction of Li in SEI"]
        rho_n_dim = pybamm.Parameter("Negative electrode density [kg.m-3]")

        r_sei_dimensional = (
            -param.therm.A_sei
            * x_sei
            * pybamm.exp(-param.therm.E_sei / (k_b * T_dimensional))
        )  # units 1/s

        Q_scale = param.i_typ * param.potential_scale / param.L_x
        Q_exo_sei = -rho_n_dim * param.therm.h_sei * r_sei_dimensional / Q_scale

        variables = {
            "SEI decomposition reaction rate [s-1]": r_sei_dimensional,
            "SEI decomposition reaction rate": r_sei_dimensional * param.timescale,
            "SEI decomposition heating": Q_exo_sei,
            "SEI decomposition heating [W.m-3]": Q_exo_sei * Q_scale,
        }

        return variables

    def set_rhs(self, variables):
        decomp_rate = variables["SEI decomposition reaction rate"]
        x_sei = variables["Fraction of Li in SEI"]
        self.rhs = {x_sei: decomp_rate}

    def set_initial_conditions(self, variables):
        x_sei = variables["Fraction of Li in SEI"]
        x_sei_0 = pybamm.FullBroadcast(
                     self.param.therm.x_sei_0, ["negative electrode"], "current collector"
                )
        self.initial_conditions = {x_sei: x_sei_0}
