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
        z = pybamm.Variable("Relative SEI thickness", domain= "negative electrode", auxiliary_domains={"secondary": "current collector"},)

        variables = {"Relative SEI thickness": z}
        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        k_b = pybamm.Scalar(constants.k)
        T = variables["Negative electrode temperature"]
        T_dimensional = param.Delta_T * T + param.T_ref
        c_s_n_surf = variables["X-averaged negative particle surface concentration"]
        z = variables["Relative SEI thickness"]
        x_an = c_s_n_surf
        rho_n_dim = pybamm.Parameter("Negative electrode density [kg.m-3]")


        r_an_dimensional = (
            -param.therm.A_an
            * x_an
            * pybamm.exp(-param.therm.E_an / (k_b * T_dimensional))
            * pybamm.exp(-z / param.therm.z_0)
        )  # units 1/s

        Q_scale = param.i_typ * param.potential_scale / param.L_x
        Q_exo_an = -rho_n_dim * param.therm.h_an * r_an_dimensional / Q_scale # original units W.m-3

        variables = {
            "Anode decomposition reaction rate [s-1]": r_an_dimensional,
            "Anode decomposition reaction rate": r_an_dimensional * param.timescale,
            "Anode decomposition heating": Q_exo_an,
            "Anode decomposition heating [W.m-3]": Q_exo_an * Q_scale,
        }

        return variables

    def set_rhs(self, variables):
        decomp_rate = variables["Anode decomposition reaction rate"]
        z = variables["Relative SEI thickness"]

        self.rhs = {z: -decomp_rate*(z>0)}

    def set_initial_conditions(self, variables):
        z = variables["Relative SEI thickness"]
        z_0 = pybamm.FullBroadcast(
                self.param.therm.z_0, ["negative electrode"], "current collector"
        )
        self.initial_conditions = {z: z_0}
