#
# Class for one-dimensional thermal submodel for use in the "1+1D" pouch cell model
#
import pybamm

from ..base_thermal import BaseThermal


class CurrentCollector1D(BaseThermal):
    """
    Class for one-dimensional thermal submodel for use in the "1+1D" pouch cell
    model. The thermal model is averaged in the x-direction and is therefore referred
    to as 'x-lumped'. For more information see [1]_ and [2]_.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    References
    ----------
    .. [1] R Timms, SG Marquis, V Sulzer, CP Please and SJ Chapman. “Asymptotic
           Reduction of a Lithium-ion Pouch Cell Model”. In preparation, 2020.
    .. [2] SG Marquis, R Timms, V Sulzer, CP Please and SJ Chapman. “A Suite of
           Reduced-Order Models of a Single-Layer Lithium-ion Pouch Cell”. In
           preparation, 2020.

    **Extends:** :class:`pybamm.thermal.BaseThermal`
    """

    def __init__(self, param):
        super().__init__(param, cc_dimension=1)

    def get_fundamental_variables(self):

        T_x_av = pybamm.standard_variables.T_av
        T_vol_av = self._yz_average(T_x_av)

        T_cn = T_x_av
        T_n = pybamm.PrimaryBroadcast(T_x_av, "negative electrode")
        T_s = pybamm.PrimaryBroadcast(T_x_av, "separator")
        T_p = pybamm.PrimaryBroadcast(T_x_av, "positive electrode")
        T_cp = T_x_av

        variables = self._get_standard_fundamental_variables(
            T_cn, T_n, T_s, T_p, T_cp, T_x_av, T_vol_av
        )

        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_coupled_variables(variables))
        return variables

    def set_rhs(self, variables):
        T_av = variables["X-averaged cell temperature"]
        Q_av = variables["X-averaged total heating"]
        T_amb = variables["Ambient temperature"]

        # Account for surface area to volume ratio of pouch cell in cooling
        # coefficient. Note: the factor 1/delta^2 comes from the choice of
        # non-dimensionalisation
        A = self.param.l_y * self.param.l_z
        V = self.param.l * self.param.l_y * self.param.l_z
        cooling_coeff = -2 * self.param.h * A / V / (self.param.delta ** 2)

        self.rhs = {
            T_av: (
                pybamm.laplacian(T_av)
                + self.param.B * Q_av
                + cooling_coeff * (T_av - T_amb)
            )
            / (self.param.C_th * self.param.rho)
        }

    def set_boundary_conditions(self, variables):
        T_amb = variables["Ambient temperature"]
        T_av = variables["X-averaged cell temperature"]
        T_av_left = pybamm.boundary_value(T_av, "negative tab")
        T_av_right = pybamm.boundary_value(T_av, "positive tab")

        # Three boundary conditions here to handle the cases of both tabs at
        # the same side (top or bottom), or one either side. For both tabs on the
        # same side, T_av_left and T_av_right are equal, and the boundary condition
        # "no tab" is used on the other side.
        self.boundary_conditions = {
            T_av: {
                "negative tab": (
                    self.param.h * (T_av_left - T_amb) / self.param.delta,
                    "Neumann",
                ),
                "positive tab": (
                    -self.param.h * (T_av_right - T_amb) / self.param.delta,
                    "Neumann",
                ),
                "no tab": (pybamm.Scalar(0), "Neumann"),
            }
        }

    def set_initial_conditions(self, variables):
        T_av = variables["X-averaged cell temperature"]
        self.initial_conditions = {T_av: self.param.T_init}
