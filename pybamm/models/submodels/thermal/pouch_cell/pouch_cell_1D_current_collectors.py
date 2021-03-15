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
        pybamm.citations.register("Timms2020")

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
        cell_volume = self.param.l * self.param.l_y * self.param.l_z

        yz_surface_area = self.param.l_y * self.param.l_z
        yz_surface_cooling_coefficient = (
            -(self.param.h_cn + self.param.h_cp)
            * yz_surface_area
            / cell_volume
            / (self.param.delta ** 2)
        )

        side_edge_area = 2 * self.param.l_z * self.param.l
        side_edge_cooling_coefficient = (
            -self.param.h_edge * side_edge_area / cell_volume / self.param.delta
        )

        total_cooling_coefficient = (
            yz_surface_cooling_coefficient + side_edge_cooling_coefficient
        )

        self.rhs = {
            T_av: (
                pybamm.laplacian(T_av)
                + self.param.B * Q_av
                + total_cooling_coefficient * (T_av - T_amb)
            )
            / (self.param.C_th * self.param.rho(T_av))
        }

    def set_boundary_conditions(self, variables):
        T_amb = variables["Ambient temperature"]
        T_av = variables["X-averaged cell temperature"]
        T_av_top = pybamm.boundary_value(T_av, "right")
        T_av_bottom = pybamm.boundary_value(T_av, "left")

        # Tab cooling only implemented for both tabs at the top.
        negative_tab_area = self.param.l_tab_n * self.param.l_cn
        positive_tab_area = self.param.l_tab_p * self.param.l_cp
        total_top_area = self.param.l * self.param.l_y
        non_tab_top_area = total_top_area - negative_tab_area - positive_tab_area

        negative_tab_cooling_coefficient = (
            self.param.h_tab_n / self.param.delta * negative_tab_area / total_top_area
        )
        positive_tab_cooling_coefficient = (
            self.param.h_tab_p / self.param.delta * positive_tab_area / total_top_area
        )

        top_edge_cooling_coefficient = (
            self.param.h_edge / self.param.delta * non_tab_top_area / total_top_area
        )

        bottom_edge_cooling_coefficient = (
            self.param.h_edge / self.param.delta * total_top_area / total_top_area
        )

        total_top_cooling_coefficient = (
            negative_tab_cooling_coefficient
            + positive_tab_cooling_coefficient
            + top_edge_cooling_coefficient
        )

        total_bottom_cooling_coefficient = bottom_edge_cooling_coefficient

        # just use left and right for clarity
        # left = bottom of cell (z=0)
        # right = top of cell (z=L_z)
        self.boundary_conditions = {
            T_av: {
                "left": (
                    total_bottom_cooling_coefficient * (T_av_bottom - T_amb),
                    "Neumann",
                ),
                "right": (
                    -total_top_cooling_coefficient * (T_av_top - T_amb),
                    "Neumann",
                ),
            }
        }

    def set_initial_conditions(self, variables):
        T_av = variables["X-averaged cell temperature"]
        self.initial_conditions = {T_av: self.param.T_init}
