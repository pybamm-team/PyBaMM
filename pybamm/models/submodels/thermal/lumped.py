#
# Class for lumped thermal submodel
#
import pybamm

from .base_thermal import BaseThermal


class Lumped(BaseThermal):
    """Class for lumped thermal submodel. For more information see [1]_.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    cc_dimension: int, optional
        The dimension of the current collectors. Can be 0 (default), 1 or 2.
    geometry: string, optional
        The geometry for the lumped thermal submodel. Can be "arbitrary" (default) or
        pouch.

    References
    ----------
    .. [1] R Timms, SG Marquis, V Sulzer, CP Please and SJ Chapman. “Asymptotic
           Reduction of a Lithium-ion Pouch Cell Model”. In preparation, 2020.

    **Extends:** :class:`pybamm.thermal.BaseThermal`
    """

    def __init__(self, param, cc_dimension=0, geometry="arbitrary"):
        self.geometry = geometry
        super().__init__(param, cc_dimension)
        pybamm.citations.register("Timms2020")

    def get_fundamental_variables(self):

        T_vol_av = pybamm.standard_variables.T_vol_av
        T_x_av = pybamm.PrimaryBroadcast(T_vol_av, ["current collector"])

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
        T_vol_av = variables["Volume-averaged cell temperature"]
        Q_vol_av = variables["Volume-averaged total heating"]
        T_amb = variables["Ambient temperature"]

        # Account for surface area to volume ratio in cooling coefficient
        # The factor 1/delta^2 comes from the choice of non-dimensionalisation.
        if self.geometry == "pouch":
            cell_volume = self.param.l * self.param.l_y * self.param.l_z

            yz_cell_surface_area = self.param.l_y * self.param.l_z
            yz_surface_cooling_coefficient = (
                -(self.param.h_cn + self.param.h_cp)
                * yz_cell_surface_area
                / cell_volume
                / (self.param.delta ** 2)
            )

            negative_tab_area = self.param.l_tab_n * self.param.l_cn
            negative_tab_cooling_coefficient = (
                -self.param.h_tab_n * negative_tab_area / cell_volume / self.param.delta
            )

            positive_tab_area = self.param.l_tab_p * self.param.l_cp
            positive_tab_cooling_coefficient = (
                -self.param.h_tab_p * positive_tab_area / cell_volume / self.param.delta
            )

            edge_area = (
                2 * self.param.l_y * self.param.l
                + 2 * self.param.l_z * self.param.l
                - negative_tab_area
                - positive_tab_area
            )
            edge_cooling_coefficient = (
                -self.param.h_edge * edge_area / cell_volume / self.param.delta
            )

            total_cooling_coefficient = (
                yz_surface_cooling_coefficient
                + negative_tab_cooling_coefficient
                + positive_tab_cooling_coefficient
                + edge_cooling_coefficient
            )
        elif self.geometry == "arbitrary":
            cell_surface_area = self.param.a_cooling
            cell_volume = self.param.v_cell
            total_cooling_coefficient = (
                -self.param.h_total
                * cell_surface_area
                / cell_volume
                / (self.param.delta ** 2)
            )

        self.rhs = {
            T_vol_av: (
                self.param.B * Q_vol_av + total_cooling_coefficient * (T_vol_av - T_amb)
            )
            / (self.param.C_th * self.param.rho(T_vol_av))
        }

    def set_initial_conditions(self, variables):
        T_vol_av = variables["Volume-averaged cell temperature"]
        self.initial_conditions = {T_vol_av: self.param.T_init}
