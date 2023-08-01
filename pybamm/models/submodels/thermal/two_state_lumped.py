#
# Class for lumped thermal submodel
#
import pybamm

from .base_thermal import BaseThermal


class TwoStateLumped(BaseThermal):

    """Class for a two-state (core and surface) lumped thermal submodel

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    cc_dimension: int, optional
        The dimension of the current collectors. Can be 0 (default), 1 or 2.

    **Extends:** :class:`pybamm.thermal.BaseThermal`
    """

    # def __init__(self, param, cc_dimension=0):
    #     super().__init__(param, cc_dimension)
    def __init__(self, param, options=None):
        super().__init__(param, options)


    def get_fundamental_variables(self):
        param = self.param

        T_vol_av = pybamm.standard_variables.T_vol_av
        T_x_av = pybamm.PrimaryBroadcast(T_vol_av, ["current collector"])

        T_cn = T_x_av
        if self.half_cell:
            T_n = None
        else:
            T_n = pybamm.PrimaryBroadcast(T_x_av, "negative electrode")
        T_s = pybamm.PrimaryBroadcast(T_x_av, "separator")
        T_p = pybamm.PrimaryBroadcast(T_x_av, "positive electrode")
        T_cp = T_x_av

        T_surf = pybamm.Variable("Surface cell temperature")
        T_surf_dim = param.Delta_T * T_surf + param.T_ref

        variables = self._get_standard_fundamental_variables(
            T_cn, T_n, T_s, T_p, T_cp, T_x_av, T_vol_av
        )

        dT_surf_dim = (T_vol_av-T_surf)*param.Delta_T
        variables.update({
            "Surface cell temperature": T_surf,
            "Surface cell temperature [K]": T_surf_dim, 
            "Core-surface temperature difference [K]": dT_surf_dim})
        return variables

    def get_coupled_variables(self, variables):
        T_surf_dim = variables["Surface cell temperature [K]"]
        dT_surf_dim = variables["Core-surface temperature difference [K]"]
        variables = self._get_standard_coupled_variables(variables)
        variables.update({
            "Core cell temperature [K]":T_surf_dim + dT_surf_dim 
        })
        return variables

    def set_rhs(self, variables):
        T_vol_av = variables["Volume-averaged cell temperature"]
        Q_vol_av = variables["Volume-averaged total heating"]
        T_amb = variables["Ambient temperature"]
        T_surf = variables["Surface cell temperature"] 

        # Account for surface area to volume ratio in cooling coefficient
        # The factor 1/delta^2 comes from the choice of non-dimensionalisation.
        if self.options["cell geometry"] == "pouch":
            cell_volume = self.param.l * self.param.l_y * self.param.l_z

            yz_cell_surface_area = self.param.l_y * self.param.l_z
            yz_surface_cooling_coefficient = (
                (self.param.n.h_cc + self.param.p.h_cc)
                * yz_cell_surface_area
                / cell_volume
                / (self.param.delta ** 2)
            )

            negative_tab_area = self.param.n.l_tab * self.param.n.l_cc
            negative_tab_cooling_coefficient = (
                self.param.n.h_tab * negative_tab_area / cell_volume / self.param.delta
            )

            positive_tab_area = self.param.p.l_tab * self.param.p.l_cc
            positive_tab_cooling_coefficient = (
                self.param.p.h_tab * positive_tab_area / cell_volume / self.param.delta
            )

            edge_area = (
                2 * self.param.l_y * self.param.l
                + 2 * self.param.l_z * self.param.l
                - negative_tab_area
                - positive_tab_area
            )
            edge_cooling_coefficient = (
                self.param.h_edge * edge_area / cell_volume / self.param.delta
            )

            total_cooling_coefficient = (
                yz_surface_cooling_coefficient
                + negative_tab_cooling_coefficient
                + positive_tab_cooling_coefficient
                + edge_cooling_coefficient
            )
        elif self.options["cell geometry"] == "arbitrary":
            cell_surface_area = self.param.a_cooling
            cell_volume = self.param.v_cell
            total_cooling_coefficient = (
                self.param.h_total
                * cell_surface_area
                / cell_volume
                / (self.param.delta ** 2)
            ) # accurate surface area and volume. nondim?
            
        T_n = variables["Negative electrode temperature"]
        T_s = variables["Separator temperature"]
        T_p = variables["Positive electrode temperature"]
        lambda_k = pybamm.x_average(pybamm.concatenation(
            self.param.n.lambda_(T_n),
            self.param.s.lambda_(T_s),
            self.param.p.lambda_(T_p),
        ))/ (self.param.delta ** 2)/100 # (1m)^2 -> (0.1m)^2 to account for actual electrode height?

        self.rhs = {
            T_vol_av: (
                self.param.B * Q_vol_av - lambda_k * (T_vol_av - T_surf)
            )
            / (self.param.C_th * self.param.rho(T_vol_av)),
            T_surf: (
                lambda_k* (T_vol_av -T_surf) - total_cooling_coefficient* (T_surf-T_amb)
            )
            / (self.param.C_th * self.param.rho(T_vol_av))
        }


    def set_initial_conditions(self, variables):
        T_vol_av = variables["Volume-averaged cell temperature"]
        T_surface = variables["Surface cell temperature"]
        self.initial_conditions = {
            T_vol_av: self.param.T_init, 
            T_surface: self.param.T_init
            }
