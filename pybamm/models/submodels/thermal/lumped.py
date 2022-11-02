#
# Class for lumped thermal submodel
#
import pybamm

from .base_thermal import BaseThermal


class Lumped(BaseThermal):
    """
    Class for lumped thermal submodel. For more information see [1]_ and [2]_.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    References
    ----------
    .. [1] R Timms, SG Marquis, V Sulzer, CP Please and SJ Chapman. “Asymptotic
           Reduction of a Lithium-ion Pouch Cell Model”. SIAM Journal on Applied
           Mathematics, 81(3), 765--788, 2021
    .. [2] SG Marquis, R Timms, V Sulzer, CP Please and SJ Chapman. “A Suite of
           Reduced-Order Models of a Single-Layer Lithium-ion Pouch Cell”. Journal
           of The Electrochemical Society, 167(14):140513, 2020

    **Extends:** :class:`pybamm.thermal.BaseThermal`
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)
        pybamm.citations.register("Timms2021")

    def get_fundamental_variables(self):
        T_vol_av = pybamm.Variable("Volume-averaged cell temperature")
        T_x_av = pybamm.PrimaryBroadcast(T_vol_av, ["current collector"])
        T_dict = {
            "negative current collector": T_x_av,
            "positive current collector": T_x_av,
            "x-averaged cell": T_x_av,
            "volume-averaged cell": T_vol_av,
        }
        for domain in ["negative electrode", "separator", "positive electrode"]:
            T_dict[domain] = pybamm.PrimaryBroadcast(T_x_av, domain)

        variables = self._get_standard_fundamental_variables(T_dict)

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
        if self.options["cell geometry"] == "pouch":
            cell_volume = self.param.l * self.param.l_y * self.param.l_z

            yz_cell_surface_area = self.param.l_y * self.param.l_z
            yz_surface_cooling_coefficient = (
                -(self.param.n.h_cc + self.param.p.h_cc)
                * yz_cell_surface_area
                / cell_volume
                / (self.param.delta**2)
            )

            negative_tab_area = self.param.n.l_tab * self.param.n.l_cc
            negative_tab_cooling_coefficient = (
                -self.param.n.h_tab * negative_tab_area / cell_volume / self.param.delta
            )

            positive_tab_area = self.param.p.l_tab * self.param.p.l_cc
            positive_tab_cooling_coefficient = (
                -self.param.p.h_tab * positive_tab_area / cell_volume / self.param.delta
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
        elif self.options["cell geometry"] == "arbitrary":
            cell_surface_area = self.param.a_cooling
            cell_volume = self.param.v_cell
            total_cooling_coefficient = (
                -self.param.h_total
                * cell_surface_area
                / cell_volume
                / (self.param.delta**2)
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
