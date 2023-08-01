#
# Class for lumped thermal submodel
#
import pybamm

from .base_thermal import BaseThermal


class ThreeStateLumped(BaseThermal):

    """Class for a two-state (core and outer) lumped thermal submodel

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

        T_outer = pybamm.Variable("Outer cell temperature", "current collector")
        T_outer_dim = param.Delta_T * T_outer + param.T_ref
        T_mid = pybamm.Variable("Middle cell temperature", "current collector")
        T_mid_dim = param.Delta_T * T_mid + param.T_ref
        T_core = pybamm.Variable("Core cell temperature", "current collector")
        T_core_dim = param.Delta_T * T_core + param.T_ref


        variables = self._get_standard_fundamental_variables(
            T_cn, T_n, T_s, T_p, T_cp, T_x_av, T_vol_av
        )

        # dT_outer_dim = (T_mid-T_outer)*param.Delta_T
        # dT_mid_dim = (T_core-T_mid)*param.Delta_T

        variables.update({
            "Outer cell temperature": T_outer,
            "Outer cell temperature [K]": T_outer_dim, 
            "Middle cell temperature": T_mid,
            "Middle cell temperature [K]": T_mid_dim,
            "Core cell temperature": T_core,
            "Core cell temperature [K]": T_core_dim, 
            # "Middle-outer temperature difference [K]": dT_surf_dim,
            # "Core-middle temperature difference [K]": dT_mid_dim
            })
        return variables

    def get_coupled_variables(self, variables):
        # T_outer_dim = variables["Surface cell temperature [K]"]
        # dT_outer_dim = variables["Middle-outer temperature difference [K]"]
        # # T_mid_dim = variables["Middle cell temperature [K]"]
        # # dT_mid_dim = variables["Core-middle temperature difference [K]"]
        variables = self._get_standard_coupled_variables(variables)
        # variables.update({
        #     "Core cell temperature [K]":T_outer_dim + dT_outer_dim 
        # })
        param = self.param
        Q_scale = param.i_typ * param.potential_scale / param.L_x # moved to accommodate tabbing I^2R
        I = variables["Current [A]"]
        R_isc = pybamm.Parameter("Internal short resistance [Ohm]")
        R_esc = pybamm.Parameter("External short resistance [Ohm]")
        I_isc = I*(R_esc/(R_esc+R_isc))
        Q_isc = I_isc**2*R_isc/ param.V_cell/Q_scale # originally W.m-3
        variables.update({
            "Internal short circuit ohmic heating": Q_isc,
            "Internal short circuit ohmic heating [W.m-3]": Q_isc * Q_scale,
        })
        return variables

    def set_rhs(self, variables):
        T_vol_av = variables["Volume-averaged cell temperature"]
        Q_vol_av_total = variables["Volume-averaged total heating"] # includes decomp
        Q_ohm_vol_av = variables["Volume-averaged Ohmic heating"]
        Q_rxn_vol_av = variables["Volume-averaged irreversible electrochemical heating"]
        Q_rev_vol_av = variables["Volume-averaged reversible heating"]
        Q_vol_av = Q_ohm_vol_av + Q_rxn_vol_av + Q_rev_vol_av # no decomp

        Q_isc = variables["Internal short circuit ohmic heating"]
        T_amb = variables["Ambient temperature"]
        T_outer = variables["Outer cell temperature"] 
        T_mid = variables["Middle cell temperature"]
        T_core = variables["Core cell temperature"]

        Q_sei = variables["SEI decomposition heating"]
        Q_sei_core = variables["Core section SEI decomposition heating"]
        Q_sei_mid = variables["Middle section SEI decomposition heating"]
        Q_sei_outer = variables["Outer section SEI decomposition heating"]

        Q_ca = variables["Cathode decomposition heating"]
        Q_ca_core = variables["Core section cathode decomposition heating"]
        Q_ca_mid = variables["Middle section cathode decomposition heating"]
        Q_ca_outer = variables["Outer section cathode decomposition heating"]
        
        Q_an = variables["Anode decomposition heating"]
        Q_an_core = variables["Core section anode decomposition heating"]
        Q_an_mid = variables["Middle section anode decomposition heating"]
        Q_an_outer = variables["Outer section anode decomposition heating"]

        gamma_core = self.param.therm.gamma_core
        gamma_mid = self.param.therm.gamma_mid
        gamma_outer = self.param.therm.gamma_outer
        
        # Q_decomp = Q_sei + Q_an + Q_ca
        # Q_decomp_core = Q_sei_core + Q_ca_core + Q_an_core
        # Q_decomp_mid = Q_sei_mid + Q_ca_mid + Q_an_mid
        # Q_decomp_outer= Q_sei_outer + Q_ca_outer + Q_an_outer
        Q_decomp_core = self._x_average(pybamm.concatenation(
            *[
                pybamm.PrimaryBroadcast(Q_sei_core + Q_an_core, ["negative electrode"]),
                pybamm.FullBroadcast(0, ["separator"], "current collector"),
                pybamm.PrimaryBroadcast(Q_ca_core, ["positive electrode"]),
            ]
        ), 0, 0)

        Q_decomp_mid = self._x_average(pybamm.concatenation(
            *[
                pybamm.PrimaryBroadcast(Q_sei_mid + Q_an_mid, ["negative electrode"]),
                pybamm.FullBroadcast(0, ["separator"], "current collector"),
                pybamm.PrimaryBroadcast(Q_ca_mid, ["positive electrode"]),
            ]
        ), 0, 0)

        Q_decomp_outer = self._x_average(pybamm.concatenation(
            *[
                pybamm.PrimaryBroadcast(Q_sei_outer + Q_an_outer, ["negative electrode"]),
                pybamm.FullBroadcast(0, ["separator"], "current collector"),
                pybamm.PrimaryBroadcast(Q_ca_outer, ["positive electrode"]),
            ]
        ), 0, 0)

        Q_decomp = Q_decomp_core*gamma_core + Q_decomp_mid*gamma_mid + Q_decomp_outer*gamma_core #(Q_sei_core + Q_ca_core + Q_an_core)*gamma_core + (Q_sei_mid + Q_ca_mid + Q_an_mid)*gamma_mid + (Q_sei_outer + Q_ca_outer + Q_an_outer)*gamma_outer #Q_sei + Q_an + Q_ca #

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
        # num_layers = 8*8
        lambda_k = pybamm.x_average(pybamm.concatenation(
            self.param.n.lambda_(T_n),
            self.param.s.lambda_(T_s),
            self.param.p.lambda_(T_p),
        ))/ (self.param.delta**2)/100  # squared? number of layers

        self.rhs = {
            T_vol_av: (
                self.param.B * (Q_vol_av + Q_decomp) - total_cooling_coefficient * (T_vol_av - T_amb)
            )
            / (self.param.C_th * self.param.rho(T_vol_av)),
            T_outer: (
                self.param.B * (Q_vol_av + Q_decomp_outer)*gamma_outer + lambda_k*(gamma_mid + gamma_core) *(T_mid -T_outer) - total_cooling_coefficient* (T_outer-T_amb)
            )
            / (self.param.C_th * self.param.rho(T_vol_av)*gamma_outer),
            T_mid: (
                self.param.B * (Q_vol_av+ Q_decomp_mid)*gamma_mid + lambda_k*gamma_core *(T_core -T_mid) - lambda_k*(gamma_mid + gamma_core) *(T_mid -T_outer)
            )
            / (self.param.C_th * self.param.rho(T_vol_av)*gamma_mid),
            T_core: (
                self.param.B * (Q_vol_av+ Q_decomp_core)*gamma_core + self.param.B * Q_isc- lambda_k*gamma_core * (T_core - T_mid)
            )
            / (self.param.C_th * self.param.rho(T_vol_av)*gamma_core)
        }

 
    def set_initial_conditions(self, variables):
        T_vol_av = variables["Volume-averaged cell temperature"]
        T_outer = variables["Outer cell temperature"]
        T_mid = variables["Middle cell temperature"]
        T_core = variables["Core cell temperature"]

        self.initial_conditions = {
            T_vol_av: self.param.T_init, 
            T_outer: self.param.T_init,
            T_mid: self.param.T_init,
            T_core: self.param.T_init,
            }
