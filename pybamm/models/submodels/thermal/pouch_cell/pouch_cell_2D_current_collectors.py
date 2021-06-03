#
# Class for two-dimensional thermal submodel for use in the "2+1D" pouch cell model
#
import pybamm

from ..base_thermal import BaseThermal


class CurrentCollector2D(BaseThermal):
    """
    Class for two-dimensional thermal submodel for use in the "2+1D" pouch cell
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
        super().__init__(param, cc_dimension=2)
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
        yz_surface_area = self.param.l_y * self.param.l_z
        cell_volume = self.param.l * self.param.l_y * self.param.l_z
        yz_surface_cooling_coefficient = (
            -(self.param.h_cn + self.param.h_cp)
            * yz_surface_area
            / cell_volume
            / (self.param.delta ** 2)
        )

        edge_cooling_coefficient = self.param.h_edge / self.param.delta

        # Governing equations contain:
        #   - source term for y-z surface cooling
        #   - boundary source term of edge cooling
        # Boundary conditions contain:
        #   - Neumann condition for tab cooling
        self.rhs = {
            T_av: (
                pybamm.laplacian(T_av)
                + self.param.B * pybamm.source(Q_av, T_av)
                + yz_surface_cooling_coefficient * pybamm.source(T_av - T_amb, T_av)
                - edge_cooling_coefficient
                * pybamm.source(T_av - T_amb, T_av, boundary=True)
            )
            / (self.param.C_th * self.param.rho(T_av))
        }

        # TODO: Make h_edge a function of position to have bottom/top/side cooled cells.

    def set_boundary_conditions(self, variables):
        T_av = variables["X-averaged cell temperature"]
        T_amb = variables["Ambient temperature"]

        # Subtract the edge cooling from the tab portion so as to not double count
        # Note: tab cooling is also only applied on the current collector hence
        # the (l_cn / l) and (l_cp / l) prefactors.
        # We also still have edge cooling on the region: x in (0, 1)
        h_tab_n_corrected = (
            (self.param.l_cn / self.param.l)
            * (self.param.h_tab_n - self.param.h_edge)
            / self.param.delta
        )
        h_tab_p_corrected = (
            (self.param.l_cp / self.param.l)
            * (self.param.h_tab_p - self.param.h_edge)
            / self.param.delta
        )

        T_av_n = pybamm.BoundaryValue(T_av, "negative tab")
        T_av_p = pybamm.BoundaryValue(T_av, "positive tab")

        self.boundary_conditions = {
            T_av: {
                "negative tab": (-h_tab_n_corrected * (T_av_n - T_amb), "Neumann"),
                "positive tab": (-h_tab_p_corrected * (T_av_p - T_amb), "Neumann"),
            }
        }

    def set_initial_conditions(self, variables):
        T_av = variables["X-averaged cell temperature"]
        self.initial_conditions = {T_av: self.param.T_init}
