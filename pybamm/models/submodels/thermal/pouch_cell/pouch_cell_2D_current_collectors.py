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
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)
        pybamm.citations.register("Timms2021")

    def get_fundamental_variables(self):
        T_x_av = pybamm.Variable(
            "X-averaged cell temperature [K]",
            domain="current collector",
            scale=self.param.T_ref,
        )
        T_vol_av = self._yz_average(T_x_av)

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
        T_av = variables["X-averaged cell temperature [K]"]
        Q_av = variables["X-averaged total heating [W.m-3]"]
        T_amb = variables["Ambient temperature [K]"]

        # Account for surface area to volume ratio of pouch cell in cooling
        # coefficient. Note: the factor 1/delta^2 comes from the choice of
        # non-dimensionalisation
        yz_surface_area = self.param.L_y * self.param.L_z
        cell_volume = self.param.L * self.param.L_y * self.param.L_z
        yz_surface_cooling_coefficient = (
            -(self.param.n.h_cc + self.param.p.h_cc) * yz_surface_area / cell_volume
        )

        edge_cooling_coefficient = self.param.h_edge

        # Governing equations contain:
        #   - source term for y-z surface cooling
        #   - boundary source term of edge cooling
        # Boundary conditions contain:
        #   - Neumann condition for tab cooling
        self.rhs = {
            T_av: (
                pybamm.laplacian(T_av)
                + pybamm.source(Q_av, T_av)
                + yz_surface_cooling_coefficient * pybamm.source(T_av - T_amb, T_av)
                - edge_cooling_coefficient
                * pybamm.source(T_av - T_amb, T_av, boundary=True)
            )
            / self.param.rho_c_p_eff(T_av)
        }

        # TODO: Make h_edge a function of position to have bottom/top/side cooled cells.

    def set_boundary_conditions(self, variables):
        T_av = variables["X-averaged cell temperature [K]"]
        T_amb = variables["Ambient temperature [K]"]

        # Subtract the edge cooling from the tab portion so as to not double count
        # Note: tab cooling is also only applied on the current collector hence
        # the (l_cn / l) and (l_cp / l) prefactors.
        # We also still have edge cooling on the region: x in (0, 1)
        h_tab_n_corrected = (self.param.n.L_cc / self.param.L) * (
            self.param.n.h_tab - self.param.h_edge
        )
        h_tab_p_corrected = (self.param.p.L_cc / self.param.L) * (
            self.param.p.h_tab - self.param.h_edge
        )

        T_av_n = pybamm.boundary_value(T_av, "negative tab")
        T_av_p = pybamm.boundary_value(T_av, "positive tab")

        self.boundary_conditions = {
            T_av: {
                "negative tab": (-h_tab_n_corrected * (T_av_n - T_amb), "Neumann"),
                "positive tab": (-h_tab_p_corrected * (T_av_p - T_amb), "Neumann"),
            }
        }

    def set_initial_conditions(self, variables):
        T_av = variables["X-averaged cell temperature [K]"]
        self.initial_conditions = {T_av: self.param.T_init}
