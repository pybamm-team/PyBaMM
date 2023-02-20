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
        T_x_av = pybamm.Variable(
            "X-averaged cell temperature", domain="current collector"
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
        T_av = variables["X-averaged cell temperature"]
        Q_av = variables["X-averaged total heating"]
        T_amb = variables["Ambient temperature"]

        # Account for surface area to volume ratio of pouch cell in cooling
        # coefficient. Note: the factor 1/delta^2 comes from the choice of
        # non-dimensionalisation
        cell_volume = self.param.l * self.param.l_y * self.param.l_z

        yz_surface_area = self.param.l_y * self.param.l_z
        yz_surface_cooling_coefficient = (
            -(self.param.n.h_cc + self.param.p.h_cc)
            * yz_surface_area
            / cell_volume
            / (self.param.delta**2)
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
        param = self.param
        T_amb = variables["Ambient temperature"]
        T_av = variables["X-averaged cell temperature"]
        T_av_top = pybamm.boundary_value(T_av, "right")
        T_av_bottom = pybamm.boundary_value(T_av, "left")

        # find tab locations (top vs bottom)
        l_z = param.l_z
        neg_tab_z = param.n.centre_z_tab
        pos_tab_z = param.p.centre_z_tab
        neg_tab_top_bool = pybamm.Equality(neg_tab_z, l_z)
        neg_tab_bottom_bool = pybamm.Equality(neg_tab_z, 0)
        pos_tab_top_bool = pybamm.Equality(pos_tab_z, l_z)
        pos_tab_bottom_bool = pybamm.Equality(pos_tab_z, 0)

        # calculate tab vs non-tab area on top and bottom
        neg_tab_area = param.n.l_tab * param.n.l_cc
        pos_tab_area = param.p.l_tab * param.p.l_cc
        total_area = param.l * param.l_y

        non_tab_top_area = (
            total_area
            - neg_tab_area * neg_tab_top_bool
            - pos_tab_area * pos_tab_top_bool
        )
        non_tab_bottom_area = (
            total_area
            - neg_tab_area * neg_tab_bottom_bool
            - pos_tab_area * pos_tab_bottom_bool
        )

        # calculate effective cooling coefficients
        top_cooling_coefficient = (
            (
                param.n.h_tab * neg_tab_area * neg_tab_top_bool
                + param.p.h_tab * pos_tab_area * pos_tab_top_bool
                + param.h_edge * non_tab_top_area
            )
            / param.delta
            / total_area
        )
        bottom_cooling_coefficient = (
            (
                param.n.h_tab * neg_tab_area * neg_tab_bottom_bool
                + param.p.h_tab * pos_tab_area * pos_tab_bottom_bool
                + param.h_edge * non_tab_bottom_area
            )
            / param.delta
            / total_area
        )

        # just use left and right for clarity
        # left = bottom of cell (z=0)
        # right = top of cell (z=L_z)
        self.boundary_conditions = {
            T_av: {
                "left": (
                    bottom_cooling_coefficient * (T_av_bottom - T_amb),
                    "Neumann",
                ),
                "right": (
                    -top_cooling_coefficient * (T_av_top - T_amb),
                    "Neumann",
                ),
            }
        }

    def set_initial_conditions(self, variables):
        T_av = variables["X-averaged cell temperature"]
        self.initial_conditions = {T_av: self.param.T_init}
