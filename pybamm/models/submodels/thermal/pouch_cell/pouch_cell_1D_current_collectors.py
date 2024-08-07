#
# Class for one-dimensional thermal submodel for use in the "1+1D" pouch cell model
#
import pybamm

from pybamm.models.submodels.thermal.base_thermal import BaseThermal


class CurrentCollector1D(BaseThermal):
    """
    Class for one-dimensional thermal submodel for use in the "1+1D" pouch cell
    model. The thermal model is averaged in the x-direction and is therefore referred
    to as 'x-lumped'. For more information see :footcite:t:`Timms2021` and
    :footcite:t:`Marquis2020`.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    """

    def __init__(self, param, options=None, x_average=True):
        super().__init__(param, options=options, x_average=x_average)
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
        y = pybamm.standard_spatial_vars.y
        z = pybamm.standard_spatial_vars.z

        # Calculate cooling, accounting for surface area to volume ratio of pouch cell
        edge_area = self.param.L_z * self.param.L
        yz_surface_area = self.param.L_y * self.param.L_z
        cell_volume = self.param.L * self.param.L_y * self.param.L_z
        Q_yz_surface = (
            -(self.param.n.h_cc(y, z) + self.param.p.h_cc(y, z))
            * (T_av - T_amb)
            * yz_surface_area
            / cell_volume
        )
        Q_edge = (
            -(self.param.h_edge(0, z) + self.param.h_edge(self.param.L_y, z))
            * (T_av - T_amb)
            * edge_area
            / cell_volume
        )
        Q_cool_total = Q_yz_surface + Q_edge

        self.rhs = {
            T_av: (
                pybamm.div(self.param.lambda_eff(T_av) * pybamm.grad(T_av))
                + Q_av
                + Q_cool_total
            )
            / self.param.rho_c_p_eff(T_av)
        }

    def set_boundary_conditions(self, variables):
        param = self.param
        T_amb = variables["Ambient temperature [K]"]
        T_av = variables["X-averaged cell temperature [K]"]

        # Find tab locations (top vs bottom)
        L_y = param.L_y
        L_z = param.L_z
        neg_tab_z = param.n.centre_z_tab
        pos_tab_z = param.p.centre_z_tab
        neg_tab_top_bool = pybamm.Equality(neg_tab_z, L_z)
        neg_tab_bottom_bool = pybamm.Equality(neg_tab_z, 0)
        pos_tab_top_bool = pybamm.Equality(pos_tab_z, L_z)
        pos_tab_bottom_bool = pybamm.Equality(pos_tab_z, 0)

        # Calculate tab vs non-tab area on top and bottom
        neg_tab_area = param.n.L_tab * param.n.L_cc
        pos_tab_area = param.p.L_tab * param.p.L_cc
        total_area = param.L * param.L_y
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

        # Calculate heat fluxes weighted by area
        # Note: can't do y-average of h_edge here since y isn't meshed. Evaluate at
        # midpoint.
        q_tab_n = -param.n.h_tab * (T_av - T_amb)
        q_tab_p = -param.p.h_tab * (T_av - T_amb)
        q_edge_top = -param.h_edge(L_y / 2, L_z) * (T_av - T_amb)
        q_edge_bottom = -param.h_edge(L_y / 2, 0) * (T_av - T_amb)
        q_top = (
            q_tab_n * neg_tab_area * neg_tab_top_bool
            + q_tab_p * pos_tab_area * pos_tab_top_bool
            + q_edge_top * non_tab_top_area
        ) / total_area
        q_bottom = (
            q_tab_n * neg_tab_area * neg_tab_bottom_bool
            + q_tab_p * pos_tab_area * pos_tab_bottom_bool
            + q_edge_bottom * non_tab_bottom_area
        ) / total_area

        # just use left and right for clarity
        # left = bottom of cell (z=0)
        # right = top of cell (z=L_z)
        lambda_eff = param.lambda_eff(T_av)
        self.boundary_conditions = {
            T_av: {
                "left": (
                    pybamm.boundary_value(-q_bottom / lambda_eff, "left"),
                    "Neumann",
                ),
                "right": (
                    pybamm.boundary_value(q_top / lambda_eff, "right"),
                    "Neumann",
                ),
            },
        }

    def set_initial_conditions(self, variables):
        T_av = variables["X-averaged cell temperature [K]"]
        self.initial_conditions = {T_av: self.param.T_init}
