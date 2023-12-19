#
# Class for two-dimensional thermal submodel for use in the "2+1D" pouch cell model
#
import pybamm

from ..base_thermal import BaseThermal


class CurrentCollector2D(BaseThermal):
    """
    Class for two-dimensional thermal submodel for use in the "2+1D" pouch cell
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
        y = pybamm.standard_spatial_vars.y
        z = pybamm.standard_spatial_vars.z

        # Calculate cooling
        Q_yz_surface_W_per_m2 = -(self.param.n.h_cc(y, z) + self.param.p.h_cc(y, z)) * (
            T_av - T_amb
        )
        Q_edge_W_per_m2 = -self.param.h_edge(y, z) * (T_av - T_amb)

        # Account for surface area to volume ratio of pouch cell in surface cooling
        # term
        yz_surface_area = self.param.L_y * self.param.L_z
        cell_volume = self.param.L * self.param.L_y * self.param.L_z
        Q_yz_surface = pybamm.source(
            Q_yz_surface_W_per_m2 * yz_surface_area / cell_volume, T_av
        )
        # Edge cooling appears as a boundary term, so no need to account for surface
        # area to volume ratio
        Q_edge = pybamm.source(Q_edge_W_per_m2, T_av, boundary=True)

        # Governing equations contain:
        #   - source term for y-z surface cooling
        #   - boundary source term of edge cooling
        # Boundary conditions contain:
        #   - Neumann condition for tab cooling
        # Note: pybamm.source() is used to ensure the source term is multiplied by the
        # correct mass matrix when discretised. The first argument is the source term
        # and the second argument is the variable governed by the equation that the
        # source term appears in.
        # Note: not correct if lambda_eff is a function of T_av - need to implement div
        # in 2D rather than doing laplacian directly
        self.rhs = {
            T_av: (
                self.param.lambda_eff(T_av) * pybamm.laplacian(T_av)
                + pybamm.source(Q_av, T_av)
                + Q_yz_surface
                + Q_edge
            )
            / self.param.rho_c_p_eff(T_av)
        }

    def set_boundary_conditions(self, variables):
        T_av = variables["X-averaged cell temperature [K]"]
        T_amb = variables["Ambient temperature [K]"]
        y = pybamm.standard_spatial_vars.y
        z = pybamm.standard_spatial_vars.z

        # Calculate heat fluxes
        q_tab_n = -self.param.n.h_tab * (T_av - T_amb)
        q_tab_p = -self.param.p.h_tab * (T_av - T_amb)
        q_edge = -self.param.h_edge(y, z) * (T_av - T_amb)

        # Subtract the edge cooling from the tab portion so as to not double count
        # Note: tab cooling is also only applied on the current collector hence
        # the (l_cn / l) and (l_cp / l) prefactors. We still have edge cooling
        # in the region: x in (0, 1)
        negative_tab_bc = (self.param.n.L_cc / self.param.L) * pybamm.boundary_value(
            (q_tab_n - q_edge) / self.param.n.lambda_cc(T_av),
            "negative tab",
        )
        positive_tab_bc = (self.param.p.L_cc / self.param.L) * pybamm.boundary_value(
            (q_tab_p - q_edge) / self.param.p.lambda_cc(T_av), "positive tab"
        )

        self.boundary_conditions = {
            T_av: {
                "negative tab": (negative_tab_bc, "Neumann"),
                "positive tab": (positive_tab_bc, "Neumann"),
            }
        }

    def set_initial_conditions(self, variables):
        T_av = variables["X-averaged cell temperature [K]"]
        self.initial_conditions = {T_av: self.param.T_init}
