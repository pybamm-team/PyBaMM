#
# Class for one-dimensional (x-direction) thermal submodel
#
import pybamm

from pybamm.models.submodels.thermal.base_thermal import BaseThermal


class OneDimensionalX(BaseThermal):
    """
    Class for one-dimensional (x-direction) thermal submodel.
    Note: this model assumes infinitely large electrical and thermal conductivity
    in the current collectors, so that the contribution to the Ohmic heating
    from the current collectors is zero and the boundary conditions are applied
    at the edges of the electrodes (at x=0 and x=1, in non-dimensional coordinates).
    For more information see :footcite:t:`Timms2021` and :footcite:t:`Marquis2020`.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    """

    def __init__(self, param, options=None, x_average=False):
        super().__init__(param, options=options, x_average=x_average)
        pybamm.citations.register("Timms2021")

    def get_fundamental_variables(self):
        T_dict = {}
        for domain in ["negative electrode", "separator", "positive electrode"]:
            Domain = domain.capitalize()
            T_k = pybamm.Variable(
                f"{Domain} temperature [K]",
                domain=domain,
                auxiliary_domains={"secondary": "current collector"},
                scale=self.param.T_ref,
            )
            T_dict[domain] = T_k

        T = pybamm.concatenation(*T_dict.values())
        T_cn = pybamm.Variable(
            "Negative current collector temperature [K]",
            domain="current collector",
            scale=self.param.T_ref,
        )
        T_cp = pybamm.Variable(
            "Positive current collector temperature [K]",
            domain="current collector",
            scale=self.param.T_ref,
        )
        T_x_av = self._x_average(T, T_cn, T_cp)
        T_vol_av = self._yz_average(T_x_av)
        T_dict.update(
            {
                "negative current collector": T_cn,
                "positive current collector": T_cp,
                "x-averaged cell": T_x_av,
                "volume-averaged cell": T_vol_av,
            }
        )

        variables = self._get_standard_fundamental_variables(T_dict)
        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_coupled_variables(variables))
        return variables

    def set_rhs(self, variables):
        T = variables["Cell temperature [K]"]
        T_cn = variables["Negative current collector temperature [K]"]
        T_n = variables["Negative electrode temperature [K]"]
        T_s = variables["Separator temperature [K]"]
        T_p = variables["Positive electrode temperature [K]"]
        T_cp = variables["Positive current collector temperature [K]"]
        Q = variables["Total heating [W.m-3]"]
        Q_cn = variables["Negative current collector Ohmic heating [W.m-3]"]
        Q_cp = variables["Positive current collector Ohmic heating [W.m-3]"]
        T_amb = variables["Ambient temperature [K]"]
        y = pybamm.standard_spatial_vars.y
        z = pybamm.standard_spatial_vars.z

        # Define volumetric heat capacity for electrode/separator/electrode sandwich
        rho_c_p = pybamm.concatenation(
            self.param.n.rho_c_p(T_n),
            self.param.s.rho_c_p(T_s),
            self.param.p.rho_c_p(T_p),
        )

        # Define thermal conductivity for electrode/separator/electrode sandwich
        lambda_ = pybamm.concatenation(
            self.param.n.lambda_(T_n),
            self.param.s.lambda_(T_s),
            self.param.p.lambda_(T_p),
        )

        # Calculate edge/tab cooling
        L_y = self.param.L_y
        L_z = self.param.L_z
        L_cn = self.param.n.L_cc
        L_cp = self.param.p.L_cc
        h_cn = self.param.n.h_cc(y, z)
        h_cp = self.param.p.h_cc(y, z)
        lambda_n = self.param.n.lambda_(T_n)
        lambda_p = self.param.p.lambda_(T_p)
        h_edge = self.param.h_edge(y, z)
        # Negative current collector
        volume_cn = L_cn * L_y * L_z
        negative_tab_area = self.param.n.L_tab * self.param.n.L_cc
        edge_area_cn = 2 * (L_y + L_z) * L_cn - negative_tab_area
        negative_tab_cooling_coefficient = (
            -self.param.n.h_tab * negative_tab_area / volume_cn
        )
        edge_cooling_coefficient_cn = -h_edge * edge_area_cn / volume_cn
        cooling_coefficient_cn = (
            negative_tab_cooling_coefficient + edge_cooling_coefficient_cn
        )
        # Electrode/separator/electrode sandwich
        area_to_volume = (
            2 * (self.param.L_y + self.param.L_z) / (self.param.L_y * self.param.L_z)
        )
        cooling_coefficient = -h_edge * area_to_volume
        # Positive current collector
        volume_cp = L_cp * L_y * L_z
        positive_tab_area = self.param.p.L_tab * self.param.p.L_cc
        edge_area_cp = 2 * (L_y + L_z) * L_cp - positive_tab_area
        positive_tab_cooling_coefficient = (
            -self.param.p.h_tab * positive_tab_area / volume_cp
        )
        edge_cooling_coefficient_cp = -h_edge * edge_area_cp / volume_cp
        cooling_coefficient_cp = (
            positive_tab_cooling_coefficient + edge_cooling_coefficient_cp
        )

        self.rhs = {
            T_cn: (
                (
                    pybamm.boundary_value(lambda_n, "left")
                    * pybamm.boundary_gradient(T_n, "left")
                    - h_cn * (T_cn - T_amb)
                )
                / L_cn
                + Q_cn
                + cooling_coefficient_cn * (T_cn - T_amb)
            )
            / self.param.n.rho_c_p_cc(T_cn),
            T: (
                pybamm.div(lambda_ * pybamm.grad(T))
                + Q
                + cooling_coefficient * (T - T_amb)
            )
            / rho_c_p,
            T_cp: (
                (
                    -pybamm.boundary_value(lambda_p, "right")
                    * pybamm.boundary_gradient(T_p, "right")
                    - h_cp * (T_cp - T_amb)
                )
                / L_cp
                + Q_cp
                + cooling_coefficient_cp * (T_cp - T_amb)
            )
            / self.param.p.rho_c_p_cc(T_cp),
        }

    def set_boundary_conditions(self, variables):
        T = variables["Cell temperature [K]"]
        T_cn = variables["Negative current collector temperature [K]"]
        T_cp = variables["Positive current collector temperature [K]"]

        self.boundary_conditions = {
            T: {"left": (T_cn, "Dirichlet"), "right": (T_cp, "Dirichlet")}
        }

    def set_initial_conditions(self, variables):
        T = variables["Cell temperature [K]"]
        T_cn = variables["Negative current collector temperature [K]"]
        T_cp = variables["Positive current collector temperature [K]"]
        T_init = self.param.T_init
        self.initial_conditions = {T_cn: T_init, T: T_init, T_cp: T_init}
