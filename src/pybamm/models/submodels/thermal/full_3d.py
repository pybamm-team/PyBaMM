import pybamm
from pybamm.models.submodels.thermal.base_thermal import BaseThermal


class FullThreeDimensional(BaseThermal):
    """
    Class for three-dimensional thermal submodel with constant heat source.

    This model solves the heat equation in 3D:
        rho_c_p * dT/dt = div(lambda * grad(T)) + Q

    where:
        - T is the temperature field in the 3D cell domain
        - rho_c_p is the volumetric heat capacity
        - lambda is the thermal conductivity
        - Q is the volumetric heat source

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    x_average : bool, optional
        Whether to include x-averaged variables. Default is False.
    """

    def __init__(self, param, options=None, x_average=False):
        options = options or {}
        options["dimensionality"] = 3
        super().__init__(param, options=options, x_average=x_average)
        self._geometry = (
            self.options.get("geometry options", {})
            .get("domains", {})
            .get("cell", {})
            .get("type", "rectangular")
        )
        pybamm.citations.register("Timms2021")

    def get_fundamental_variables(self):
        """Define the fundamental variables in the model."""
        T = pybamm.Variable(
            "Cell temperature [K]",
            domain="cell",
            auxiliary_domains={
                "secondary": ["current collector y"],
                "tertiary": ["current collector z"],
            },
            scale=self.param.T_ref,
        )

        T_cn = pybamm.Variable(
            "Negative current collector temperature [K]",
            domain="negative current collector",
            auxiliary_domains={
                "secondary": ["current collector y"],
                "tertiary": ["current collector z"],
            },
            scale=self.param.T_ref,
        )

        T_cp = pybamm.Variable(
            "Positive current collector temperature [K]",
            domain="positive current collector",
            auxiliary_domains={
                "secondary": ["current collector y"],
                "tertiary": ["current collector z"],
            },
            scale=self.param.T_ref,
        )

        y = pybamm.SpatialVariable("y", domain=["current collector y"])
        z = pybamm.SpatialVariable("z", domain=["current collector z"])

        T_y_av = pybamm.Integral(T, y) / self.param.L_y
        T_cn_y_av = pybamm.Integral(T_cn, y) / self.param.L_y
        T_cp_y_av = pybamm.Integral(T_cp, y) / self.param.L_y

        T_yz_av = pybamm.Integral(T_y_av, z) / self.param.L_z
        T_cn_yz_av = pybamm.Integral(T_cn_y_av, z) / self.param.L_z
        T_cp_yz_av = pybamm.Integral(T_cp_y_av, z) / self.param.L_z

        L_n = self.param.n.L_cc
        L_p = self.param.p.L_cc
        L_x = self.param.L_x

        T_cn_yz_av_n = pybamm.PrimaryBroadcast(T_cn_yz_av, ["negative electrode"])
        T_cp_yz_av_p = pybamm.PrimaryBroadcast(T_cp_yz_av, ["positive electrode"])

        T_cn_yz_av_cell = pybamm.PrimaryBroadcast(T_cn_yz_av_n, ["cell"])
        T_cp_yz_av_cell = pybamm.PrimaryBroadcast(T_cp_yz_av_p, ["cell"])

        T_vol_av = (L_n * T_cn_yz_av_cell + L_x * T_yz_av + L_p * T_cp_yz_av_cell) / (
            L_n + L_x + L_p
        )

        T_dict = {
            "cell": T,
            "negative current collector": T_cn,
            "positive current collector": T_cp,
            "volume-averaged cell": T_vol_av,
        }

        variables = {
            "Negative current collector temperature [K]": T_cn,
            "Positive current collector temperature [K]": T_cp,
            "Volume-averaged cell temperature [K]": T_vol_av,
            "X-averaged cell temperature [K]": T_yz_av,
            "Y-averaged cell temperature [K]": pybamm.Integral(T, y) / self.param.L_y,
            "Z-averaged cell temperature [K]": pybamm.Integral(T, z) / self.param.L_z,
        }

        variables.update(
            {
                "Negative current collector temperature [C]": T_cn - 273.15,
                "Positive current collector temperature [C]": T_cp - 273.15,
            }
        )

        variables = self._get_standard_fundamental_variables(T_dict)
        return variables

    def get_coupled_variables(self, variables):
        """Define coupled variables that depend on the fundamental variables."""
        variables.update(self._get_standard_coupled_variables(variables))
        return variables

    def set_rhs(self, variables):
        """Define the right-hand side of the PDE."""
        # Get key variables
        T = variables["Cell temperature [K]"]
        T_cn = variables["Negative current collector temperature [K]"]
        T_cp = variables["Positive current collector temperature [K]"]
        # T_amb = variables["Ambient temperature [K]"]

        # Get material properties and heat source
        Q = variables["Total heating [W.m-3]"]
        Q_cn = variables["Negative current collector Ohmic heating [W.m-3]"]
        Q_cp = variables["Positive current collector Ohmic heating [W.m-3]"]

        lambda_ = variables["Cell thermal conductivity [W.m-1.K-1]"]
        rho_c_p = variables["Cell volumetric heat capacity [J.K-1.m-3]"]
        lambda_cn = variables[
            "Negative current collector thermal conductivity [W.m-1.K-1]"
        ]
        rho_c_p_cn = variables[
            "Negative current collector volumetric heat capacity [J.K-1.m-3]"
        ]
        lambda_cp = variables[
            "Positive current collector thermal conductivity [W.m-1.K-1]"
        ]
        rho_c_p_cp = variables[
            "Positive current collector volumetric heat capacity [J.K-1.m-3]"
        ]

        self.rhs = {
            # Cell temperature: diffusion + heat source
            T: (pybamm.div(lambda_ * pybamm.grad(T)) + Q) / rho_c_p,
            # Negative current collector: diffusion + heat source
            T_cn: (pybamm.div(lambda_cn * pybamm.grad(T_cn)) + Q_cn) / rho_c_p_cn,
            # Positive current collector: diffusion + heat source
            T_cp: (pybamm.div(lambda_cp * pybamm.grad(T_cp)) + Q_cp) / rho_c_p_cp,
        }

    def set_boundary_conditions(self, variables):
        """Define the boundary conditions for the PDE."""
        # Get key variables
        T = variables["Cell temperature [K]"]
        T_cn = variables["Negative current collector temperature [K]"]
        T_cp = variables["Positive current collector temperature [K]"]
        T_amb = variables["Ambient temperature [K]"]

        # Get thermal conductivities
        lambda_ = variables["Cell thermal conductivity [W.m-1.K-1]"]
        lambda_cn = variables[
            "Negative current collector thermal conductivity [W.m-1.K-1]"
        ]
        lambda_cp = variables[
            "Positive current collector thermal conductivity [W.m-1.K-1]"
        ]

        # Get heat transfer coefficients
        h_edge = self.param.h_edge
        h_tab_n = self.param.n.h_tab
        h_tab_p = self.param.p.h_tab

        T_cn_right = pybamm.boundary_value(T_cn, "right")
        T_cp_left = pybamm.boundary_value(T_cp, "left")
        T_left = pybamm.boundary_value(T, "left")
        T_right = pybamm.boundary_value(T, "right")

        geometry = self._geometry.get("cell", {}).get("type", "rectangular")

        if geometry == "rectangular":
            self.boundary_conditions = {
                # Cell temperature
                T: {
                    # x-direction boundaries couple with current collectors
                    # Use boundary values to ensure domain compatibility
                    "left": (T_cn_right, "Dirichlet"),
                    "right": (T_cp_left, "Dirichlet"),
                    # y-direction and z-direction boundaries have convective cooling
                    "front": (h_edge, lambda_, h_edge * T_amb, "Robin"),
                    "back": (h_edge, lambda_, h_edge * T_amb, "Robin"),
                    "bottom": (h_edge, lambda_, h_edge * T_amb, "Robin"),
                    "top": (h_edge, lambda_, h_edge * T_amb, "Robin"),
                },
                # Negative current collector temperature
                T_cn: {
                    # x-direction boundary couples with cell
                    # Use boundary values to ensure domain compatibility
                    "right": (T_left, "Dirichlet"),
                    # External boundary has convective cooling
                    "left": (h_tab_n, lambda_cn, h_tab_n * T_amb, "Robin"),
                    # y-direction and z-direction boundaries have convective cooling
                    "front": (h_edge, lambda_cn, h_edge * T_amb, "Robin"),
                    "back": (h_edge, lambda_cn, h_edge * T_amb, "Robin"),
                    "bottom": (h_edge, lambda_cn, h_edge * T_amb, "Robin"),
                    "top": (h_edge, lambda_cn, h_edge * T_amb, "Robin"),
                },
                T_cp: {
                    # x-direction boundary couples with cell
                    # Use boundary values to ensure domain compatibility
                    "left": (T_right, "Dirichlet"),
                    # External boundary has convective cooling
                    "right": (h_tab_p, lambda_cp, h_tab_p * T_amb, "Robin"),
                    # y-direction and z-direction boundaries have convective cooling
                    "front": (h_edge, lambda_cp, h_edge * T_amb, "Robin"),
                    "back": (h_edge, lambda_cp, h_edge * T_amb, "Robin"),
                    "bottom": (h_edge, lambda_cp, h_edge * T_amb, "Robin"),
                    "top": (h_edge, lambda_cp, h_edge * T_amb, "Robin"),
                },
            }
        else:
            self.boundary_conditions = {
                T: {
                    "left": (T_cn_right, "Dirichlet"),
                    "right": (T_cp_left, "Dirichlet"),
                    "front": (h_edge, lambda_, h_edge * T_amb, "Robin"),
                    "back": (h_edge, lambda_, h_edge * T_amb, "Robin"),
                    "bottom": (h_edge, lambda_, h_edge * T_amb, "Robin"),
                    "top": (h_edge, lambda_, h_edge * T_amb, "Robin"),
                },
                T_cn: {
                    "right": (T_left, "Dirichlet"),
                    "left": (h_tab_n, lambda_cn, h_tab_n * T_amb, "Robin"),
                    "front": (h_edge, lambda_cn, h_edge * T_amb, "Robin"),
                    "back": (h_edge, lambda_cn, h_edge * T_amb, "Robin"),
                    "bottom": (h_edge, lambda_cn, h_edge * T_amb, "Robin"),
                    "top": (h_edge, lambda_cn, h_edge * T_amb, "Robin"),
                },
                T_cp: {
                    "left": (T_right, "Dirichlet"),
                    "right": (h_tab_p, lambda_cp, h_tab_p * T_amb, "Robin"),
                    "front": (h_edge, lambda_cp, h_edge * T_amb, "Robin"),
                    "back": (h_edge, lambda_cp, h_edge * T_amb, "Robin"),
                    "bottom": (h_edge, lambda_cp, h_edge * T_amb, "Robin"),
                    "top": (h_edge, lambda_cp, h_edge * T_amb, "Robin"),
                },
            }

    def set_initial_conditions(self, variables):
        """Define the initial conditions for the PDE."""
        T = variables["Cell temperature [K]"]
        T_cn = variables["Negative current collector temperature [K]"]
        T_cp = variables["Positive current collector temperature [K]"]

        T_init = self.param.T_init

        self.initial_conditions = {
            T: T_init,
            T_cn: T_init,
            T_cp: T_init,
        }
