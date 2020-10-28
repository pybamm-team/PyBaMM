#
# Standard thermal parameters
#
import pybamm


class ThermalParameters:
    """
    Standard thermal parameters

    Layout:
        1. Dimensional Parameters
        2. Dimensional Functions
        3. Dimensionless Parameters
        4. Dimensionless Functions
    """

    def __init__(self):

        # Get geometric parameters
        self.geo = pybamm.GeometricParameters()

        # Set parameters
        self._set_dimensional_parameters()
        self._set_dimensionless_parameters()

    def _set_dimensional_parameters(self):
        "Defines the dimensional parameters"

        # Reference temperature
        self.T_ref = pybamm.Parameter("Reference temperature [K]")

        # Density
        self.rho_cn_dim = pybamm.Parameter(
            "Negative current collector density [kg.m-3]"
        )
        self.rho_n_dim = pybamm.Parameter("Negative electrode density [kg.m-3]")
        self.rho_s_dim = pybamm.Parameter("Separator density [kg.m-3]")
        self.rho_p_dim = pybamm.Parameter("Positive electrode density [kg.m-3]")
        self.rho_cp_dim = pybamm.Parameter(
            "Positive current collector density [kg.m-3]"
        )

        # Specific heat capacity
        self.c_p_cn_dim = pybamm.Parameter(
            "Negative current collector specific heat capacity [J.kg-1.K-1]"
        )
        self.c_p_n_dim = pybamm.Parameter(
            "Negative electrode specific heat capacity [J.kg-1.K-1]"
        )
        self.c_p_s_dim = pybamm.Parameter(
            "Separator specific heat capacity [J.kg-1.K-1]"
        )
        self.c_p_p_dim = pybamm.Parameter(
            "Negative electrode specific heat capacity [J.kg-1.K-1]"
        )
        self.c_p_cp_dim = pybamm.Parameter(
            "Positive current collector specific heat capacity [J.kg-1.K-1]"
        )

        # Thermal conductivity
        self.lambda_cn_dim = pybamm.Parameter(
            "Negative current collector thermal conductivity [W.m-1.K-1]"
        )
        self.lambda_n_dim = pybamm.Parameter(
            "Negative electrode thermal conductivity [W.m-1.K-1]"
        )
        self.lambda_s_dim = pybamm.Parameter(
            "Separator thermal conductivity [W.m-1.K-1]"
        )
        self.lambda_p_dim = pybamm.Parameter(
            "Positive electrode thermal conductivity [W.m-1.K-1]"
        )
        self.lambda_cp_dim = pybamm.Parameter(
            "Positive current collector thermal conductivity [W.m-1.K-1]"
        )

        # Effective volumetic heat capacity
        self.rho_eff_dim = (
            self.rho_cn_dim * self.c_p_cn_dim * self.geo.L_cn
            + self.rho_n_dim * self.c_p_n_dim * self.geo.L_n
            + self.rho_s_dim * self.c_p_s_dim * self.geo.L_s
            + self.rho_p_dim * self.c_p_p_dim * self.geo.L_p
            + self.rho_cp_dim * self.c_p_cp_dim * self.geo.L_cp
        ) / self.geo.L

        # Effective thermal conductivity
        self.lambda_eff_dim = (
            self.lambda_cn_dim * self.geo.L_cn
            + self.lambda_n_dim * self.geo.L_n
            + self.lambda_s_dim * self.geo.L_s
            + self.lambda_p_dim * self.geo.L_p
            + self.lambda_cp_dim * self.geo.L_cp
        ) / self.geo.L

        # Cooling coefficient
        self.h_cn_dim = pybamm.Parameter(
            "Negative current collector surface heat transfer coefficient [W.m-2.K-1]"
        )
        self.h_cp_dim = pybamm.Parameter(
            "Positive current collector surface heat transfer coefficient [W.m-2.K-1]"
        )
        self.h_tab_n_dim = pybamm.Parameter(
            "Negative tab heat transfer coefficient [W.m-2.K-1]"
        )
        self.h_tab_p_dim = pybamm.Parameter(
            "Positive tab heat transfer coefficient [W.m-2.K-1]"
        )
        self.h_edge_dim = pybamm.Parameter("Edge heat transfer coefficient [W.m-2.K-1]")
        self.h_total_dim = pybamm.Parameter(
            "Total heat transfer coefficient [W.m-2.K-1]"
        )

        # Typical temperature rise
        self.Delta_T = pybamm.Scalar(1)

        # Initial temperature
        self.T_init_dim = pybamm.Parameter("Initial temperature [K]")

        # Planar (y,z) thermal diffusion timescale
        self.tau_th_yz = self.rho_eff_dim * (self.geo.L_z ** 2) / self.lambda_eff_dim

    def T_amb_dim(self, t):
        "Dimensional ambient temperature"
        return pybamm.FunctionParameter("Ambient temperature [K]", {"Times [s]": t})

    def _set_dimensionless_parameters(self):
        "Defines the dimensionless parameters"

        # Density
        self.rho_cn = self.rho_cn_dim * self.c_p_cn_dim / self.rho_eff_dim
        self.rho_n = self.rho_n_dim * self.c_p_n_dim / self.rho_eff_dim
        self.rho_s = self.rho_s_dim * self.c_p_s_dim / self.rho_eff_dim
        self.rho_p = self.rho_p_dim * self.c_p_p_dim / self.rho_eff_dim
        self.rho_cp = self.rho_cp_dim * self.c_p_cp_dim / self.rho_eff_dim

        self.rho_k = pybamm.Concatenation(
            pybamm.FullBroadcast(
                self.rho_n, ["negative electrode"], "current collector"
            ),
            pybamm.FullBroadcast(self.rho_s, ["separator"], "current collector"),
            pybamm.FullBroadcast(
                self.rho_p, ["positive electrode"], "current collector"
            ),
        )

        # Thermal conductivity
        self.lambda_cn = self.lambda_cn_dim / self.lambda_eff_dim
        self.lambda_n = self.lambda_n_dim / self.lambda_eff_dim
        self.lambda_s = self.lambda_s_dim / self.lambda_eff_dim
        self.lambda_p = self.lambda_p_dim / self.lambda_eff_dim
        self.lambda_cp = self.lambda_cp_dim / self.lambda_eff_dim

        self.lambda_k = pybamm.Concatenation(
            pybamm.FullBroadcast(
                self.lambda_n, ["negative electrode"], "current collector"
            ),
            pybamm.FullBroadcast(self.lambda_s, ["separator"], "current collector"),
            pybamm.FullBroadcast(
                self.lambda_p, ["positive electrode"], "current collector"
            ),
        )

        # Relative temperature rise
        self.Theta = self.Delta_T / self.T_ref

        # Cooling coefficient
        self.h_cn = self.h_cn_dim * self.geo.L_x / self.lambda_eff_dim
        self.h_cp = self.h_cp_dim * self.geo.L_x / self.lambda_eff_dim
        self.h_tab_n = self.h_tab_n_dim * self.geo.L_x / self.lambda_eff_dim
        self.h_tab_p = self.h_tab_p_dim * self.geo.L_x / self.lambda_eff_dim
        self.h_edge = self.h_edge_dim * self.geo.L_x / self.lambda_eff_dim
        self.h_total = self.h_total_dim * self.geo.L_x / self.lambda_eff_dim

        # Initial temperature
        self.T_init = (self.T_init_dim - self.T_ref) / self.Delta_T

    def T_amb(self, t):
        "Dimensionless ambient temperature"
        return (self.T_amb_dim(t) - self.T_ref) / self.Delta_T
