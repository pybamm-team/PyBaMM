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
        self.geo = pybamm.geometric_parameters

        # Set parameters
        self._set_dimensional_parameters()
        self._set_dimensionless_parameters()

    def _set_dimensional_parameters(self):
        "Defines the dimensional parameters"

        # Reference temperature
        self.T_ref = pybamm.Parameter("Reference temperature [K]")

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
        self.tau_th_yz = (
            self.rho_eff_dim(self.T_ref) * (self.geo.L_z ** 2) / self.lambda_eff_dim
        )

    def T_amb_dim(self, t):
        "Dimensional ambient temperature"
        return pybamm.FunctionParameter("Ambient temperature [K]", {"Times [s]": t})

    def rho_cn_dim(self, T):
        "Negative current collector density [kg.m-3]"
        inputs = {
            "Temperature [K]": self.Delta_T * T + self.T_ref,
        }
        return pybamm.FunctionParameter(
            "Negative current collector density [kg.m-3]", inputs
        )

    def rho_n_dim(self, T):
        "Negative electrode density [kg.m-3]"
        inputs = {
            "Temperature [K]": self.Delta_T * T + self.T_ref,
        }
        return pybamm.FunctionParameter("Negative electrode density [kg.m-3]", inputs)

    def rho_s_dim(self, T):
        "Separator density [kg.m-3]"
        inputs = {
            "Temperature [K]": self.Delta_T * T + self.T_ref,
        }
        return pybamm.FunctionParameter("Separator density [kg.m-3]", inputs)

    def rho_p_dim(self, T):
        "Positive electrode density [kg.m-3]"
        inputs = {
            "Temperature [K]": self.Delta_T * T + self.T_ref,
        }
        return pybamm.FunctionParameter("Positive electrode density [kg.m-3]", inputs)

    def rho_cp_dim(self, T):
        "Positive current collector density [kg.m-3]"
        inputs = {
            "Temperature [K]": self.Delta_T * T + self.T_ref,
        }
        return pybamm.FunctionParameter(
            "Positive current collector density [kg.m-3]", inputs
        )

    def rho_eff_dim(self, T):
        "Effective volumetric heat capacity"
        return (
            self.rho_cn_dim(T) * self.c_p_cn_dim * self.geo.L_cn
            + self.rho_n_dim(T) * self.c_p_n_dim * self.geo.L_n
            + self.rho_s_dim(T) * self.c_p_s_dim * self.geo.L_s
            + self.rho_p_dim(T) * self.c_p_p_dim * self.geo.L_p
            + self.rho_cp_dim(T) * self.c_p_cp_dim * self.geo.L_cp
        ) / self.geo.L

    def _set_dimensionless_parameters(self):
        "Defines the dimensionless parameters"

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

    def rho_cn(self, T):
        "Dimensionless negative current collector density [kg.m-3]"
        return self.rho_cn_dim(T) * self.c_p_cn_dim / self.rho_eff_dim(T)

    def rho_n(self, T):
        "Dimensionless negative electrode density"
        return self.rho_n_dim(T) * self.c_p_n_dim / self.rho_eff_dim(T)

    def rho_s(self, T):
        "Dimensionless separator density"
        return self.rho_s_dim(T) * self.c_p_s_dim / self.rho_eff_dim(T)

    def rho_p(self, T):
        "Dimensionless positive electrode density"
        return self.rho_p_dim(T) * self.c_p_p_dim / self.rho_eff_dim(T)

    def rho_cp(self, T):
        "Dimensionless positive current collector density"
        return self.rho_cp_dim(T) * self.c_p_cp_dim / self.rho_eff_dim(T)

    def rho_k(self, T):
        "Concatenated dimensionless density"
        return pybamm.Concatenation(
            pybamm.FullBroadcast(
                self.rho_n(T), ["negative electrode"], "current collector"
            ),
            pybamm.FullBroadcast(self.rho_s(T), ["separator"], "current collector"),
            pybamm.FullBroadcast(
                self.rho_p(T), ["positive electrode"], "current collector"
            ),
        )


thermal_parameters = ThermalParameters()
