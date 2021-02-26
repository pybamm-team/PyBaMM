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
        """Defines the dimensional parameters"""

        # Reference temperature
        self.T_ref = pybamm.Parameter("Reference temperature [K]")

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
            self.rho_eff_dim(self.T_ref)
            * (self.geo.L_z ** 2)
            / self.lambda_eff_dim(self.T_ref)
        )

    def T_amb_dim(self, t):
        """Dimensional ambient temperature"""
        return pybamm.FunctionParameter("Ambient temperature [K]", {"Times [s]": t})

    def rho_cn_dim(self, T):
        """Negative current collector density [kg.m-3]"""
        inputs = {
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Negative current collector density [kg.m-3]", inputs
        )

    def rho_n_dim(self, T):
        """Negative electrode density [kg.m-3]"""
        inputs = {
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter("Negative electrode density [kg.m-3]", inputs)

    def rho_s_dim(self, T):
        """Separator density [kg.m-3]"""
        inputs = {
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter("Separator density [kg.m-3]", inputs)

    def rho_p_dim(self, T):
        """Positive electrode density [kg.m-3]"""
        inputs = {
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter("Positive electrode density [kg.m-3]", inputs)

    def rho_cp_dim(self, T):
        """Positive current collector density [kg.m-3]"""
        inputs = {
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Positive current collector density [kg.m-3]", inputs
        )

    def rho_eff_dim(self, T):
        """Effective volumetric heat capacity [J.m-3.K-1]"""
        return (
            self.rho_cn_dim(T) * self.c_p_cn_dim(T) * self.geo.L_cn
            + self.rho_n_dim(T) * self.c_p_n_dim(T) * self.geo.L_n
            + self.rho_s_dim(T) * self.c_p_s_dim(T) * self.geo.L_s
            + self.rho_p_dim(T) * self.c_p_p_dim(T) * self.geo.L_p
            + self.rho_cp_dim(T) * self.c_p_cp_dim(T) * self.geo.L_cp
        ) / self.geo.L

    def c_p_cn_dim(self, T):
        """Negative current collector specific heat capacity [J.kg-1.K-1]"""
        inputs = {
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Negative current collector specific heat capacity [J.kg-1.K-1]", inputs
        )

    def c_p_n_dim(self, T):
        """Negative electrode specific heat capacity [J.kg-1.K-1]"""
        inputs = {
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Negative electrode specific heat capacity [J.kg-1.K-1]", inputs
        )

    def c_p_s_dim(self, T):
        """Separator specific heat capacity [J.kg-1.K-1]"""
        inputs = {
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Separator specific heat capacity [J.kg-1.K-1]", inputs
        )

    def c_p_p_dim(self, T):
        """Positive electrode specific heat capacity [J.kg-1.K-1]"""
        inputs = {
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Positive electrode specific heat capacity [J.kg-1.K-1]", inputs
        )

    def c_p_cp_dim(self, T):
        """Positive current collector specific heat capacity [J.kg-1.K-1]"""
        inputs = {
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Positive current collector specific heat capacity [J.kg-1.K-1]", inputs
        )

    def lambda_cn_dim(self, T):
        """Negative current collector thermal conductivity [W.m-1.K-1]"""
        inputs = {
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Negative current collector thermal conductivity [W.m-1.K-1]", inputs
        )

    def lambda_n_dim(self, T):
        """Negative electrode thermal conductivity [W.m-1.K-1]"""
        inputs = {
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Negative electrode thermal conductivity [W.m-1.K-1]", inputs
        )

    def lambda_s_dim(self, T):
        """Separator thermal conductivity [W.m-1.K-1]"""
        inputs = {
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Separator thermal conductivity [W.m-1.K-1]", inputs
        )

    def lambda_p_dim(self, T):
        """Positive electrode thermal conductivity [W.m-1.K-1]"""
        inputs = {
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Positive electrode thermal conductivity [W.m-1.K-1]", inputs
        )

    def lambda_cp_dim(self, T):
        """Positive current collector thermal conductivity [W.m-1.K-1]"""
        inputs = {
            "Temperature [K]": T,
        }
        return pybamm.FunctionParameter(
            "Positive current collector thermal conductivity [W.m-1.K-1]", inputs
        )

    def lambda_eff_dim(self, T):
        """Effective thermal conductivity [W.m-1.K-1]"""
        return (
            self.lambda_cn_dim(T) * self.geo.L_cn
            + self.lambda_n_dim(T) * self.geo.L_n
            + self.lambda_s_dim(T) * self.geo.L_s
            + self.lambda_p_dim(T) * self.geo.L_p
            + self.lambda_cp_dim(T) * self.geo.L_cp
        ) / self.geo.L

    def _set_dimensionless_parameters(self):
        """Defines the dimensionless parameters"""

        # Relative temperature rise
        self.Theta = self.Delta_T / self.T_ref

        # Cooling coefficient
        self.h_cn = self.h_cn_dim * self.geo.L_x / self.lambda_eff_dim(self.T_ref)
        self.h_cp = self.h_cp_dim * self.geo.L_x / self.lambda_eff_dim(self.T_ref)
        self.h_tab_n = self.h_tab_n_dim * self.geo.L_x / self.lambda_eff_dim(self.T_ref)
        self.h_tab_p = self.h_tab_p_dim * self.geo.L_x / self.lambda_eff_dim(self.T_ref)
        self.h_edge = self.h_edge_dim * self.geo.L_x / self.lambda_eff_dim(self.T_ref)
        self.h_total = self.h_total_dim * self.geo.L_x / self.lambda_eff_dim(self.T_ref)

        # Initial temperature
        self.T_init = (self.T_init_dim - self.T_ref) / self.Delta_T

    def T_amb(self, t):
        """Dimensionless ambient temperature"""
        return (self.T_amb_dim(t) - self.T_ref) / self.Delta_T

    def rho_cn(self, T):
        """Dimensionless negative current collector density"""
        T_dim = self.Delta_T * T + self.T_ref
        return (
            self.rho_cn_dim(T_dim)
            * self.c_p_cn_dim(T_dim)
            / self.rho_eff_dim(self.T_ref)
        )

    def rho_n(self, T):
        """Dimensionless negative electrode density"""
        T_dim = self.Delta_T * T + self.T_ref
        return (
            self.rho_n_dim(T_dim) * self.c_p_n_dim(T_dim) / self.rho_eff_dim(self.T_ref)
        )

    def rho_s(self, T):
        """Dimensionless separator density"""
        T_dim = self.Delta_T * T + self.T_ref
        return (
            self.rho_s_dim(T_dim) * self.c_p_s_dim(T_dim) / self.rho_eff_dim(self.T_ref)
        )

    def rho_p(self, T):
        """Dimensionless positive electrode density"""
        T_dim = self.Delta_T * T + self.T_ref
        return (
            self.rho_p_dim(T_dim) * self.c_p_p_dim(T_dim) / self.rho_eff_dim(self.T_ref)
        )

    def rho_cp(self, T):
        """Dimensionless positive current collector density"""
        T_dim = self.Delta_T * T + self.T_ref
        return (
            self.rho_cp_dim(T_dim)
            * self.c_p_cp_dim(T_dim)
            / self.rho_eff_dim(self.T_ref)
        )

    def lambda_cn(self, T):
        """Dimensionless negative current collector thermal conductivity"""
        T_dim = self.Delta_T * T + self.T_ref
        return self.lambda_cn_dim(T_dim) / self.lambda_eff_dim(self.T_ref)

    def lambda_n(self, T):
        """Dimensionless negative electrode thermal conductivity"""
        T_dim = self.Delta_T * T + self.T_ref
        return self.lambda_n_dim(T_dim) / self.lambda_eff_dim(self.T_ref)

    def lambda_s(self, T):
        """Dimensionless separator thermal conductivity"""
        T_dim = self.Delta_T * T + self.T_ref
        return self.lambda_s_dim(T_dim) / self.lambda_eff_dim(self.T_ref)

    def lambda_p(self, T):
        """Dimensionless positive electrode thermal conductivity"""
        T_dim = self.Delta_T * T + self.T_ref
        return self.lambda_p_dim(T_dim) / self.lambda_eff_dim(self.T_ref)

    def lambda_cp(self, T):
        """Dimensionless positive current collector thermal conductivity"""
        T_dim = self.Delta_T * T + self.T_ref
        return self.lambda_cp_dim(T_dim) / self.lambda_eff_dim(self.T_ref)


thermal_parameters = ThermalParameters()
