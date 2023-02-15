#
# Standard thermal parameters
#
import pybamm
from .base_parameters import BaseParameters


class ThermalParameters(BaseParameters):
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

        self.n = DomainThermalParameters("negative", self)
        self.s = DomainThermalParameters("separator", self)
        self.p = DomainThermalParameters("positive", self)
        self.domain_params = {
            "negative": self.n,
            "separator": self.s,
            "positive": self.p,
        }

        # Set parameters
        self._set_dimensional_parameters()
        self._set_dimensionless_parameters()

    def _set_dimensional_parameters(self):
        """Defines the dimensional parameters"""
        for domain in self.domain_params.values():
            domain._set_dimensional_parameters()

        # Reference temperature
        self.T_ref = pybamm.Parameter("Reference temperature [K]")

        # Cooling coefficient
        self.h_edge_dim = pybamm.Parameter("Edge heat transfer coefficient [W.m-2.K-1]")
        self.h_total_dim = pybamm.Parameter(
            "Total heat transfer coefficient [W.m-2.K-1]"
        )

        # Typical temperature rise
        self.Delta_T = pybamm.Scalar(1)

        # Initial temperature
        self.T_init_dim = pybamm.Parameter("Initial temperature [K]")

        # References
        self.rho_eff_dim_ref = self.rho_eff_dim(self.T_ref)
        self.lambda_eff_dim_ref = self.lambda_eff_dim(self.T_ref)

        # Planar (y,z) thermal diffusion timescale
        self.tau_th_yz = (
            self.rho_eff_dim_ref * (self.geo.L_z**2) / self.lambda_eff_dim_ref
        )

    def T_amb_dim(self, t):
        """Dimensional ambient temperature"""
        return pybamm.FunctionParameter("Ambient temperature [K]", {"Time [s]": t})

    def rho_eff_dim(self, T):
        """Effective volumetric heat capacity [J.m-3.K-1]"""
        return (
            self.n.rho_cc_dim(T) * self.n.c_p_cc_dim(T) * self.geo.n.L_cc
            + self.n.rho_dim(T) * self.n.c_p_dim(T) * self.geo.n.L
            + self.s.rho_dim(T) * self.s.c_p_dim(T) * self.geo.s.L
            + self.p.rho_dim(T) * self.p.c_p_dim(T) * self.geo.p.L
            + self.p.rho_cc_dim(T) * self.p.c_p_cc_dim(T) * self.geo.p.L_cc
        ) / self.geo.L

    def lambda_eff_dim(self, T):
        """Effective thermal conductivity [W.m-1.K-1]"""
        return (
            self.n.lambda_cc_dim(T) * self.geo.n.L_cc
            + self.n.lambda_dim(T) * self.geo.n.L
            + self.s.lambda_dim(T) * self.geo.s.L
            + self.p.lambda_dim(T) * self.geo.p.L
            + self.p.lambda_cc_dim(T) * self.geo.p.L_cc
        ) / self.geo.L

    def _set_dimensionless_parameters(self):
        """Defines the dimensionless parameters"""
        for domain in self.domain_params.values():
            domain._set_dimensionless_parameters()

        # Relative temperature rise
        self.Theta = self.Delta_T / self.T_ref

        # Cooling coefficient
        self.h_edge = self.h_edge_dim * self.geo.L_x / self.lambda_eff_dim_ref
        self.h_total = self.h_total_dim * self.geo.L_x / self.lambda_eff_dim_ref

        # Initial temperature
        self.T_init = (self.T_init_dim - self.T_ref) / self.Delta_T

    def rho(self, T):
        """
        Dimensionless effective density, not to be confused with rho_eff_dim,
        which is the dimensional effective volumetric heat capacity
        """
        return (
            self.n.rho_cc(T) * self.geo.n.l_cc
            + self.n.rho(T) * self.geo.n.l
            + self.s.rho(T) * self.geo.s.l
            + self.p.rho(T) * self.geo.p.l
            + self.p.rho_cc(T) * self.geo.p.l_cc
        ) / self.geo.l

    def T_amb(self, t):
        """Dimensionless ambient temperature"""
        return (self.T_amb_dim(t) - self.T_ref) / self.Delta_T


class DomainThermalParameters(BaseParameters):
    def __init__(self, domain, main_param):
        self.domain = domain
        self.main_param = main_param

    def _set_dimensional_parameters(self):
        Domain = self.domain.capitalize()
        self.h_cc_dim = pybamm.Parameter(
            f"{Domain} current collector surface heat transfer coefficient "
            "[W.m-2.K-1]"
        )
        self.h_tab_dim = pybamm.Parameter(
            f"{Domain} tab heat transfer coefficient [W.m-2.K-1]"
        )

    def c_p_dim(self, T):
        """Electrode specific heat capacity [J.kg-1.K-1]"""
        inputs = {"Temperature [K]": T}
        if self.domain == "separator":
            name = "Separator specific heat capacity [J.kg-1.K-1]"
        else:
            Domain = self.domain.capitalize()
            name = f"{Domain} electrode specific heat capacity [J.kg-1.K-1]"
        return pybamm.FunctionParameter(name, inputs)

    def c_p_cc_dim(self, T):
        """Current collector specific heat capacity [J.kg-1.K-1]"""
        inputs = {"Temperature [K]": T}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} current collector specific heat capacity [J.kg-1.K-1]",
            inputs,
        )

    def lambda_dim(self, T):
        """Electrode thermal conductivity [W.m-1.K-1]"""
        inputs = {"Temperature [K]": T}
        if self.domain == "separator":
            name = "Separator thermal conductivity [W.m-1.K-1]"
        else:
            Domain = self.domain.capitalize()
            name = f"{Domain} electrode thermal conductivity [W.m-1.K-1]"
        return pybamm.FunctionParameter(name, inputs)

    def lambda_cc_dim(self, T):
        """Current collector thermal conductivity [W.m-1.K-1]"""
        inputs = {"Temperature [K]": T}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} current collector thermal conductivity [W.m-1.K-1]", inputs
        )

    def rho_dim(self, T):
        """Electrode density [kg.m-3]"""
        inputs = {"Temperature [K]": T}
        if self.domain == "separator":
            name = "Separator density [kg.m-3]"
        else:
            Domain = self.domain.capitalize()
            name = f"{Domain} electrode density [kg.m-3]"
        return pybamm.FunctionParameter(name, inputs)

    def rho_cc_dim(self, T):
        """Current collector density [kg.m-3]"""
        inputs = {"Temperature [K]": T}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} current collector density [kg.m-3]", inputs
        )

    def _set_dimensionless_parameters(self):
        main = self.main_param
        self.h_cc = self.h_cc_dim * main.geo.L_x / main.lambda_eff_dim_ref
        self.h_tab = self.h_tab_dim * main.geo.L_x / main.lambda_eff_dim_ref

    def rho(self, T):
        """Dimensionless electrode density"""
        T_dim = self.main_param.Delta_T * T + self.main_param.T_ref
        return (
            self.rho_dim(T_dim) * self.c_p_dim(T_dim) / self.main_param.rho_eff_dim_ref
        )

    def rho_cc(self, T):
        """Dimensionless current collector density"""
        T_dim = self.main_param.Delta_T * T + self.main_param.T_ref
        return (
            self.rho_cc_dim(T_dim)
            * self.c_p_cc_dim(T_dim)
            / self.main_param.rho_eff_dim_ref
        )

    def lambda_(self, T):  # cannot call a function "lambda"
        """Dimensionless electrode thermal conductivity"""
        T_dim = self.main_param.Delta_T * T + self.main_param.T_ref
        return self.lambda_dim(T_dim) / self.main_param.lambda_eff_dim_ref

    def lambda_cc(self, T):
        """Dimensionless current collector thermal conductivity"""
        T_dim = self.main_param.Delta_T * T + self.main_param.T_ref
        return self.lambda_cc_dim(T_dim) / self.main_param.lambda_eff_dim_ref


thermal_parameters = ThermalParameters()
