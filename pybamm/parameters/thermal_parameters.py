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
        # self._set_dimensionless_parameters()

    def _set_dimensional_parameters(self):
        """Defines the dimensional parameters"""
        for domain in self.domain_params.values():
            domain._set_dimensional_parameters()

        # Reference temperature
        self.T_ref = pybamm.Parameter("Reference temperature [K]")

        # Cooling coefficient
        self.h_edge = pybamm.Parameter("Edge heat transfer coefficient [W.m-2.K-1]")
        self.h_total = pybamm.Parameter("Total heat transfer coefficient [W.m-2.K-1]")

        # Initial temperature
        self.T_init = pybamm.Parameter("Initial temperature [K]")

    def T_amb(self, t):
        """Dimensional ambient temperature"""
        return pybamm.FunctionParameter("Ambient temperature [K]", {"Time [s]": t})

    def rho_eff(self, T):
        """Effective volumetric heat capacity [J.m-3.K-1]"""
        return (
            self.n.rho_cc(T) * self.n.c_p_cc(T) * self.geo.n.L_cc
            + self.n.rho(T) * self.n.c_p(T) * self.geo.n.L
            + self.s.rho(T) * self.s.c_p(T) * self.geo.s.L
            + self.p.rho(T) * self.p.c_p(T) * self.geo.p.L
            + self.p.rho_cc(T) * self.p.c_p_cc(T) * self.geo.p.L_cc
        ) / self.geo.L

    def lambda_eff(self, T):
        """Effective thermal conductivity [W.m-1.K-1]"""
        return (
            self.n.lambda_cc(T) * self.geo.n.L_cc
            + self.n.lambda_(T) * self.geo.n.L
            + self.s.lambda_(T) * self.geo.s.L
            + self.p.lambda_(T) * self.geo.p.L
            + self.p.lambda_cc(T) * self.geo.p.L_cc
        ) / self.geo.L

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


class DomainThermalParameters(BaseParameters):
    def __init__(self, domain, main_param):
        self.domain = domain
        self.main_param = main_param

    def _set_dimensional_parameters(self):
        Domain = self.domain.capitalize()
        self.h_cc = pybamm.Parameter(
            f"{Domain} current collector surface heat transfer coefficient "
            "[W.m-2.K-1]"
        )
        self.h_tab = pybamm.Parameter(
            f"{Domain} tab heat transfer coefficient [W.m-2.K-1]"
        )

    def c_p(self, T):
        """Electrode specific heat capacity [J.kg-1.K-1]"""
        inputs = {"Temperature [K]": T}
        if self.domain == "separator":
            name = "Separator specific heat capacity [J.kg-1.K-1]"
        else:
            Domain = self.domain.capitalize()
            name = f"{Domain} electrode specific heat capacity [J.kg-1.K-1]"
        return pybamm.FunctionParameter(name, inputs)

    def c_p_cc(self, T):
        """Current collector specific heat capacity [J.kg-1.K-1]"""
        inputs = {"Temperature [K]": T}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} current collector specific heat capacity [J.kg-1.K-1]",
            inputs,
        )

    def lambda_(self, T):
        """Electrode thermal conductivity [W.m-1.K-1]"""
        inputs = {"Temperature [K]": T}
        if self.domain == "separator":
            name = "Separator thermal conductivity [W.m-1.K-1]"
        else:
            Domain = self.domain.capitalize()
            name = f"{Domain} electrode thermal conductivity [W.m-1.K-1]"
        return pybamm.FunctionParameter(name, inputs)

    def lambda_cc(self, T):
        """Current collector thermal conductivity [W.m-1.K-1]"""
        inputs = {"Temperature [K]": T}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} current collector thermal conductivity [W.m-1.K-1]", inputs
        )

    def rho(self, T):
        """Electrode density [kg.m-3]"""
        inputs = {"Temperature [K]": T}
        if self.domain == "separator":
            name = "Separator density [kg.m-3]"
        else:
            Domain = self.domain.capitalize()
            name = f"{Domain} electrode density [kg.m-3]"
        return pybamm.FunctionParameter(name, inputs)

    def rho_cc(self, T):
        """Current collector density [kg.m-3]"""
        inputs = {"Temperature [K]": T}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{Domain} current collector density [kg.m-3]", inputs
        )


thermal_parameters = ThermalParameters()
