#
# Standard thermal parameters
#
import pybamm
from .base_parameters import BaseParameters


class ThermalParameters(BaseParameters):
    """
    Standard thermal parameters
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
        self._set_parameters()

    def _set_parameters(self):
        """Defines the dimensional parameters"""
        for domain in self.domain_params.values():
            domain._set_parameters()

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

    def rho_c_p_eff(self, T):
        """Effective volumetric heat capacity [J.m-3.K-1]"""
        return (
            self.n.rho_c_p_cc(T) * self.geo.n.L_cc
            + self.n.rho_c_p(T) * self.geo.n.L
            + self.s.rho_c_p(T) * self.geo.s.L
            + self.p.rho_c_p(T) * self.geo.p.L
            + self.p.rho_c_p_cc(T) * self.geo.p.L_cc
        ) / self.geo.L


class DomainThermalParameters(BaseParameters):
    def __init__(self, domain, main_param):
        self.domain = domain
        self.main_param = main_param

    def _set_parameters(self):
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

    def rho_c_p(self, T):
        return self.rho(T) * self.c_p(T)

    def rho_c_p_cc(self, T):
        return self.rho_cc(T) * self.c_p_cc(T)


thermal_parameters = ThermalParameters()
