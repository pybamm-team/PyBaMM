#
# Class for Bruggemantransport_efficiency
#
import pybamm
from .base_transport_efficiency import BaseModel


class GeneralTransportEfficiency(BaseModel):
    """Submodel for Bruggeman transport_efficiency

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    component : str
        The material for the model ('electrolyte' or 'electrode').
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, component, options=None):
        super().__init__(param, component, options=options)

    def _tortuosity_factor_model(self, eps_k):
        if self.options["transport efficiency"] == "ordered packing":
            pybamm.citations.register("akanni1987effective")
            tor_k = (3 - eps_k)*0.5
        elif self.options["transport efficiency"] == "hyperbola of revolution":
            pybamm.citations.register("petersen1958diffusion")
            tor_k = 2 - eps_k
        elif self.options["transport efficiency"] == "overlapping spheres":
            pybamm.citations.register("weissberg1963effective")
            tor_k = 1 - pybamm.Log(eps_k*0.5)
        elif self.options["transport efficiency"] == "random overlapping cylinders":
            pybamm.citations.register("tomadakis1993transport")
            tor_k = 1 - pybamm.Log(eps_k)
        elif self.options["transport efficiency"] == "heterogeneous catalyst":
            pybamm.citations.register("beeckman1990mathematical")
            tor_k = eps_k / (1 - (1- eps_k)**(1/3))
        elif self.options["transport efficiency"] == "cation-exchange membrane":
            pybamm.citations.register("mackie1955diffusion")
            tor_k = ((2 - eps_k) / eps_k)**2

        return tor_k

    def get_coupled_variables(self, variables):
        if self.component == "Electrolyte":
            tor_dict = {}
            for domain in self.options.whole_cell_domains:
                Domain = domain.capitalize()
                eps_k = variables[f"{Domain} porosity"]
                if self.options["transport efficiency"] == "Bruggeman":
                    pybamm.citations.register("bruggeman1935berechnung")
                    b_k = self.param.domain_params[domain.split()[0]].b_e
                    tor_k = eps_k**b_k
                elif self.options["transport efficiency"] == "tortuosity factor":
                    tau_k = self.param.domain_params[domain.split()[0]].tau_e
                    tor_k = eps_k / tau_k
                else:
                    tor_k = self._tortuosity_factor_model(eps_k)
                tor_dict[domain] = tor_k
        elif self.component == "Electrode":
            tor_dict = {}
            for domain in self.options.whole_cell_domains:
                if domain == "separator":
                    tor_k = pybamm.FullBroadcast(0, "separator", "current collector")
                else:
                    Domain = domain.capitalize()
                    phi_k = (1 - variables[f"{Domain} porosity"])
                    if self.options["transport efficiency"] == "Bruggeman":
                        pybamm.citations.register("bruggeman1935berechnung")
                        b_k = self.param.domain_params[domain.split()[0]].b_s
                        tor_k = phi_k**b_k
                    elif self.options["transport efficiency"] == "tortuosity factor":
                        tau_k = self.param.domain_params[domain.split()[0]].tau_s
                        tor_k = phi_k / tau_k
                    else:
                        tor_k = self._tortuosity_factor_model(phi_k)
                tor_dict[domain] = tor_k
        variables.update(self._get_standard_transport_efficiency_variables(tor_dict))

        return variables
