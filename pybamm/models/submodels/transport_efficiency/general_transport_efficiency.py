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

    def _transport_efficiency_models(self, eps_k):
        if self.options["transport efficiency"] == "log square root":
            tor_k = 1 - pybamm.Log(eps_k**0.5)
        elif self.options["transport efficiency"] == "log":
            tor_k = 1 - pybamm.Log(eps_k)
        elif self.options["transport efficiency"] == "linear":
            tor_k = 2 - eps_k
        elif self.options["transport efficiency"] == "half volume fraction":
            tor_k = 1.5 - eps_k / 2

        return tor_k

    def get_coupled_variables(self, variables):
        if self.component == "Electrolyte":
            tor_dict = {}
            for domain in self.options.whole_cell_domains:
                Domain = domain.capitalize()
                eps_k = variables[f"{Domain} porosity"]
                if self.options["transport efficiency"] == "Bruggeman":
                    b_k = self.param.domain_params[domain.split()[0]].b_e
                    tor_k = eps_k**b_k
                elif self.options["transport efficiency"] == "tortuosity factor":
                    tau_k = self.param.domain_params[domain.split()[0]].tau_e
                    tor_k = eps_k / tau_k
                else:
                    tor_k = self._transport_efficiency_models(eps_k)
                tor_dict[domain] = tor_k
        elif self.component == "Electrode":
            tor_dict = {}
            for domain in self.options.whole_cell_domains:
                if domain == "separator":
                    tor_k = pybamm.FullBroadcast(0, "separator", "current collector")
                else:
                    Domain = domain.capitalize()
                    eps_k = variables[f"{Domain} active material volume fraction"]
                    if self.options["transport efficiency"] == "Bruggeman":
                        b_k = self.param.domain_params[domain.split()[0]].b_s
                        tor_k = eps_k**b_k
                    elif self.options["transport efficiency"] == "tortuosity factor":
                        tau_k = self.param.domain_params[domain.split()[0]].tau_s
                        tor_k = eps_k / tau_k
                    else:
                        tor_k = self._transport_efficiency_models(eps_k)
                tor_dict[domain] = tor_k
        variables.update(self._get_standard_transport_efficiency_variables(tor_dict))

        return variables
