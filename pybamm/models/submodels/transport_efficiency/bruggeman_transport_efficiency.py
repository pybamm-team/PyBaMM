#
# Class for Bruggemantransport_efficiency
#
import pybamm
from .base_transport_efficiency import BaseModel


class Bruggeman(BaseModel):
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

    def get_coupled_variables(self, variables):
        if self.component == "Electrolyte":
            tor_dict = {}
            for domain in self.options.whole_cell_domains:
                Domain = domain.capitalize()
                eps_k = variables[f"{Domain} porosity"]
                b_k = self.param.domain_params[domain.split()[0]].b_e
                tor_dict[domain] = eps_k**b_k
        elif self.component == "Electrode":
            tor_dict = {}
            for domain in self.options.whole_cell_domains:
                if domain == "separator":
                    tor_k = pybamm.FullBroadcast(0, "separator", "current collector")
                else:
                    Domain = domain.capitalize()
                    eps_k = variables[f"{Domain} active material volume fraction"]
                    b_k = self.param.domain_params[domain.split()[0]].b_s
                    tor_k = eps_k**b_k
                tor_dict[domain] = tor_k

        variables.update(self._get_standard_transport_efficiency_variables(tor_dict))

        return variables
