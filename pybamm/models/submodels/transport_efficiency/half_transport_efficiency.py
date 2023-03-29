#
# Class for Bruggemantransport_efficiency
#
import pybamm
from .base_transport_efficiency import BaseModel


class HalfTransfer(BaseModel):
    """Submodel for transport_efficiency = (3 - eps_k) / 2

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
                tor_dict[domain] = (3 - eps_k) / 2
        elif self.component == "Electrode":
            tor_dict = {}
            for domain in self.options.whole_cell_domains:
                if domain == "separator":
                    tor_k = pybamm.FullBroadcast(0, "separator", "current collector")
                else:
                    Domain = domain.capitalize()
                    eps_k = variables[f"{Domain} active material volume fraction"]
                    tor_k = (3 - eps_k) / 2
                tor_dict[domain] = tor_k

        variables.update(self._get_standard_transport_efficiency_variables(tor_dict))

        return variables
