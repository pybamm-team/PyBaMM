#
# Class for Ordered packing transport_efficiency
#
import pybamm
from .base_transport_efficiency import BaseModel


class OrderedPacking(BaseModel):
    """Submodel for Ordered Packing transport_efficiency
    :footcite:t:`akanni1987effective`, :footcite:t:`shen2007critical`

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
        pybamm.citations.register("shen2007critical")
        pybamm.citations.register("akanni1987effective")
        if self.component == "Electrolyte":
            tor_dict = {}
            for domain in self.options.whole_cell_domains:
                Domain = domain.capitalize()
                eps_k = variables[f"{Domain} porosity"]
                tor_k = (3 - eps_k) * 0.5
                tor_dict[domain] = tor_k
        elif self.component == "Electrode":
            tor_dict = {}
            for domain in self.options.whole_cell_domains:
                if domain == "separator":
                    tor_k = pybamm.FullBroadcast(0, "separator", "current collector")
                else:
                    Domain = domain.capitalize()
                    phi_k = 1 - variables[f"{Domain} porosity"]
                    tor_k = (3 - phi_k) * 0.5
                tor_dict[domain] = tor_k
        variables.update(self._get_standard_transport_efficiency_variables(tor_dict))

        return variables
