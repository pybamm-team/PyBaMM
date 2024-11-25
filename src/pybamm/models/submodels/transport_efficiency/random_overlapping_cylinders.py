#
# Class for random overlapping cylinders transport_efficiency
#
import pybamm
from .base_transport_efficiency import BaseModel


class RandomOverlappingCylinders(BaseModel):
    """Submodel for Random Overlapping Cylinders transport_efficiency,
    :footcite:t:`Tomadakis1993`, :footcite:t:`Shen2007`

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

    def build(self):
        pybamm.citations.register("Shen2007")
        pybamm.citations.register("Tomadakis1993")
        if self.component == "Electrolyte":
            tor_dict = {}
            for domain in self.options.whole_cell_domains:
                Domain = domain.capitalize()
                eps_k = pybamm.CoupledVariable(
                    f"{Domain} porosity",
                    domain=domain,
                    auxiliary_domains={"secondary": "current collector"},
                )
                self.coupled_variables.update({eps_k.name: eps_k})
                tor_k = 1 - pybamm.Log(eps_k)
                tor_dict[domain] = tor_k
        elif self.component == "Electrode":
            tor_dict = {}
            for domain in self.options.whole_cell_domains:
                if domain == "separator":
                    tor_k = pybamm.FullBroadcast(0, "separator", "current collector")
                else:
                    Domain = domain.capitalize()
                    eps_k = pybamm.CoupledVariable(
                        f"{Domain} porosity",
                        domain=domain,
                        auxiliary_domains={"secondary": "current collector"},
                    )
                    self.coupled_variables.update({eps_k.name: eps_k})
                    phi_k = 1 - eps_k
                    tor_k = 1 - pybamm.Log(phi_k)
                tor_dict[domain] = tor_k
        variables = self._get_standard_transport_efficiency_variables(tor_dict)
        self.variables.update(variables)
