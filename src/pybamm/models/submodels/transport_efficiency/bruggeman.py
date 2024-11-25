#
# Class for Bruggeman transport_efficiency
#
import pybamm
from .base_transport_efficiency import BaseModel


class Bruggeman(BaseModel):
    """Submodel for Bruggeman transport_efficiency,
    :footcite:t:`Bruggeman1935`

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
                pybamm.citations.register("Bruggeman1935")
                b_k = self.param.domain_params[domain.split()[0]].b_e
                tor_k = eps_k**b_k
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
                    pybamm.citations.register("Bruggeman1935")
                    b_k = self.param.domain_params[domain.split()[0]].b_s
                    tor_k = phi_k**b_k
                tor_dict[domain] = tor_k
        variables = self._get_standard_transport_efficiency_variables(tor_dict)
        self.variables.update(variables)
