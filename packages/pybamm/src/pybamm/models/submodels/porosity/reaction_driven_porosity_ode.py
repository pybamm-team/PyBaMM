#
# Class for reaction driven porosity changes as an ODE
#
import pybamm

from .base_porosity import BaseModel


class ReactionDrivenODE(BaseModel):
    """Reaction-driven porosity changes as an ODE

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict
        Options dictionary passed from the full model
    x_average : bool
        Whether to use x-averaged variables (SPM, SPMe, etc) or full variables (DFN)
    """

    def __init__(self, param, options, x_average):
        super().__init__(param, options)
        self.x_average = x_average

    def build(self):
        eps_dict = {}
        depsdt_dict = {}
        for domain in self.options.whole_cell_domains:
            Domain = domain.capitalize()
            domain_param = self.param.domain_params[domain.split()[0]]
            if self.x_average is True:
                eps_k_av = pybamm.Variable(
                    f"X-averaged {domain} porosity",
                    domain="current collector",
                    bounds=(0, 1),
                )
                eps_k = pybamm.PrimaryBroadcast(eps_k_av, domain)
            else:
                eps_k = pybamm.Variable(
                    f"{Domain} porosity",
                    domain=domain,
                    auxiliary_domains={"secondary": "current collector"},
                    bounds=(0, 1),
                )
            eps_dict[domain] = eps_k

            if domain == "separator":
                depsdt_k = pybamm.FullBroadcast(0, domain, "current collector")
            elif self.x_average is True:
                a_j_k_av = self.coupled_variable(
                    f"X-averaged {domain} volumetric interfacial current density "
                    "[A.m-3]",
                    domain="current collector",
                )
                depsdt_k_av = domain_param.DeltaVsurf * a_j_k_av / self.param.F
                depsdt_k = pybamm.PrimaryBroadcast(depsdt_k_av, domain)
            else:
                a_j_k = self.coupled_variable(
                    f"{Domain} volumetric interfacial current density [A.m-3]",
                    domain=domain,
                    auxiliary_domains={"secondary": "current collector"},
                )
                depsdt_k = domain_param.DeltaVsurf * a_j_k / self.param.F
            depsdt_dict[domain] = depsdt_k

            if self.x_average is True:
                self.rhs[eps_k_av] = pybamm.x_average(depsdt_k)
                self.initial_conditions[eps_k_av] = domain_param.epsilon_init

            if domain != "separator":
                self.events.extend(self._get_porosity_cut_off_events(domain, eps_k))

        variables = self._get_standard_porosity_variables(eps_dict)
        variables.update(self._get_standard_porosity_change_variables(depsdt_dict))
        self.variables.update(variables)

        if self.x_average is False:
            eps = variables["Porosity"]
            self.rhs[eps] = variables["Porosity change"]
            self.initial_conditions[eps] = self.param.epsilon_init
