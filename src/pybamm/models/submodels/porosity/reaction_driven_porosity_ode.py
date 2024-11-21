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
        for domain in self.options.whole_cell_domains:
            Domain = domain.capitalize()
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
        variables = self._get_standard_porosity_variables(eps_dict)
        self.variables.update(variables)
    
        depsdt_dict = {}
        for domain in self.options.whole_cell_domains:
            domain_param = self.param.domain_params[domain.split()[0]]
            if domain == "separator":
                depsdt_k = pybamm.FullBroadcast(0, domain, "current collector")
            else:
                if self.x_average is True:
                    a_j_k_av = pybamm.CoupledVariable(
                        f"X-averaged {domain} volumetric "
                        "interfacial current density [A.m-3]"
                    )
                    self.coupled_variables.update({f"X-averaged {domain} volumetric interfacial current density [A.m-3]": a_j_k_av})
                    depsdt_k_av = domain_param.DeltaVsurf * a_j_k_av / self.param.F
                    depsdt_k = pybamm.PrimaryBroadcast(depsdt_k_av, domain)
                else:
                    Domain = domain.capitalize()
                    a_j_k = pybamm.CoupledVariable(
                        f"{Domain} volumetric interfacial current density [A.m-3]",
                        domain=domain,
                        auxiliary_domains={"secondary": "current collector"},
                    )
                    self.coupled_variables.update({f"{Domain} volumetric interfacial current density [A.m-3]": a_j_k})
                    depsdt_k = domain_param.DeltaVsurf * a_j_k / self.param.F

            depsdt_dict[domain] = depsdt_k
        self.variables.update(self._get_standard_porosity_change_variables(depsdt_dict))
    
        if self.x_average is True:
            for domain in self.options.whole_cell_domains:
                eps_av = self.variables[f"X-averaged {domain} porosity"]
                deps_dt_av = self.variables[f"X-averaged {domain} porosity change [s-1]"]
                self.rhs.update({eps_av: deps_dt_av})
        else:
            eps = self.variables["Porosity"]
            #self.coupled_variables.update({f"Porosity": eps})
            deps_dt = self.variables["Porosity change"]
            #self.coupled_variables.update({f"Porosity change": deps_dt})
            self.rhs = {eps: deps_dt}

        if self.x_average is True:
            for domain in self.options.whole_cell_domains:
                eps_k_av = variables[f"X-averaged {domain} porosity"]
                domain_param = self.param.domain_params[domain.split()[0]]
                self.initial_conditions[eps_k_av] = domain_param.epsilon_init
        else:
            eps = self.variables["Porosity"]
            self.initial_conditions = {eps: self.param.epsilon_init}

    def add_events_from(self, variables):
        for domain in self.options.whole_cell_domains:
            if domain == "separator":
                continue
            Domain = domain.capitalize()
            eps_k = variables[f"{Domain} porosity"]
            self.events.append(
                pybamm.Event(
                    f"Zero {domain} porosity cut-off",
                    pybamm.min(eps_k),
                    pybamm.EventType.TERMINATION,
                )
            )
            self.events.append(
                pybamm.Event(
                    f"Max {domain} porosity cut-off",
                    1 - pybamm.max(eps_k),
                    pybamm.EventType.TERMINATION,
                )
            )
