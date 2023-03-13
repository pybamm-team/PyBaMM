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

    def get_fundamental_variables(self):
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

        return variables

    def get_coupled_variables(self, variables):
        param = self.param

        depsdt_dict = {}
        for domain in self.options.whole_cell_domains:
            domain_param = self.param.domain_params[domain.split()[0]]
            if domain == "separator":
                depsdt_k = pybamm.FullBroadcast(0, domain, "current collector")
            else:
                if self.x_average is True:
                    a_j_k_av = variables[
                        f"X-averaged {domain} volumetric "
                        "interfacial current density [A.m-3]"
                    ]
                    depsdt_k_av = domain_param.DeltaVsurf * a_j_k_av / param.F
                    depsdt_k = pybamm.PrimaryBroadcast(depsdt_k_av, domain)
                else:
                    Domain = domain.capitalize()
                    a_j_k = variables[
                        f"{Domain} volumetric interfacial current density [A.m-3]"
                    ]
                    depsdt_k = domain_param.DeltaVsurf * a_j_k / param.F

            depsdt_dict[domain] = depsdt_k
        variables.update(self._get_standard_porosity_change_variables(depsdt_dict))

        return variables

    def set_rhs(self, variables):
        if self.x_average is True:
            for domain in self.options.whole_cell_domains:
                eps_av = variables[f"X-averaged {domain} porosity"]
                deps_dt_av = variables[f"X-averaged {domain} porosity change [s-1]"]
                self.rhs.update({eps_av: deps_dt_av})
        else:
            eps = variables["Porosity"]
            deps_dt = variables["Porosity change"]
            self.rhs = {eps: deps_dt}

    def set_initial_conditions(self, variables):
        if self.x_average is True:
            for domain in self.options.whole_cell_domains:
                eps_k_av = variables[f"X-averaged {domain} porosity"]
                domain_param = self.param.domain_params[domain.split()[0]]
                self.initial_conditions[eps_k_av] = domain_param.epsilon_init
        else:
            eps = variables["Porosity"]
            self.initial_conditions = {eps: self.param.epsilon_init}

    def set_events(self, variables):
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
