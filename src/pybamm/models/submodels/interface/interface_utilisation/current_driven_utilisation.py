#
# Class for current-driven ODE for interface utilisation
#
import pybamm
from .base_utilisation import BaseModel


class CurrentDriven(BaseModel):
    """Current-driven ODE for interface utilisation

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'negative' or 'positive'
    options : dict, optional
        A dictionary of options to be passed to the model.
    reaction_loc : str
        Where the reaction happens: "x-average" (SPM, SPMe, etc),
        "full electrode" (full DFN), or "interface" (half-cell model)
    """

    def __init__(self, param, domain, options, reaction_loc):
        super().__init__(param, domain, options)
        self.reaction_loc = reaction_loc

    def build(self):
        domain, Domain = self.domain_Domain

        if self.reaction_loc == "full electrode":
            u = pybamm.Variable(
                f"{Domain} electrode interface utilisation variable",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
        elif self.reaction_loc == "x-average":
            u_xav = pybamm.Variable(
                f"X-averaged {domain} electrode interface utilisation variable",
                domain="current collector",
            )
            u = pybamm.PrimaryBroadcast(u_xav, f"{domain} electrode")
        else:
            u = pybamm.Variable(
                "Lithium metal interface utilisation variable",
                domain="current collector",
            )
        variables = self._get_standard_interface_utilisation_variables(u)

        if self.reaction_loc == "full electrode":
            a_j = pybamm.CoupledVariable(
                f"{Domain} electrode volumetric interfacial current density [A.m-3]",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
            self.coupled_variables.update({a_j.name: a_j})
        elif self.reaction_loc == "x-average":
            a_j = pybamm.CoupledVariable(
                f"X-averaged {domain} electrode volumetric "
                "interfacial current density [A.m-3]",
                domain="current collector",
            )
            self.coupled_variables.update({a_j.name: a_j})
        else:
            a_j = pybamm.CoupledVariable(
                "Lithium metal total interfacial current density [A.m-2]",
                domain="current collector",
            )
            self.coupled_variables.update({a_j.name: a_j})
        beta = self.domain_param.beta_utilisation

        self.rhs = {u: beta * u * a_j / self.param.F}

        u_init = self.domain_param.u_init

        self.initial_conditions = {u: u_init}
        self.variables.update(variables)

    def add_events_from(self, variables):
        domain, Domain = self.domain_Domain

        if self.reaction_loc == "full electrode":
            u = variables[f"{Domain} electrode interface utilisation"]
        elif self.reaction_loc == "x-average":
            u = variables[f"X-averaged {domain} electrode interface utilisation"]
        else:
            u = variables["Lithium metal interface utilisation"]
        self.events.append(
            pybamm.Event(
                f"Zero {domain} electrode utilisation cut-off",
                pybamm.min(u),
                pybamm.EventType.TERMINATION,
            )
        )
