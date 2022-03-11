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
        Either 'Negative' or 'Positive'
    options : dict, optional
        A dictionary of options to be passed to the model.
    reaction_loc : str
        Where the reaction happens: "x-average" (SPM, SPMe, etc),
        "full electrode" (full DFN), or "interface" (half-cell model)

    **Extends:** :class:`pybamm.interface utilisation.BaseModel`
    """

    def __init__(self, param, domain, options, reaction_loc):
        super().__init__(param, domain, options)
        self.reaction_loc = reaction_loc

    def get_fundamental_variables(self):
        if self.reaction_loc == "full electrode":
            if self.domain == "Negative":
                u = pybamm.Variable(
                    "Negative electrode interface utilisation variable",
                    domain="negative electrode",
                    auxiliary_domains={"secondary": "current collector"},
                )
            else:
                u = pybamm.Variable(
                    "Positive electrode interface utilisation variable",
                    domain="positive electrode",
                    auxiliary_domains={"secondary": "current collector"},
                )
        elif self.reaction_loc == "x-average":
            if self.domain == "Negative":
                u_xav = pybamm.Variable(
                    "X-averaged negative electrode interface utilisation variable",
                    domain="current collector",
                )
            else:
                u_xav = pybamm.Variable(
                    "X-averaged positive electrode interface utilisation variable",
                    domain="current collector",
                )
            u = pybamm.PrimaryBroadcast(u_xav, self.domain.lower() + " electrode")
        else:
            u = pybamm.Variable(
                "Lithium metal interface utilisation variable",
                domain="current collector",
            )
        variables = self._get_standard_interface_utilisation_variables(u)

        return variables

    def set_rhs(self, variables):
        if self.reaction_loc == "full electrode":
            u = variables[self.domain + " electrode interface utilisation variable"]
            a = variables[self.domain + " electrode surface area to volume ratio"]
            j = variables[self.domain + " electrode interfacial current density"]
        elif self.reaction_loc == "x-average":
            u = variables[
                "X-averaged "
                + self.domain.lower()
                + " electrode interface utilisation variable"
            ]
            a = variables[
                "X-averaged "
                + self.domain.lower()
                + " electrode surface area to volume ratio"
            ]
            j = variables[
                "X-averaged "
                + self.domain.lower()
                + " electrode interfacial current density"
            ]
        else:
            u = variables["Lithium metal interface utilisation variable"]
            a = 1
            j = variables["Lithium metal total interfacial current density"]

        if self.domain == "Negative":
            beta = self.param.beta_utilisation_n
        else:
            beta = self.param.beta_utilisation_p

        self.rhs = {u: beta * a * u * j}

    def set_initial_conditions(self, variables):
        if self.reaction_loc == "full electrode":
            u = variables[self.domain + " electrode interface utilisation variable"]
        elif self.reaction_loc == "x-average":
            u = variables[
                "X-averaged "
                + self.domain.lower()
                + " electrode interface utilisation variable"
            ]
        else:
            u = variables["Lithium metal interface utilisation variable"]

        if self.domain == "Negative":
            u_init = self.param.u_n_init
        else:
            u_init = self.param.u_p_init

        self.initial_conditions = {u: u_init}

    def set_events(self, variables):
        if self.reaction_loc == "full electrode":
            u = variables[self.domain + " electrode interface utilisation"]
        elif self.reaction_loc == "x-average":
            u = variables[
                "X-averaged " + self.domain.lower() + " electrode interface utilisation"
            ]
        else:
            u = variables["Lithium metal interface utilisation"]
        self.events.append(
            pybamm.Event(
                "Zero " + self.domain.lower() + " electrode utilisation cut-off",
                pybamm.min(u),
                pybamm.EventType.TERMINATION,
            )
        )
