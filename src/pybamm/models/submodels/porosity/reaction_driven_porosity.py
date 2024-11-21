#
# Class for reaction driven porosity changes as a multiple of SEI/plating thicknesses
#
import pybamm
from .base_porosity import BaseModel


class ReactionDriven(BaseModel):
    """Reaction-driven porosity changes as a multiple of SEI/plating thicknesses

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
            if domain == "separator":
                delta_eps_k = 0  # separator porosity does not change
            else:
                Domain = domain.split()[0].capitalize()
                L_sei_k = pybamm.CoupledVariable(f"{Domain} total SEI thickness [m]", domain=domain, auxiliary_domains = {"secondary": "current collector"})
                self.coupled_variables.update({L_sei_k.name: L_sei_k})
                if Domain == "Negative":
                    L_sei_0 = self.param.n.prim.L_inner_0 + self.param.n.prim.L_outer_0
                elif Domain == "Positive":
                    L_sei_0 = self.param.p.prim.L_inner_0 + self.param.p.prim.L_outer_0
                L_pl_k = pybamm.CoupledVariable(f"{Domain} lithium plating thickness [m]", domain=domain, auxiliary_domains = {"secondary": "current collector"})
                self.coupled_variables.update({L_pl_k.name: L_pl_k})
                L_dead_k = pybamm.CoupledVariable(f"{Domain} dead lithium thickness [m]", domain=domain, auxiliary_domains = {"secondary": "current collector"})
                self.coupled_variables.update({L_dead_k.name: L_dead_k})
                L_sei_cr_k = pybamm.CoupledVariable(f"{Domain} total SEI on cracks thickness [m]", domain=domain, auxiliary_domains = {"secondary": "current collector"})
                self.coupled_variables.update({L_sei_cr_k.name: L_sei_cr_k})
                roughness_k = pybamm.CoupledVariable(f"{Domain} electrode roughness ratio", domain=domain, auxiliary_domains = {"secondary": "current collector"})
                self.coupled_variables.update({roughness_k.name: roughness_k})

                L_tot = (
                    (L_sei_k - L_sei_0)
                    + L_pl_k
                    + L_dead_k
                    + L_sei_cr_k * (roughness_k - 1)
                )

                a_k = pybamm.CoupledVariable(f"{Domain} electrode surface area to volume ratio [m-1]", domain=domain, auxiliary_domains = {"secondary": "current collector"})
                self.coupled_variables.update({a_k.name: a_k})

                # This assumes a thin film so curvature effects are neglected.
                # They could be included (e.g. for a sphere it is
                # a_n * (L_tot + L_tot ** 2 / R_n + L_tot ** # 3 / (3 * R_n ** 2)))
                # but it is not clear if it is relevant or not.
                delta_eps_k = -a_k * L_tot

            domain_param = self.param.domain_params[domain.split()[0]]
            eps_k = domain_param.epsilon_init + delta_eps_k
            eps_dict[domain] = eps_k

        variables = self._get_standard_porosity_variables(eps_dict)
        self.variables.update(variables)

    def add_events_from(self, variables):
        eps_p = variables["Positive electrode porosity"]
        self.events.append(
            pybamm.Event(
                "Zero positive electrode porosity cut-off",
                pybamm.min(eps_p),
                pybamm.EventType.TERMINATION,
            )
        )
        self.events.append(
            pybamm.Event(
                "Max positive electrode porosity cut-off",
                1 - pybamm.max(eps_p),
                pybamm.EventType.TERMINATION,
            )
        )
        if "negative electrode" in self.options.whole_cell_domains:
            eps_n = variables["Negative electrode porosity"]
            self.events.append(
                pybamm.Event(
                    "Zero negative electrode porosity cut-off",
                    pybamm.min(eps_n),
                    pybamm.EventType.TERMINATION,
                )
            )
            self.events.append(
                pybamm.Event(
                    "Max negative electrode porosity cut-off",
                    1 - pybamm.max(eps_n),
                    pybamm.EventType.TERMINATION,
                )
            )
