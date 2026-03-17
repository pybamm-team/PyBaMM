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

    def get_coupled_variables(self, variables):
        eps_dict = {}
        for domain in self.options.whole_cell_domains:
            delta_eps_k = 0
            if domain != "separator":  # separator porosity does not change
                dom = domain.split()[0]
                Domain = dom.capitalize()
                SEI_option = getattr(self.options, dom)["SEI"]
                plating_option = getattr(self.options, dom)["lithium plating"]
                phases_option = getattr(self.options, dom)["particle phases"]
                phases = self.options.phases[dom]
                for phase in phases:
                    if phases_option == "1" and phase == "primary":
                        # `domain` has one phase
                        phase_name = ""
                        pref = ""
                    else:
                        # `domain` has more than one phase
                        phase_name = phase + " "
                        pref = phase.capitalize() + ": "
                    if SEI_option == "none":
                        V_bar_sei = pybamm.Scalar(0)
                    else:
                        V_bar_sei = pybamm.Parameter(
                            f"{pref}SEI partial molar volume [m3.mol-1]"
                        )
                    if plating_option == "none":
                        V_bar_Li = pybamm.Scalar(0)
                    else:
                        V_bar_Li = pybamm.Parameter(
                            "Lithium metal partial molar volume [m3.mol-1]"
                        )
                    c_sei_k = variables[
                        f"{Domain} {phase_name}SEI concentration [mol.m-3]"
                    ]
                    c_sei_cr_k = variables[
                        f"{Domain} {phase_name}SEI on cracks concentration [mol.m-3]"
                    ]
                    c_sei_tot_k = c_sei_k + c_sei_cr_k
                    c_pl_k = variables[
                        f"{Domain} {phase_name}lithium plating concentration [mol.m-3]"
                    ]
                    c_dead_k = variables[
                        f"{Domain} {phase_name}dead lithium concentration [mol.m-3]"
                    ]
                    c_pl_tot_k = c_pl_k + c_dead_k

                    delta_eps_k += (V_bar_sei * c_sei_tot_k + V_bar_Li * c_pl_tot_k)

            domain_param = self.param.domain_params[domain.split()[0]]
            eps_k = domain_param.epsilon_init + delta_eps_k
            eps_dict[domain] = eps_k

        variables = self._get_standard_porosity_variables(eps_dict)

        return variables

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
