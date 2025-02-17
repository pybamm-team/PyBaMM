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
                if Domain == "Negative":
                    L_sei_0 = self.param.n.prim.L_sei_0
                elif Domain == "Positive":
                    L_sei_0 = self.param.p.prim.L_sei_0
                SEI_option = getattr(self.options, dom)["SEI"]
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
                    L_sei_k = variables[f"{Domain} total {phase_name}SEI thickness [m]"]
                    if SEI_option == "none":
                        L_sei_0 = pybamm.Scalar(0)
                    else:
                        L_sei_0 = pybamm.Parameter(f"{pref}Initial SEI thickness [m]")
                    L_pl_k = variables[
                        f"{Domain} {phase_name}lithium plating thickness [m]"
                    ]
                    L_dead_k = variables[
                        f"{Domain} {phase_name}dead lithium thickness [m]"
                    ]
                    L_sei_cr_k = variables[
                        f"{Domain} total {phase_name}SEI on cracks thickness [m]"
                    ]
                    roughness_k = variables[
                        f"{Domain} {phase_name}electrode roughness ratio"
                    ]

                    L_tot = (
                        (L_sei_k - L_sei_0)
                        + L_pl_k
                        + L_dead_k
                        + L_sei_cr_k * (roughness_k - 1)
                    )

                    a_k = variables[
                        f"{Domain} electrode {phase_name}"
                        "surface area to volume ratio [m-1]"
                    ]

                    # This assumes a thin film so curvature effects are neglected.
                    # They could be included (e.g. for a sphere it is
                    # a_n * (L_tot + L_tot ** 2 / R_n + L_tot ** # 3 / (3 * R_n ** 2)))
                    # but it is not clear if it is relevant or not.
                    delta_eps_k += -a_k * L_tot

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
