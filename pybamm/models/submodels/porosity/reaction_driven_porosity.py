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

    **Extends:** :class:`pybamm.porosity.BaseModel`
    """

    def __init__(self, param, options, x_average):
        super().__init__(param, options)
        self.x_average = x_average

    def get_coupled_variables(self, variables):
        eps_dict = {}
        for domain in self.options.whole_cell_domains:
            if domain == "negative electrode":
                # Only the negative electrode porosity changes
                L_sei_n = variables["Total SEI thickness [m]"]
                L_sei_0 = self.param.n.prim.L_inner_0 + self.param.n.prim.L_outer_0
                L_pl_n = variables["Lithium plating thickness [m]"]
                L_dead_n = variables["Dead lithium thickness [m]"]
                L_sei_cr_n = variables["Total SEI on cracks thickness [m]"]
                roughness_n = variables["Negative electrode roughness ratio"]

                L_tot = (
                    (L_sei_n - L_sei_0)
                    + L_pl_n
                    + L_dead_n
                    + L_sei_cr_n * (roughness_n - 1)
                )

                a_n = variables["Negative electrode surface area to volume ratio [m-1]"]

                # This assumes a thin film so curvature effects are neglected.
                # They could be included (e.g. for a sphere it is
                # a_n * (L_tot + L_tot ** 2 / R_n + L_tot ** # 3 / (3 * R_n ** 2)))
                # but it is not clear if it is relevant or not.
                delta_eps_k = -a_n * L_tot
            else:
                delta_eps_k = 0

            domain_param = self.param.domain_params[domain.split()[0]]
            eps_k = domain_param.epsilon_init + delta_eps_k
            eps_dict[domain] = eps_k

        variables = self._get_standard_porosity_variables(eps_dict)

        return variables

    def set_events(self, variables):
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
