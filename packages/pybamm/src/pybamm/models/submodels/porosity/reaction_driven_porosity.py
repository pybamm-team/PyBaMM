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
            delta_eps_k = 0
            if domain != "separator":  # separator porosity does not change
                delta_eps_k = self._get_porosity_loss(domain)
            domain_param = self.param.domain_params[domain.split()[0]]
            eps_dict[domain] = domain_param.epsilon_init + delta_eps_k

        self.variables.update(self._get_standard_porosity_variables(eps_dict))

        for domain, eps_k in eps_dict.items():
            if domain != "separator":
                self.events.extend(self._get_porosity_cut_off_events(domain, eps_k))

    def _get_porosity_loss(self, domain):
        """Porosity lost to films (SEI, plating, dead lithium) on each phase."""
        dom = domain.split()[0]
        Domain = dom.capitalize()
        electrode_domains = {"primary": domain, "secondary": "current collector"}
        phases_option = getattr(self.options, dom)["particle phases"]
        delta_eps_k = 0
        for phase in self.options.phases[dom]:
            SEI_option = getattr(getattr(self.options, dom), phase)["SEI"]
            if phases_option == "1" and phase == "primary":
                phase_name = ""
                pref = ""
            else:
                phase_name = phase + " "
                pref = phase.capitalize() + ": "

            def coupled(name):
                return self.coupled_variable(name, domains=electrode_domains)

            a_k = coupled(
                f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]"
            )
            L_sei_k = coupled(f"{Domain} {phase_name}SEI thickness [m]")
            L_pl_k = coupled(f"{Domain} {phase_name}lithium plating thickness [m]")
            L_dead_k = coupled(f"{Domain} {phase_name}dead lithium thickness [m]")
            L_sei_cr_k = coupled(f"{Domain} {phase_name}SEI on cracks thickness [m]")
            roughness_k = coupled(f"{Domain} {phase_name}electrode roughness ratio")
            if SEI_option == "none":
                L_sei_0 = pybamm.Scalar(0)
            else:
                L_sei_0 = pybamm.Parameter(f"{pref}Initial SEI thickness [m]")

            L_tot = (
                (L_sei_k - L_sei_0) + L_pl_k + L_dead_k + L_sei_cr_k * (roughness_k - 1)
            )
            # Thin-film approximation: curvature effects are neglected (for a sphere
            # they add a_n * (L_tot ** 2 / R_n + L_tot ** 3 / (3 * R_n ** 2))).
            delta_eps_k += -a_k * L_tot
        return delta_eps_k
