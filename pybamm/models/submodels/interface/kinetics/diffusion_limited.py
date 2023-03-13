#
# Diffusion-limited kinetics
#

import pybamm
from ..base_interface import BaseInterface


class DiffusionLimited(BaseInterface):
    """
    Submodel for diffusion-limited kinetics

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model. See
        :class:`pybamm.BaseBatteryModel`
    order : str
        The order of the model ("leading" or "full")
    """

    def __init__(self, param, domain, reaction, options, order):
        super().__init__(param, domain, reaction, options)
        self.order = order

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        reaction_name = self.reaction_name

        delta_phi_s = variables[f"{Domain} electrode surface potential difference [V]"]
        # If delta_phi_s was broadcast, take only the orphan
        if isinstance(delta_phi_s, pybamm.Broadcast):
            delta_phi_s = delta_phi_s.orphans[0]

        # Get exchange-current density
        j0 = self._get_exchange_current_density(variables)
        # Get open-circuit potential variables and reaction overpotential
        ocp = variables[f"{Domain} electrode {reaction_name}open-circuit potential [V]"]
        eta_r = delta_phi_s - ocp

        # Get interfacial current densities
        j = self._get_diffusion_limited_current_density(variables)
        j_tot_av, a_j_tot_av = self._get_average_total_interfacial_current_density(
            variables
        )

        variables.update(self._get_standard_interfacial_current_variables(j))
        variables.update(
            self._get_standard_total_interfacial_current_variables(j_tot_av, a_j_tot_av)
        )
        variables.update(self._get_standard_exchange_current_variables(j0))
        variables.update(self._get_standard_overpotential_variables(eta_r))

        variables.update(
            self._get_standard_volumetric_current_density_variables(variables)
        )

        # No SEI film resistance in this model
        eta_sei = pybamm.Scalar(0)
        variables.update(self._get_standard_sei_film_overpotential_variables(eta_sei))

        return variables

    def _get_diffusion_limited_current_density(self, variables):
        param = self.param
        if self.domain == "negative":
            if self.order == "leading":
                j_p = variables[
                    f"X-averaged positive electrode {self.reaction_name}"
                    "interfacial current density [A.m-2]"
                ]
                j = -self.param.p.L * j_p / self.param.n.L
            elif self.order == "full":
                tor_s = variables["Separator electrolyte transport efficiency"]
                c_ox_s = variables["Separator oxygen concentration [mol.m-3]"]
                N_ox_neg_sep_interface = (
                    -pybamm.boundary_value(tor_s, "left")
                    * param.D_ox
                    * pybamm.boundary_gradient(c_ox_s, "left")
                )

                j = -N_ox_neg_sep_interface / -param.s_ox_Ox / param.n.L

        return j
