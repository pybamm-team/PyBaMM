#
# Diffusion-limited kinetics
#

import pybamm
from pybamm.models.submodels.interface.base_interface import BaseInterface


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

    def build(self):
        reaction_name = self.reaction_name

        delta_phi_s = self._coupled_electrode_variable(
            "electrode surface potential difference [V]"
        )
        # Get exchange-current density
        j0 = self._get_exchange_current_density()
        # Get open-circuit potential variables and reaction overpotential
        ocp = self._coupled_electrode_variable(
            f"electrode {reaction_name}open-circuit potential [V]"
        )
        eta_r = delta_phi_s - ocp

        # Get interfacial current densities
        j = self._get_diffusion_limited_current_density()
        j_tot_av, a_j_tot_av = self._get_average_total_interfacial_current_density()

        variables = self._get_standard_interfacial_current_variables(j)
        j = variables[
            f"{self.domain.capitalize()} electrode {reaction_name}"
            "interfacial current density [A.m-2]"
        ]
        variables.update(
            self._get_standard_total_interfacial_current_variables(j_tot_av, a_j_tot_av)
        )
        variables.update(self._get_standard_exchange_current_variables(j0))
        variables.update(self._get_standard_overpotential_variables(eta_r))
        variables.update(self._get_standard_volumetric_current_density_variables(j))

        # No SEI film resistance in this model
        eta_sei = pybamm.Scalar(0)
        variables.update(self._get_standard_sei_film_overpotential_variables(eta_sei))
        self.variables.update(variables)

    def _get_diffusion_limited_current_density(self):
        if self.domain == "negative":
            if self.order == "leading":
                j_p = self.coupled_variable(
                    f"X-averaged positive electrode {self.reaction_name}"
                    "interfacial current density [A.m-2]",
                    domain="current collector",
                )
                j = -self.param.p.L * j_p / self.param.n.L
            elif self.order == "full":
                separator_domains = {
                    "primary": "separator",
                    "secondary": "current collector",
                }
                tor_s = self.coupled_variable(
                    "Separator electrolyte transport efficiency",
                    domains=separator_domains,
                )
                c_ox_s = self.coupled_variable(
                    "Separator oxygen concentration [mol.m-3]",
                    domains=separator_domains,
                )
                N_ox_neg_sep_interface = (
                    -pybamm.boundary_value(tor_s, "left")
                    * self.param.D_ox
                    * pybamm.boundary_gradient(c_ox_s, "left")
                )

                j = -N_ox_neg_sep_interface / -self.param.s_ox_Ox / self.param.n.L

        return j
