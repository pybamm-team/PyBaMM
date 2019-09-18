#
# Leading-order diffusion limited kinetics
#

import pybamm
from ..base_interface import BaseInterface


class BaseModel(BaseInterface):
    """
    Leading-order submodel for diffusion-limited kinetics

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.


    **Extends:** :class:`pybamm.interface.BaseInterface`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_coupled_variables(self, variables):
        # Calculate delta_phi_s from phi_s and phi_e if it isn't already known
        if self.domain + " electrode surface potential difference" not in variables:
            variables = self._get_delta_phi(variables)
        delta_phi_s = variables[self.domain + " electrode surface potential difference"]
        # If delta_phi_s was broadcast, take only the orphan
        if isinstance(delta_phi_s, pybamm.Broadcast):
            delta_phi_s = delta_phi_s.orphans[0]

        # Get exchange-current density
        j0 = self._get_exchange_current_density(variables)
        # Get open-circuit potential variables and reaction overpotential
        ocp, dUdT = self._get_open_circuit_potential(variables)
        eta_r = delta_phi_s - ocp

        # Get interfacial current densities
        j = self._get_diffusion_limited_current_density(variables)
        j_tot_av = self._get_average_total_interfacial_current_density(variables)

        variables.update(self._get_standard_interfacial_current_variables(j))
        variables.update(
            self._get_standard_total_interfacial_current_variables(j_tot_av)
        )
        variables.update(self._get_standard_exchange_current_variables(j0))
        variables.update(self._get_standard_overpotential_variables(eta_r))
        variables.update(self._get_standard_ocp_variables(ocp, dUdT))

        if (
            "Negative electrode" + self.reaction_name + " interfacial current density"
            in variables
            and "Positive electrode"
            + self.reaction_name
            + " interfacial current density"
            in variables
        ):
            variables.update(
                self._get_standard_whole_cell_interfacial_current_variables(variables)
            )
            variables.update(
                self._get_standard_whole_cell_exchange_current_variables(variables)
            )

        return variables

    def _get_exchange_current_density(self, variables):
        raise NotImplementedError

    def _get_open_circuit_potential(self, variables):
        raise NotImplementedError

    def _get_diffusion_limited_current_density(self, variables):
        raise NotImplementedError

    def _get_dj_dc(self, variables):
        return pybamm.Scalar(0)

    def _get_dj_ddeltaphi(self, variables):
        return pybamm.Scalar(0)
