#
# Base kinetics class
#

import pybamm
from ..base_interface import BaseInterface


class BaseModel(BaseInterface):
    """
    Base submodel for kinetics

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
        # Calculate delta_phi from phi_s and phi_e if it isn't already known
        if self.domain + " electrode surface potential difference" not in variables:
            variables = self._get_delta_phi(variables)
        delta_phi = variables[self.domain + " electrode surface potential difference"]
        # If delta_phi was broadcast, take only the orphan
        if isinstance(delta_phi, pybamm.Broadcast):
            delta_phi = delta_phi.orphans[0]

        # Get exchange-current density
        j0 = self._get_exchange_current_density(variables)
        # Get open-circuit potential variables and reaction overpotential
        ocp, dUdT = self._get_open_circuit_potential(variables)
        eta_r = delta_phi - ocp
        # Get number of electrons in reaction
        ne = self._get_number_of_electrons_in_reaction()

        j = self._get_kinetics(j0, ne, eta_r)
        j_tot_av = self._get_average_total_interfacial_current_density(variables)
        # j = j_tot_av + (j - pybamm.average(j))  # enforce true average

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

    def _get_kinetics(self, j0, ne, eta_r):
        raise NotImplementedError

    def _get_open_circuit_potential(self, variables):
        raise NotImplementedError
