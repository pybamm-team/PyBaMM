#
# Bulter volmer class
#

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

    def _get_delta_phi_s(self, variables):
        "Calculate delta_phi_s, and derived variables, using phi_s and phi_e"
        phi_s = variables[self.domain + " electrode potential"]
        phi_e = variables[self.domain + " electrolyte potential"]
        delta_phi_s = phi_s - phi_e
        variables.update(
            self._get_standard_surface_potential_difference_variables(delta_phi_s)
        )
        return variables

    def get_coupled_variables(self, variables):
        # Calculate delta_phi_s from phi_s and phi_e if it isn't already known
        if self.domain + " electrode surface potential difference" not in variables:
            variables = self._get_delta_phi_s(variables)
        delta_phi_s = variables[self.domain + " electrode surface potential difference"]

        # Get exchange-current density
        j0 = self._get_exchange_current_density(variables)
        # Get open-circuit potential variables and reaction overpotential
        variables.update(self._get_standard_ocp_variables(variables))
        ocp = variables[self.domain + " electrode open circuit potential"]
        eta_r = delta_phi_s - ocp

        if self.domain == "Negative":
            ne = self.param.ne_n
        elif self.domain == "Positive":
            ne = self.param.ne_p

        j = self._get_kinetics(j0, ne, eta_r)
        j_av = self._get_average_interfacial_current_density(variables)
        # j = j_av + (j - pybamm.average(j))  # enforce true average

        variables.update(self._get_standard_interfacial_current_variables(j, j_av))
        variables.update(self._get_standard_exchange_current_variables(j0))
        variables.update(self._get_standard_overpotential_variables(eta_r))

        if (
            "Negative electrode interfacial current density" in variables
            and "Positive electrode interfacial current density" in variables
        ):
            variables.update(
                self._get_standard_whole_cell_interfacial_current_variables(variables)
            )
            variables.update(
                self._get_standard_whole_cell_exchange_current_variables(variables)
            )

        return variables

    def _get_open_circuit_potential(self, variables):
        raise NotImplementedError

    def _get_exchange_current_density(self, variables):
        raise NotImplementedError
