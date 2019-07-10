#
# First-order Butler-Volmer kinetics
#

import pybamm
from ..base_interface import BaseInterface


class BaseFirstOrderButlerVolmer(BaseInterface):
    """
    First-order Butler-Volmer kinetics

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
        # Update delta_phi with new phi_e and phi_s
        # variables = self._get_delta_phi(variables)
        # Unpack
        # Multiply c_e_0 by 1 to change its id slightly and thus avoid clash with
        # the c_e_0 in delta_phi_0
        c_e_0 = variables["Leading-order average electrolyte concentration"] * 1
        delta_phi_0 = variables[
            "Leading-order average "
            + self.domain.lower()
            + " electrode surface potential difference"
        ]
        c_e = variables[self.domain + " electrolyte concentration"]
        delta_phi = variables[self.domain + " electrode surface potential difference"]
        j_0 = variables[
            "Average " + self.domain.lower() + " electrode interfacial current density"
        ]

        c_e_1 = (c_e - c_e_0) / self.param.C_e
        delta_phi_1 = (delta_phi - delta_phi_0) / self.param.C_e
        j_1 = j_0.diff(c_e_0) * c_e_1 + j_0.diff(delta_phi_0) * delta_phi_1
        j = j_0 + self.param.C_e * j_1

        # Get exchange-current density
        j0 = self._get_exchange_current_density(variables)
        # Get open-circuit potential variables and reaction overpotential
        ocp, dUdT = self._get_open_circuit_potential(variables)
        eta_r = delta_phi - ocp

        variables.update(self._get_standard_interfacial_current_variables(j))
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
