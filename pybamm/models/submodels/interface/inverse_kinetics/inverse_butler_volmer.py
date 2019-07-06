#
# Bulter volmer class
#

import pybamm
import autograd.numpy as np
from ..base_interface import BaseInterface


class BaseInverseButlerVolmer(BaseInterface):
    """
    A base submodel that implements the inverted form of the Butler-Volmer relation to
    solve for the reaction overpotential.

    Parameters
    ----------
    param
        Model parameters
    domain : iter of str, optional
        The domain(s) in which to compute the interfacial current. Default is None,
        in which case j.domain is used.

    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_coupled_variables(self, variables):
        ocp, dUdT = self._get_open_circuit_potential(variables)

        j0 = self._get_exchange_current_density(variables)
        j_av = self._get_average_interfacial_current_density(variables)
        j = pybamm.Broadcast(j_av, [self.domain.lower() + " electrode"])

        if self.domain == "Negative":
            ne = self.param.ne_n
        elif self.domain == "Positive":
            ne = self.param.ne_p

        eta_r = (2 / ne) * pybamm.Function(np.arcsinh, j / (2 * j0))

        delta_phi = eta_r + ocp

        variables.update(self._get_standard_interfacial_current_variables(j, j_av))
        variables.update(self._get_standard_exchange_current_variables(j0))
        variables.update(self._get_standard_overpotential_variables(eta_r))
        variables.update(
            self._get_standard_surface_potential_difference_variables(delta_phi)
        )
        variables.update(self._get_standard_ocp_variables(ocp, dUdT))

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

    def _get_exchange_current_density(self, variables):
        raise NotImplementedError

    def _get_open_circuit_potential(self, variables):
        raise NotImplementedError
