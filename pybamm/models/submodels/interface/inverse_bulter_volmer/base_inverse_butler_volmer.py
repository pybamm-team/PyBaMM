#
# Bulter volmer class
#

import pybamm
import autograd.numpy as np
from ..base_interface import BaseInterface


class BaseModel(BaseInterface):
    """
    Inverts the Butler-Volmer relation to solve for the reaction overpotential.

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
        """
        Returns variables which are derived from the fundamental variables in the model.
        """

        j0 = self._get_exchange_current_density(variables)
        j0_av = pybamm.average(j0)
        j_av = self._get_average_interfacial_current_density(variables)
        j = pybamm.Broadcast(j_av, [self._domain.lower() + " electrode"])

        if self._domain == "Negative":
            ne = self.param.ne_n
        elif self._domain == "Positive":
            ne = self.param.ne_p
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(self._domain))

        eta_r = (2 / ne) * pybamm.Function(np.arcsinh, j / (2 * j0))
        eta_r_av = pybamm.average(eta_r)

        variables.update(self._get_standard_interfacial_current_variables(j, j_av))
        variables.update(self._get_standard_exchange_current_variables(j0, j0_av))
        variables.update(self._get_standard_overpotential_variables(eta_r, eta_r_av))

        if self._domain == "Positive":
            variables.update(
                self._get_standard_whole_cell_interfacial_current_variables(variables)
            )
            variables.update(
                self._get_standard_whole_cell_exchange_current_variables(variables)
            )

        return variables

    def _get_exchange_current_density(self, variables):
        raise NotImplementedError

    def _get_average_interfacial_current_density(self, variables):

        i_boundary_cc = variables["Current collector current density"]

        if self._domain == "Negative":
            j_av = i_boundary_cc / pybamm.geometric_parameters.l_n
        elif self._domain == "Positive":
            j_av = -i_boundary_cc / pybamm.geometric_parameters.l_p

        return j_av
