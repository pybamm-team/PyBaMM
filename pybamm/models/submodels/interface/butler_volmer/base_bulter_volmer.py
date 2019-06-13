#
# Bulter volmer class
#

import pybamm
import numpy as np
from ..base_interface import BaseInterface


class BaseModel(BaseInterface):
    """
       Butler-Volmer class

    .. math::
        j = j_0(c) * \\sinh(\\eta_r(c))

    Parameters
    ----------
    param : 
        model parameters
    domain : iter of str, optional

    Returns
    -------
    :class:`pybamm.Symbol`
        Interfacial current density

    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_coupled_variables(self, variables):
        """
        Returns variables which are derived from the fundamental variables in the model.
        """

        phi_s = variables[self._domain + " electrode potential"]
        phi_e = variables[self._domain + " electrolyte potential"]
        ocp = variables[self._domain + " open circuit potential"]

        delta_phi_s = phi_s - phi_e
        eta_r = phi_s - phi_e - ocp
        j0 = self._get_exchange_current_density(variables)

        if self._domain == "Negative":
            ne = self.param.ne_n
        elif self._domain == "Positive":
            ne = self.param.ne_p
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(self._domain))

        j = 2 * j0 * pybamm.Function(np.sinh, (ne / 2) * eta_r)

        j0_av = pybamm.average(j0)
        j_av = pybamm.average(j)
        delta_phi_s_av = pybamm.average(delta_phi_s)
        eta_r_av = pybamm.average(eta_r)

        variables.update(self._get_standard_interfacial_current_variables(j, j_av))
        variables.update(self._get_standard_exchange_current_variables(j0, j0_av))
        variables.update(
            self._get_standard_surface_potential_difference_variables(
                delta_phi_s, delta_phi_s_av
            )
        )
        variables.update(self._get_standard_overpotential_variables(eta_r, eta_r_av))

        return variables

    def _get_exchange_current_density(self, variables):
        raise NotImplementedError
