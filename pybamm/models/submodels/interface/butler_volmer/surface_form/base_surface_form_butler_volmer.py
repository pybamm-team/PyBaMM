#
# Bulter volmer class
#

# N.B. this can be a child of the standard butler-volmer class but
# i have left for now because there is a lot to do
# maybe a small function _get_delta_phi_s which is overwritten depending
# if surface form or standard form

import pybamm
import autograd.numpy as np
from ...base_interface import BaseInterface


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

        i_boundary_cc = variables["Current collector current density"]
        delta_phi_s = variables[
            self._domain + " electrode surface potential difference"
        ]
        ocp = variables[self._domain + " electrode open circuit potential"]

        eta_r = delta_phi_s - ocp
        j0 = self._get_exchange_current_density(variables)

        if self._domain == "Negative":
            ne = self.param.ne_n
            j_av = i_boundary_cc / pybamm.geometric_parameters.l_n

        elif self._domain == "Positive":
            ne = self.param.ne_p
            j_av = -i_boundary_cc / pybamm.geometric_parameters.l_p

        j = 2 * j0 * pybamm.Function(np.sinh, (ne / 2) * eta_r)

        j0_av = pybamm.average(j0)

        j = j_av + (j - pybamm.average(j))  # enforce true average

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
