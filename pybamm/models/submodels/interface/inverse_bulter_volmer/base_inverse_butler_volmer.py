#
# Bulter volmer class
#

import pybamm
import numpy as np


class BaseInverseButlerVolmer(pybamm.BaseInterface):
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
        super().__init__(param)
        self._domain = domain

    def get_derived_variables(self, variables):
        """
        Returns variables which are derived from the fundamental variables in the model.
        """

        j0 = self._get_exchange_current_density(variables)
        j = self._get_interfacial_current_density(variables)

        if self._domain == "Negative":
            ne = self.param.ne_n
        elif self._domain == "Positive":
            ne = self.param.ne_p
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(self._domain))

        eta_r = (2 / ne) * pybamm.Function(np.arcsinh, j / (2 * j0))

        derived_variables = {
            self._domain + " exchange current density": j0,
            self._domain + " interfacial current density": j,
            self._domain + " reaction overpotential": eta_r,
        }

        return derived_variables

    def _get_exchange_current_density(self, variables):
        raise NotImplementedError

    def _get_interfacial_current_density(self, variables):

        i_boundary_cc = variables["Current collector current density"]

        if self._domain == "Negative":
            j = i_boundary_cc / pybamm.geometric_parameters.l_n
        elif self._domain == "Positive":
            j - i_boundary_cc / pybamm.geometric_parameters.l_p
