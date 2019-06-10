#
# Bulter volmer class
#

import pybamm
import numpy as np


class BaseButlerVolmer(pybamm.BaseInterface):
    """
       Butler-Volmer class

    .. math::
        j = j_0(c) * \\sinh(\\eta_r(c))

    Parameters
    ----------
    j0 : :class:`pybamm.Symbol`
        Exchange-current density
    eta_r : :class:`pybamm.Symbol`
        Reaction overpotential
    domain : iter of str, optional
        The domain(s) in which to compute the interfacial current. Default is None,
        in which case j0.domain is used.

    Returns
    -------
    :class:`pybamm.Symbol`
        Interfacial current density

    """

    def __init__(self, param, domain):
        super().__init__(param)
        self._domain = domain

    def get_derived_variables(self, variables):
        """
        Returns variables which are derived from the fundamental variables in the model.
        """

        eta_r = self._get_overpotential(variables)
        j0 = self._get_exchange_current_density(variables)

        if self._domain == "Negative":
            ne = self.param.ne_n
        elif self._domain == "Positive":
            ne = self.param.ne_p
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(self._domain))

        j = 2 * j0 * pybamm.Function(np.sinh, (ne / 2) * eta_r)

        derived_variables = {
            self._domain + " overpotential": eta_r,
            self._domain + " exchange current density": j0,
            self._domain + " interfacial current density": j,
        }

        return derived_variables

    def _get_overpotential(self, variables):

        phi_s = variables[self._domain + " electrode potential"]
        phi_e = variables[self._domain + " electrolyte potential"]
        ocp = variables[self._domain + " open circuit potential"]

        return phi_s - phi_e - ocp

    def _get_exchange_current_density(self, variables):
        raise NotImplementedError
