#
# Bulter volmer class
#

import pybamm
from .base_kinetics import BaseModel
from .base_first_order_kinetics import BaseFirstOrderKinetics


class ButlerVolmer(BaseModel):
    """
    Base submodel which implements the forward Butler-Volmer equation:

    .. math::
        j = j_0(c) * \\sinh(\\eta_r(c))

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.


    **Extends:** :class:`pybamm.interface.kinetics.BaseModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_kinetics(self, j0, ne, eta_r):
        return 2 * j0 * pybamm.sinh((ne / 2) * eta_r)

    def _get_dj_dc(self, variables):
        "See :meth:`pybamm.interface.kinetics.BaseModel._get_dj_dc`"
        c_e, delta_phi, j0, ne, ocp = self._get_interface_variables_for_first_order(
            variables
        )
        eta_r = delta_phi - ocp
        return (2 * j0.diff(c_e) * pybamm.sinh((ne / 2) * eta_r)) - (
            2 * j0 * (ne / 2) * ocp.diff(c_e) * pybamm.cosh((ne / 2) * eta_r)
        )

    def _get_dj_ddeltaphi(self, variables):
        "See :meth:`pybamm.interface.kinetics.BaseModel._get_dj_ddeltaphi`"
        _, delta_phi, j0, ne, ocp = self._get_interface_variables_for_first_order(
            variables
        )
        eta_r = delta_phi - ocp
        return 2 * j0 * (ne / 2) * pybamm.cosh((ne / 2) * eta_r)


class FirstOrderButlerVolmer(ButlerVolmer, BaseFirstOrderKinetics):
    def __init__(self, param, domain):
        super().__init__(param, domain)
