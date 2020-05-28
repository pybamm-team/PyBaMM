#
# Bulter volmer class
#

import pybamm
from .base_kinetics import BaseKinetics


class ButlerVolmer(BaseKinetics):
    """
    Base submodel which implements the forward Butler-Volmer equation:

    .. math::
        j = 2 * j_0(c) * \\sinh( (ne / (2 * (1 + \\Theta T)) * \\eta_r(c))

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model. In this case "sei film
        resistance" is the important option. See :class:`pybamm.BaseBatteryModel`

    **Extends:** :class:`pybamm.interface.kinetics.BaseKinetics`
    """

    def __init__(self, param, domain, reaction, options=None):
        super().__init__(param, domain, reaction, options)

    def _get_kinetics(self, j0, ne, eta_r, T):
        prefactor = ne / (2 * (1 + self.param.Theta * T))
        return 2 * j0 * pybamm.sinh(prefactor * eta_r)

    def _get_dj_dc(self, variables):
        "See :meth:`pybamm.interface.kinetics.BaseKinetics._get_dj_dc`"
        c_e, delta_phi, j0, ne, ocp, T = self._get_interface_variables_for_first_order(
            variables
        )
        eta_r = delta_phi - ocp
        prefactor = ne / (2 * (1 + self.param.Theta * T))
        return (2 * j0.diff(c_e) * pybamm.sinh(prefactor * eta_r)) - (
            2 * j0 * prefactor * ocp.diff(c_e) * pybamm.cosh(prefactor * eta_r)
        )

    def _get_dj_ddeltaphi(self, variables):
        "See :meth:`pybamm.interface.kinetics.BaseKinetics._get_dj_ddeltaphi`"
        _, delta_phi, j0, ne, ocp, T = self._get_interface_variables_for_first_order(
            variables
        )
        eta_r = delta_phi - ocp
        prefactor = ne / (2 * (1 + self.param.Theta * T))
        return 2 * j0 * prefactor * pybamm.cosh(prefactor * eta_r)
