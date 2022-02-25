#
# Bulter volmer class
#

import pybamm
from .base_kinetics import BaseKinetics


class SymmetricButlerVolmer(BaseKinetics):
    """
    Submodel which implements the symmetric forward Butler-Volmer equation:

    .. math::
        j = 2 * j_0(c) * \\sinh( (ne / (2 * (1 + \\Theta T)) * \\eta_r(c))

    Parameters
    ----------
    param : parameter class
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`

    **Extends:** :class:`pybamm.interface.kinetics.BaseKinetics`
    """

    def __init__(self, param, domain, reaction, options):
        super().__init__(param, domain, reaction, options)

    def _get_kinetics(self, j0, ne, eta_r, T, u):
        prefactor = ne / (2 * (1 + self.param.Theta * T))
        return 2 * u * j0 * pybamm.sinh(prefactor * eta_r)

    def _get_dj_dc(self, variables):
        """See :meth:`pybamm.interface.kinetics.BaseKinetics._get_dj_dc`"""
        (
            c_e,
            delta_phi,
            j0,
            ne,
            ocp,
            T,
            u,
        ) = self._get_interface_variables_for_first_order(variables)
        eta_r = delta_phi - ocp
        prefactor = ne / (2 * (1 + self.param.Theta * T))
        return (2 * u * j0.diff(c_e) * pybamm.sinh(prefactor * eta_r)) - (
            2 * u * j0 * prefactor * ocp.diff(c_e) * pybamm.cosh(prefactor * eta_r)
        )

    def _get_dj_ddeltaphi(self, variables):
        """See :meth:`pybamm.interface.kinetics.BaseKinetics._get_dj_ddeltaphi`"""
        _, delta_phi, j0, ne, ocp, T, u = self._get_interface_variables_for_first_order(
            variables
        )
        eta_r = delta_phi - ocp
        prefactor = ne / (2 * (1 + self.param.Theta * T))
        return 2 * u * j0 * prefactor * pybamm.cosh(prefactor * eta_r)


class AsymmetricButlerVolmer(BaseKinetics):
    """
    Submodel which implements the asymmetric forward Butler-Volmer equation

    Parameters
    ----------
    param : parameter class
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`

    **Extends:** :class:`pybamm.interface.kinetics.BaseKinetics`
    """

    def __init__(self, param, domain, reaction, options):
        super().__init__(param, domain, reaction, options)

    def _get_kinetics(self, j0, ne, eta_r, T, u):
        if self.domain == "Negative":
            alpha = self.param.alpha_bv_n
        elif self.domain == "Positive":
            alpha = self.param.alpha_bv_p
        arg_ox = ne * alpha * eta_r / (1 + self.param.Theta * T)
        arg_red = -ne * (1 - alpha) * eta_r / (1 + self.param.Theta * T)
        return u * j0 * (pybamm.exp(arg_ox) - pybamm.exp(arg_red))
