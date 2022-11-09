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
    phase : str, optional
        Phase of the particle (default is "primary")

    **Extends:** :class:`pybamm.interface.kinetics.BaseKinetics`
    """

    def __init__(self, param, domain, reaction, options, phase="primary"):
        super().__init__(param, domain, reaction, options, phase)

    def _get_kinetics(self, j0, ne, eta_r, T, u):
        prefactor = ne / (2 * (1 + self.param.Theta * T))
        return 2 * u * j0 * pybamm.sinh(prefactor * eta_r)


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
    phase : str, optional
        Phase of the particle (default is "primary")

    **Extends:** :class:`pybamm.interface.kinetics.BaseKinetics`
    """

    def __init__(self, param, domain, reaction, options, phase="primary"):
        super().__init__(param, domain, reaction, options, phase)

    def _get_kinetics(self, j0, ne, eta_r, T, u):
        alpha = self.phase_param.alpha_bv
        arg_ox = ne * alpha * eta_r / (1 + self.param.Theta * T)
        arg_red = -ne * (1 - alpha) * eta_r / (1 + self.param.Theta * T)
        return u * j0 * (pybamm.exp(arg_ox) - pybamm.exp(arg_red))
