#
# Marcus and Asymptotic Marcus-Hush-Chidsey classes
#

import pybamm
import numpy as np
from .base_kinetics import BaseKinetics


class Marcus(BaseKinetics):
    """
    Submodel which implements Marcus kinetics.

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
        pybamm.citations.register("Sripad2020")

    def _get_kinetics(self, j0, ne, eta_r, T, u):
        RT = self.param.R * T
        Feta_RT = self.param.F * eta_r / RT
        mhc_lambda = self.phase_param.mhc_lambda

        exp_arg_ox = -((mhc_lambda + Feta_RT) ** 2) / (4 * mhc_lambda * RT)
        exp_arg_red = -((mhc_lambda - Feta_RT) ** 2) / (4 * mhc_lambda * RT)
        return u * j0 * (pybamm.exp(exp_arg_ox) - pybamm.exp(exp_arg_red))


class MarcusHushChidsey(BaseKinetics):
    """
    Submodel which implements asymptotic Marcus-Hush-Chidsey kinetics, as derived in
    [1]_.

    References
    ----------
    .. [1] Sripad, S., Korff, D., DeCaluwe, S. C., & Viswanathan, V. (2020). "Kinetics
           of lithium electrodeposition and stripping".
           [The Journal of Chemical Physics](https://doi.org/10.1063/5.0023771),
           153(19), 194701.

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
        pybamm.citations.register("Sripad2020")

    def _get_kinetics(self, j0, ne, eta_r, T, u):
        mhc_lambda = self.phase_param.mhc_lambda

        F_RT = self.param.F / (self.param.R * T)
        Feta_RT = F_RT * eta_r
        lambda_T = F_RT * mhc_lambda
        a = 1 + pybamm.sqrt(lambda_T)

        arg = (lambda_T - pybamm.sqrt(a + Feta_RT**2)) / (2 * pybamm.sqrt(lambda_T))
        pref = pybamm.sqrt(np.pi * lambda_T) * pybamm.tanh(Feta_RT / 2)
        return j0 * u * pref * pybamm.erfc(arg)
