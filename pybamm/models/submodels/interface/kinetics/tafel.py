#
# Tafel classes
#
import pybamm
from .base_kinetics import BaseKinetics


class ForwardTafel(BaseKinetics):
    """
    Base submodel which implements the forward Tafel equation:

    .. math::
        j = u * j_0(c) * \\exp((ne * alpha * F * \\eta_r(c) / RT)

    Parameters
    ----------
    param :
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
    """

    def __init__(self, param, domain, reaction, options, phase="primary"):
        super().__init__(param, domain, reaction, options, phase)

    def _get_kinetics(self, j0, ne, eta_r, T, u):
        alpha = self.phase_param.alpha_bv
        Feta_RT = self.param.F * eta_r / (self.param.R * T)
        return u * j0 * pybamm.exp(ne * alpha * Feta_RT)


# backwardtafel not used by any of the models
# class BackwardTafel(BaseKinetics):
#     """
#     Base submodel which implements the backward Tafel equation:

#     .. math::
#         j = -j_0(c) * \\exp(-\\eta_r(c))

#     Parameters
#     ----------
#     param :
#         model parameters
#     domain : str
#         The domain to implement the model, either: 'Negative' or 'Positive'.


#     **Extends:** :class:`pybamm.interface.kinetics.BaseKinetics`
#     """

#     def __init__(self, param, domain, reaction, options):
#         super().__init__(param, domain, reaction, options)

#     def _get_kinetics(self, j0, ne, eta_r, T):
#         return -j0 * pybamm.exp(-(ne / (2 * (1 + self.param.Theta * T))) * eta_r)
