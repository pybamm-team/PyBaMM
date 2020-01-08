#
# Tafel classes
#
import pybamm
from .base_kinetics import BaseModel
from .base_first_order_kinetics import BaseFirstOrderKinetics


class ForwardTafel(BaseModel):
    """
    Base submodel which implements the forward Tafel equation:

    .. math::
        j = j_0(c) * \\exp((ne / (2 * (1 + \\Theta T)) * \\eta_r(c))

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

    def _get_kinetics(self, j0, ne, eta_r, T):
        return j0 * pybamm.exp((ne / (2 * (1 + self.param.Theta * T))) * eta_r)

    def _get_dj_dc(self, variables):
        "See :meth:`pybamm.interface.kinetics.BaseKinetics._get_dj_dc`"
        c_e, delta_phi, j0, ne, ocp, T = self._get_interface_variables_for_first_order(
            variables
        )
        eta_r = delta_phi - ocp
        return (
            2
            * j0.diff(c_e)
            * pybamm.exp((ne / (2 * (1 + self.param.Theta * T))) * eta_r)
        )

    def _get_dj_ddeltaphi(self, variables):
        "See :meth:`pybamm.interface.kinetics.BaseKinetics._get_dj_ddeltaphi`"
        _, delta_phi, j0, ne, ocp, T = self._get_interface_variables_for_first_order(
            variables
        )
        eta_r = delta_phi - ocp
        return (
            2
            * j0
            * (ne / (2 * (1 + self.param.Theta * T)))
            * pybamm.exp((ne / 2) * eta_r)
        )


class FirstOrderForwardTafel(ForwardTafel, BaseFirstOrderKinetics):
    def __init__(self, param, domain):
        super().__init__(param, domain)


class BackwardTafel(BaseModel):
    """
    Base submodel which implements the backward Tafel equation:

    .. math::
        j = -j_0(c) * \\exp(-\\eta_r(c))

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

    def _get_kinetics(self, j0, ne, eta_r, T):
        return -j0 * pybamm.exp(-(ne / (2 * (1 + self.param.Theta * T))) * eta_r)
