#
# Bulter volmer class
#

import pybamm
from ..base_interface import BaseInterface


class NoReaction(BaseInterface):
    """
    Base submodel for when no reaction occurs

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    phase : str
        Phase of the particle

    **Extends:** :class:`pybamm.interface.kinetics.BaseKinetics`
    """

    def __init__(self, param, domain, reaction, phase):
        options = {
            "SEI film resistance": "none",
            "total interfacial current density as a state": "false",
        }
        super().__init__(param, domain, reaction, options, phase)

    def get_fundamental_variables(self):
        zero = pybamm.Scalar(0)
        variables = self._get_standard_interfacial_current_variables(zero)
        variables.update(self._get_standard_exchange_current_variables(zero))
        return variables

    def _get_dj_dc(self, variables):
        return pybamm.Scalar(0)

    def _get_dj_ddeltaphi(self, variables):
        return pybamm.Scalar(0)

    def _get_j_diffusion_limited_first_order(self, variables):
        return pybamm.Scalar(0)
