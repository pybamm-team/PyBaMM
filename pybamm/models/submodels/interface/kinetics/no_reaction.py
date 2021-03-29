#
# Bulter volmer class
#

import pybamm
from .base_kinetics import BaseKinetics


class NoReaction(BaseKinetics):
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

    **Extends:** :class:`pybamm.interface.kinetics.BaseKinetics`
    """

    def __init__(self, param, domain, reaction):
        options = {
            "SEI film resistance": "none",
            "total interfacial current density as a state": "false",
        }
        super().__init__(param, domain, reaction, options)

    def _get_kinetics(self, j0, ne, eta_r, T):
        return pybamm.Scalar(0)
