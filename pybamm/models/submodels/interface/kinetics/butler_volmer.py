#
# Bulter volmer class
#

import pybamm
from .base_kinetics import BaseModel


class BaseButlerVolmer(BaseModel):
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


    **Extends:** :class:`pybamm.interface.kinetics.BaseKinetics`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_kinetics(self, j0, ne, eta_r):
        return 2 * j0 * pybamm.sinh((ne / 2) * eta_r)
