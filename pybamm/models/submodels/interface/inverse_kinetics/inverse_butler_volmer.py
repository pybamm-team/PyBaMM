#
# Bulter volmer class
#

import pybamm
import autograd.numpy as np
from .base_inverse_kinetics import BaseInverseKinetics
from ..kinetics.butler_volmer import ButlerVolmer


class InverseButlerVolmer(BaseInverseKinetics, ButlerVolmer):
    """
    A base submodel that implements the inverted form of the Butler-Volmer relation to
    solve for the reaction overpotential.

    Parameters
    ----------
    param
        Model parameters
    domain : iter of str, optional
        The domain(s) in which to compute the interfacial current. Default is None,
        in which case j.domain is used.

    **Extends:** :class:`pybamm.interface.kinetics.ButlerVolmer`

    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_inverse_kinetics(self, j, j0, ne):
        return (2 / ne) * pybamm.Function(np.arcsinh, j / (2 * j0))
