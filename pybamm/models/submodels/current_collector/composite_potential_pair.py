#
# Class for one- and two-dimensional composite potential pair current collector models
#
import pybamm
from .potential_pair import (
    BasePotentialPair,
    PotentialPair1plus1D,
    PotentialPair2plus1D,
)


class BaseCompositePotentialPair(BasePotentialPair):
    """
    Composite potential pair model for the current collectors.
    This is identical to the BasePotentialPair model, except the name of the fundamental
    variables are changed to avoid clashes with leading order.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.current_collector.BasePotentialPair`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        phi_s_cn = pybamm.standard_variables.phi_s_cn_composite

        variables = self._get_standard_negative_potential_variables(phi_s_cn)

        # TODO: grad not implemented for 2D yet
        i_cc = pybamm.Scalar(0)
        i_boundary_cc = pybamm.standard_variables.i_boundary_cc_composite

        variables.update(self._get_standard_current_variables(i_cc, i_boundary_cc))

        return variables


class CompositePotentialPair1plus1D(BaseCompositePotentialPair, PotentialPair1plus1D):
    def __init__(self, param):
        super().__init__(param)


class CompositePotentialPair2plus1D(BaseCompositePotentialPair, PotentialPair2plus1D):
    def __init__(self, param):
        super().__init__(param)
