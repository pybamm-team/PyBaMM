#
# Class for two-dimensional current collectors - composite models
#
import pybamm
from .potential_pair import (
    BasePotentialPair,
    PotentialPair1plus1D,
    PotentialPair2plus1D,
)


class BaseCompositePotentialPair(BasePotentialPair):
    """A submodel for Ohm's law plus conservation of current in the current collectors,
    which uses the voltage-current relationship from the SPM(e).

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
