#
# Class for constant SEI thickness
#
import pybamm
from .base_sei import BaseModel


class ConstantSEI(BaseModel):
    """Base class for no SEI.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'

    **Extends:** :class:`pybamm.particle.BaseParticle`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):
        param = self.param

        # Constant thicknesses
        L_inner = pybamm.sei_parameters.L_inner_0
        L_outer = pybamm.sei_parameters.L_outer_0
        variables = self._get_standard_thickness_variables(L_inner, L_outer)

        # Reactions
        zero = pybamm.FullBroadcast(
            pybamm.Scalar(0), self.domain.lower(), "current collector"
        )
        variables.update(self._get_standard_reaction_variables(zero, zero))

        return variables
