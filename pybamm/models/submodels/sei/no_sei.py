#
# Class for no SEI
#
import pybamm
from .base_sei import BaseModel


class NoSEI(BaseModel):
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
        zero = pybamm.FullBroadcast(
            pybamm.Scalar(0), self.domain.lower(), "current collector"
        )
        variables = self._get_standard_thickness_variables(zero, zero)
        variables.update(self._get_standard_reaction_variables(zero, zero))
        return variables
