#
# Base class for particles with Fickian diffusion
#
import pybamm

from ..base_particle import BaseParticle


class BaseModel(BaseParticle):
    """Base class for molar conservation in particles which employ Fick's law.

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

    def _flux_law(self, c, T):

        if self.domain == "Negative":
            D = self.param.D_n(c, T)
        elif self.domain == "Positive":
            D = self.param.D_p(c, T)

        return -D * pybamm.grad(c)

    def _unpack(self, variables):
        raise NotImplementedError

    def set_rhs(self, variables):

        c, N, _ = self._unpack(variables)

        if self.domain == "Negative":
            self.rhs = {c: -(1 / self.param.C_n) * pybamm.div(N)}

        elif self.domain == "Positive":
            self.rhs = {c: -(1 / self.param.C_p) * pybamm.div(N)}

    def set_boundary_conditions(self, variables):
        raise NotImplementedError
