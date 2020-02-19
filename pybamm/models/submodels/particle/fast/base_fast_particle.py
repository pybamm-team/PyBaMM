#
# Base class for particles each with uniform concentration (i.e. infinitely fast
# diffusion in r)
#
from ..base_particle import BaseParticle


class BaseModel(BaseParticle):
    """Base class for molar conservation in particles with uniform concentration
    in r (i.e. infinitely fast diffusion within particles).

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

    def _unpack(self, variables):
        raise NotImplementedError

    def set_rhs(self, variables):

        c, _, j = self._unpack(variables)

        if self.domain == "Negative":
            self.rhs = {c: -3 * j / self.param.a_n}

        elif self.domain == "Positive":
            self.rhs = {c: -3 * j / self.param.a_p / self.param.gamma_p}
