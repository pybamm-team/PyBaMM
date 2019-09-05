#
# Base class for particles each with uniform concentration (i.e. infinitely fast
# diffusion in r)
#
import pybamm

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

    def _flux_law(self, c, T):
        # Infinitely fast diffusion
        if self.domain == "Negative":
            N = pybamm.PrimaryBroadcast(
                pybamm.PrimaryBroadcast(0, "current collector"), "negative particle"
            )
        elif self.domain == "Positive":
            N = pybamm.PrimaryBroadcast(
                pybamm.PrimaryBroadcast(0, "current collector"), "positive particle"
            )
        return N

    def _unpack(self, variables):
        raise NotImplementedError

    def get_coupled_variables(self, variables):
        # Flux always zero, so can pass None
        N_s_xav = self._flux_law(None, None)
        N_s = pybamm.PrimaryBroadcast(N_s_xav, [self._domain.lower() + " electrode"])

        variables.update(self._get_standard_flux_variables(N_s, N_s_xav))

        return variables

    def set_rhs(self, variables):

        c, _, j = self._unpack(variables)

        if self.domain == "Negative":
            self.rhs = {c: -3 * j / self.param.a_n}

        elif self.domain == "Positive":
            self.rhs = {c: -3 * j / self.param.a_p / self.param.gamma_p}
