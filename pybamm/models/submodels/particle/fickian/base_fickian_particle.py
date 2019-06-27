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

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _flux_law(self, c):
        return -pybamm.grad(c)

    def _unpack():
        raise NotImplementedError

    def set_rhs(self, variables):

        c, N, _ = self._unpack(variables)

        if self._domain == "Negative":
            self.rhs = {c: -(1 / self.param.C_n) * pybamm.div(N)}

        elif self._domain == "Positive":
            self.rhs = {c: -(1 / self.param.C_p) * pybamm.div(N)}

    def set_boundary_conditions(self, variables):

        c, _, j = self._unpack(variables)

        if self._domain == "Negative":
            rbc = -self.param.C_n * j / self.param.a_n

        elif self._domain == "Positive":
            rbc = -self.param.C_p * j / self.param.a_p / self.param.gamma_p

        self.boundary_conditions = {
            c: {"left": (pybamm.Scalar(0), "Neumann"), "right": (rbc, "Neumann")}
        }

