#
# Base class for particles with Fickian diffusion
#
import pybamm


class BaseFickianParticle(pybamm.BaseSubModel):
    """Base class for molar conservation in particles which employ Fick's law.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'

    *Extends:* :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param)
        self._domain = domain

    def _ficks_law(self, c):
        return pybamm.grad(c)

    def _conservation_of_mass(self, c, N):

        if self._domain == "Negative":
            rhs = {c: -(1 / self.param.C_n) * pybamm.div(N)}

        elif self._domain == "Positive":
            rhs = {c: -(1 / self.param.C_p) * pybamm.div(N)}

        else:
            raise pybamm.DomainError("Invalid particle domain")

        return rhs

    def _boundary_conditions(self, c, j):

        if self._domain == "Negative":
            rbc = -self.param.C_n * j / self.param.a_n

        elif self._domain == "Positive":
            rbc = -self.param.C_p * j / self.param.a_p / self.param.gamma_p

        else:
            raise pybamm.DomainError("Invalid particle domain")

        boundary_conditions = {c: {"left": (0, "Neumann"), "right": (rbc, "Neumann")}}

        return boundary_conditions

