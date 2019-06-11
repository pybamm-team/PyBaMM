#
# Class for many particles with Fickian diffusion
#
import pybamm


class FickianManyParticle(pybamm.BaseFickianParticle):
    """Base class for molar conservation in many particles which employs
    Fick's law.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'

    *Extends:* :class:`pybamm.BaseFickianParticle`
    """

    def __init__(self, param, domain):
        super().__init__(param)
        self._domain = domain

    def get_fundamental_variables(self):
        """
        Returns the variables in the submodel for which a PDE must be solved to obtains
        """

        if self._domain == "Negative":
            c_s = pybamm.standard_variables.c_s_n

        elif self._domain == "Positive":
            c_s = pybamm.standard_variables.c_s_p

        else:
            pybamm.DomainError("Domain must be either: 'Negative' or 'Positive'")

        fundamental_variables = {self._domain + " particle concentration": c_s}

        return fundamental_variables

    def get_derived_variables(self, variables):

        c_s = variables[self._domain + " particle concentration"]
        c_s_xav = pybamm.average(c_s)

        variables.update(
            {"X-average " + self._domain + " particle concentration": c_s_xav}
        )

        variables.update(self.get_standard_derived_variables(variables))

        return variables

    def _unpack(self, variables):
        c_s = variables[self._domain + " particle concentration"]
        N_s = variables[self._domain + " particle flux"]
        j = variables[self._domain + " electrode interfacial current density"]

        return c_s, N_s, j

