#
# Base class for particles with Fickian diffusion
#
import pybamm


class FickianSingleParticle(pybamm.BaseSubModel):
    """Base class for molar conservation in a single x-averaged particle which employs
    Fick's law.

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

    def get_fundamental_variables(self):
        """
        Returns the variables in the submodel for which a PDE must be solved to obtains
        """

        if self._domain == "Negative":
            c_s_xav = pybamm.standard_variables.c_s_n_xav

        elif self._domain == "Positive":
            c_s_xav = pybamm.standard_variables.c_s_p_xav

        else:
            pybamm.DomainError("Domain must be either: 'Negative' or 'Positive'")

        fundamental_variables = {
            "X-average " + self._domain.lower() + " particle concentration": c_s_xav
        }

        return fundamental_variables

    def get_derived_variables(self, variables):

        c_s_xav = variables[
            "X-average " + self._domain.lower() + "particle concentration"
        ]

        c_s = pybamm.Broadcast(c_s_xav, [self._domain.lower() + " electrode"])

        variables.update({self._domain + " particle concentration": c_s})

        variables.update(self.get_standard_derived_variables(variables))

        return variables

