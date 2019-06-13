#
# Class for a single particle with Fickian diffusion
#
import pybamm

from .base_fickian_particle import BaseFickianParticle


class FickianSingleParticle(BaseFickianParticle):
    """Base class for molar conservation in a single x-averaged particle which employs
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
        Returns the variables in the submodel which can be created indpendent of
        other submodels
        """

        if self._domain == "Negative":
            c_s_xav = pybamm.standard_variables.c_s_n_xav
            c_s = pybamm.Broadcast(c_s_xav, ["negative electrode"])

        elif self._domain == "Positive":
            c_s_xav = pybamm.standard_variables.c_s_p_xav
            c_s = pybamm.Broadcast(c_s_xav, ["positive electrode"])

        else:
            pybamm.DomainError("Domain must be either: 'Negative' or 'Positive'")

        variables = self._get_standard_fundamental_variables(c_s, c_s_xav)

        return variables

    def _unpack(self, variables):
        c_s_xav = variables[
            "X-average " + self._domain.lower() + " particle concentration"
        ]
        N_s_xav = variables["X-average " + self._domain.lower() + " particle flux"]
        j_av = variables[
            "Average " + self._domain.lower() + " electrode interfacial current density"
        ]

        return c_s_xav, N_s_xav, j_av
