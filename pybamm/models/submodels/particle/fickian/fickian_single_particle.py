#
# Class for a single particle with Fickian diffusion
#
import pybamm

from .base_fickian_particle import BaseModel


class SingleParticle(BaseModel):
    """Base class for molar conservation in a single x-averaged particle which employs
    Fick's law.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'


    **Extends:** :class:`pybamm.particle.fickian.BaseModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):
        if self.domain == "Negative":
            c_s_xav = pybamm.standard_variables.c_s_n_xav
            c_s = pybamm.SecondaryBroadcast(
                pybamm.PrimaryBroadcast(c_s_xav, ["negative electrode"]),
                "current collector",
            )

        elif self.domain == "Positive":
            c_s_xav = pybamm.standard_variables.c_s_p_xav
            c_s = pybamm.SecondaryBroadcast(
                pybamm.PrimaryBroadcast(c_s_xav, ["positive electrode"]),
                "current collector",
            )

        N_s_xav = self._flux_law(c_s_xav)
        N_s = pybamm.SecondaryBroadcast(
            pybamm.PrimaryBroadcast(N_s_xav, [self._domain.lower() + " electrode"]),
            "current collector",
        )

        variables = self._get_standard_concentration_variables(c_s, c_s_xav)
        variables.update(self._get_standard_flux_variables(N_s, N_s_xav))

        return variables

    def _unpack(self, variables):
        c_s_xav = variables[
            "X-average " + self.domain.lower() + " particle concentration"
        ]
        N_s_xav = variables["X-average " + self.domain.lower() + " particle flux"]
        j_av = variables[
            "Average " + self.domain.lower() + " electrode interfacial current density"
        ]

        return c_s_xav, N_s_xav, j_av
