#
# Base class for particles with Fickian diffusion
#
import pybamm
from .base_particle import BaseParticle


class BaseFickian(BaseParticle):
    """
    Base class for molar conservation in particles with Fickian diffusion.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`

    **Extends:** :class:`pybamm.particle.BaseParticle`
    """

    def __init__(self, param, domain, options):
        super().__init__(param, domain, options)

        if self.options["stress induced diffusion"] == "true":
            pybamm.citations.register("Ai2019")
            pybamm.citations.register("Deshpande2012")

    def _get_effective_diffusivity(self, c, T):
        if self.domain == "Negative":
            D = self.param.D_n(c, T)
        elif self.domain == "Positive":
            D = self.param.D_p(c, T)

        # Account for stress induced diffusion
        if self.options["stress induced diffusion"] == "true":
            if self.domain == "Negative":
                theta = self.param.theta_n
                c_0 = self.param.c_0_n
            elif self.domain == "Positive":
                theta = self.param.theta_p
                c_0 = self.param.c_0_p

            D_eff = D * (1 + theta * (c - c_0) / (1 + self.param.Theta * T))
        else:
            D_eff = D

        return D_eff

    def _get_standard_flux_variables(self, N_s, N_s_xav, D_eff):
        variables = {
            self.domain + " particle flux": N_s,
            "X-averaged " + self.domain.lower() + " particle flux": N_s_xav,
            self.domain + " effective diffusivity": D_eff,
            "X-averaged "
            + self.domain.lower()
            + " effective diffusivity": pybamm.x_average(D_eff),
        }

        return variables
