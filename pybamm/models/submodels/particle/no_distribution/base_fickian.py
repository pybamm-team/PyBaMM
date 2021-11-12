#
# Base class for particles with Fickian diffusion
#
import pybamm
from ..base_particle import BaseParticle


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

    def _get_effective_diffusivity(self, c, T):
        param = self.param

        # Get diffusivity
        if self.domain == "Negative":
            D = param.D_n(c, T)
        elif self.domain == "Positive":
            D = param.D_p(c, T)

        # Account for stress-induced diffusion by defining a multiplicative
        # "stress factor"
        # This option can either be a string (both sides the same) or a 2-tuple
        # to indicate different options in negative and positive electrodes
        if isinstance(self.options["stress-induced diffusion"], str):
            stress_left = self.options["stress-induced diffusion"]
            stress_right = self.options["stress-induced diffusion"]
        else:
            stress_left, stress_right = self.options["stress-induced diffusion"]

        if self.domain == "Negative" and stress_left == "true":
            stress_factor = 1 + param.theta_n * (c - param.c_n_0) / (
                1 + param.Theta * T
            )
        elif self.domain == "Positive" and stress_right == "true":
            stress_factor = 1 + param.theta_p * (c - param.c_p_0) / (
                1 + param.Theta * T
            )
        else:
            stress_factor = 1

        return D * stress_factor

    def _get_standard_diffusivity_variables(self, D_eff):
        if self.domain == "Negative":
            D_scale = self.param.D_n_typ_dim
        elif self.domain == "Positive":
            D_scale = self.param.D_p_typ_dim

        variables = {
            self.domain + " effective diffusivity": D_eff,
            self.domain + " effective diffusivity [m2.s-1]": D_eff * D_scale,
            "X-averaged "
            + self.domain.lower()
            + " effective diffusivity": pybamm.x_average(D_eff),
            "X-averaged "
            + self.domain.lower()
            + " effective diffusivity [m2.s-1]": pybamm.x_average(D_eff * D_scale),
        }

        return variables
