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
        domain_param = self.domain_param

        # Get diffusivity
        D = domain_param.D(c, T)

        # Account for stress-induced diffusion by defining a multiplicative
        # "stress factor"
        stress_option = getattr(self.options, self.domain.lower())[
            "stress-induced diffusion"
        ]

        if stress_option == "true":
            stress_factor = 1 + domain_param.theta * (c - domain_param.c_0) / (
                1 + param.Theta * T
            )
        else:
            stress_factor = 1

        return D * stress_factor

    def _get_standard_diffusivity_variables(self, D_eff):
        D_scale = self.domain_param.D_typ_dim

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
