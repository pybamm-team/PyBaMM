#
# Class for many particles, each with uniform concentration (i.e. infinitely
# fast diffusion in r)
#
import pybamm

from .base_particle import BaseParticle


class FastManyParticles(BaseParticle):
    """Base class for molar conservation in many particles with
    uniform concentration in r (i.e. infinitely fast diffusion within particles).

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

    def get_fundamental_variables(self):
        # The particle concentration is uniform throughout the particle, so we
        # can just use the surface value.
        if self.domain == "Negative":
            c_s_surf = pybamm.standard_variables.c_s_n_surf
            x = pybamm.standard_spatial_vars.x_n
            R = self.param.R_n_of_x(x)
            variables = {"Negative particle distribution in x": R}

        elif self.domain == "Positive":
            c_s_surf = pybamm.standard_variables.c_s_p_surf
            x = pybamm.standard_spatial_vars.x_p
            R = self.param.R_p_of_x(x)
            variables = {"Positive particle distribution in x": R}

        c_s = pybamm.PrimaryBroadcast(c_s_surf, [self.domain.lower() + " particle"])

        N_s = pybamm.FullBroadcastToEdges(
            0,
            [self.domain.lower() + " particle"],
            auxiliary_domains={
                "secondary": self.domain.lower() + " electrode",
                "tertiary": "current collector",
            },
        )
        N_s_xav = pybamm.FullBroadcastToEdges(
            0, self.domain.lower() + " particle", "current collector"
        )

        c_s_xav = pybamm.x_average(c_s)

        variables.update(
            self._get_standard_concentration_variables(c_s, c_s_xav=c_s_xav)
        )
        variables.update(self._get_standard_flux_variables(N_s, N_s_xav))

        return variables

    def set_rhs(self, variables):
        c_s_surf = variables[self.domain + " particle surface concentration"]
        j = variables[self.domain + " electrode interfacial current density"]
        R = variables[self.domain + " particle distribution in x"]

        if self.domain == "Negative":
            self.rhs = {c_s_surf: -3 * j / self.param.a_n / R}

        elif self.domain == "Positive":
            self.rhs = {c_s_surf: -3 * j / self.param.a_p / self.param.gamma_p / R}

    def set_initial_conditions(self, variables):
        c_s_surf = variables[self.domain + " particle surface concentration"]
        if self.domain == "Negative":
            x_n = pybamm.standard_spatial_vars.x_n
            c_init = self.param.c_n_init(x_n)

        elif self.domain == "Positive":
            x_p = pybamm.standard_spatial_vars.x_p
            c_init = self.param.c_p_init(x_p)

        self.initial_conditions = {c_s_surf: c_init}
