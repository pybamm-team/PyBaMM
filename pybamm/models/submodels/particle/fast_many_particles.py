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
            c_s = pybamm.PrimaryBroadcast(c_s_surf, ["negative particle"])
            c_s_xav = pybamm.x_average(c_s)

            N_s = pybamm.FullBroadcastToEdges(
                0,
                ["negative particle"],
                auxiliary_domains={
                    "secondary": "negative electrode",
                    "tertiary": "current collector",
                },
            )
            N_s_xav = pybamm.FullBroadcast(0, "negative electrode", "current collector")

        elif self.domain == "Positive":
            c_s_surf = pybamm.standard_variables.c_s_p_surf
            c_s = pybamm.PrimaryBroadcast(c_s_surf, ["positive particle"])
            c_s_xav = pybamm.x_average(c_s)

            N_s = pybamm.FullBroadcastToEdges(
                0,
                ["positive particle"],
                auxiliary_domains={
                    "secondary": "positive electrode",
                    "tertiary": "current collector",
                },
            )
            N_s_xav = pybamm.FullBroadcast(0, "positive electrode", "current collector")

        variables = self._get_standard_concentration_variables(c_s, c_s_xav)
        variables.update(self._get_standard_flux_variables(N_s, N_s_xav))

        return variables

    def set_rhs(self, variables):
        c_s_surf = variables[self.domain + " particle surface concentration"]
        j = variables[self.domain + " electrode interfacial current density"]
        if self.domain == "Negative":
            self.rhs = {c_s_surf: -3 * j / self.param.a_n}

        elif self.domain == "Positive":
            self.rhs = {c_s_surf: -3 * j / self.param.a_p / self.param.gamma_p}

    def set_initial_conditions(self, variables):
        c_s_surf = variables[self.domain + " particle surface concentration"]
        if self.domain == "Negative":
            x_n = pybamm.standard_spatial_vars.x_n
            c_init = self.param.c_n_init(x_n)

        elif self.domain == "Positive":
            x_p = pybamm.standard_spatial_vars.x_p
            c_init = self.param.c_p_init(x_p)

        self.initial_conditions = {c_s_surf: c_init}
