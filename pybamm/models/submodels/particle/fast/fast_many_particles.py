#
# Class for many particles, each with uniform concentration (i.e. infinitely
# fast diffusion in r)
#
import pybamm

from .base_fast_particle import BaseModel


class ManyParticles(BaseModel):
    """Base class for molar conservation in many particles with
    uniform concentration in r (i.e. infinitely fast diffusion within particles).

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'


    **Extends:** :class:`pybamm.particle.fast.BaseModel`
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

            N_s = pybamm.FullBroadcast(
                0,
                ["negative particle"],
                auxiliary_domains={
                    "secondary": "negative electrode",
                    "tertiary": "current collector",
                },
            )
            N_s_xav = pybamm.x_average(N_s)

        elif self.domain == "Positive":
            c_s_surf = pybamm.standard_variables.c_s_p_surf
            c_s = pybamm.PrimaryBroadcast(c_s_surf, ["positive particle"])
            c_s_xav = pybamm.x_average(c_s)

            N_s = pybamm.FullBroadcast(
                0,
                ["positive particle"],
                auxiliary_domains={
                    "secondary": "positive electrode",
                    "tertiary": "current collector",
                },
            )
            N_s_xav = pybamm.x_average(N_s)

        variables = self._get_standard_concentration_variables(c_s, c_s_xav)
        variables.update(self._get_standard_flux_variables(N_s, N_s_xav))

        return variables

    def _unpack(self, variables):
        c_s_surf = variables[self.domain + " particle surface concentration"]
        N_s = variables[self.domain + " particle flux"]
        j = variables[self.domain + " electrode interfacial current density"]

        return c_s_surf, N_s, j

    def set_initial_conditions(self, variables):
        c, _, _ = self._unpack(variables)

        if self.domain == "Negative":
            x_n = pybamm.standard_spatial_vars.x_n
            c_init = self.param.c_n_init(x_n)

        elif self.domain == "Positive":
            x_p = pybamm.standard_spatial_vars.x_p
            c_init = self.param.c_p_init(x_p)

        self.initial_conditions = {c: c_init}
