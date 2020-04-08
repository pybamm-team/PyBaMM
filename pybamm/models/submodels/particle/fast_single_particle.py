#
# Class for single particle with uniform concentration (i.e. infinitely fast
# diffusion in r)
#
import pybamm

from .base_particle import BaseParticle


class FastSingleParticle(BaseParticle):
    """Base class for molar conservation in a single x-averaged particle with
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
        # can just use the surface value. This avoids dealing with both
        # x *and* r averaged quantities, which may be confusing.

        if self.domain == "Negative":
            c_s_surf_xav = pybamm.standard_variables.c_s_n_surf_xav
            c_s_xav = pybamm.PrimaryBroadcast(c_s_surf_xav, ["negative particle"])
            c_s = pybamm.SecondaryBroadcast(c_s_xav, ["negative electrode"])

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
            c_s_surf_xav = pybamm.standard_variables.c_s_p_surf_xav
            c_s_xav = pybamm.PrimaryBroadcast(c_s_surf_xav, ["positive particle"])
            c_s = pybamm.SecondaryBroadcast(c_s_xav, ["positive electrode"])

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

        c_s_surf_xav = variables[
            "X-averaged " + self.domain.lower() + " particle surface concentration"
        ]
        j_xav = variables[
            "X-averaged "
            + self.domain.lower()
            + " electrode interfacial current density"
        ]

        if self.domain == "Negative":
            self.rhs = {c_s_surf_xav: -3 * j_xav / self.param.a_n}

        elif self.domain == "Positive":
            self.rhs = {c_s_surf_xav: -3 * j_xav / self.param.a_p / self.param.gamma_p}

    def set_initial_conditions(self, variables):
        """
        For single particle models, initial conditions can't depend on x so we
        arbitrarily evaluate them at x=0 in the negative electrode and x=1 in the
        positive electrode (they will usually be constant)
        """
        c_s_surf_xav = variables[
            "X-averaged " + self.domain.lower() + " particle surface concentration"
        ]

        if self.domain == "Negative":
            c_init = self.param.c_n_init(0)

        elif self.domain == "Positive":
            c_init = self.param.c_p_init(1)

        self.initial_conditions = {c_s_surf_xav: c_init}
