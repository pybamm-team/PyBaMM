#
# Base class for particles
#
import pybamm


class BaseParticle(pybamm.BaseSubModel):
    """Base class for molar conservation in particles.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'


    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_standard_concentration_variables(self, c_s, c_s_xav):

        c_s_surf = pybamm.surf(c_s)
        c_s_surf_av = pybamm.x_average(c_s_surf)

        if self.domain == "Negative":
            c_scale = self.param.c_n_max
            eps_s = self.param.epsilon_s_n
            L = self.param.L_n
        elif self.domain == "Positive":
            c_scale = self.param.c_p_max
            eps_s = self.param.epsilon_s_p
            L = self.param.L_p

        c_s_rav = pybamm.r_average(c_s)
        # c_s_vol_av = pybamm.x_average(eps_s * c_s_rav)
        c_s_vol_av = c_s_rav
        variables = {
            self.domain + " particle concentration": c_s,
            self.domain + " particle concentration [mol.m-3]": c_s * c_scale,
            "X-averaged " + self.domain.lower() + " particle concentration": c_s_xav,
            "X-averaged "
            + self.domain.lower()
            + " particle concentration [mol.m-3]": c_s_xav * c_scale,
            self.domain + " particle surface concentration": c_s_surf,
            self.domain
            + " particle surface concentration [mol.m-3]": c_scale * c_s_surf,
            "X-averaged "
            + self.domain.lower()
            + " particle surface concentration": c_s_surf_av,
            "X-averaged "
            + self.domain.lower()
            + " particle surface concentration [mol.m-3]": c_scale * c_s_surf_av,
            self.domain + " electrode active volume fraction": eps_s,
            self.domain + " electrode volume-averaged concentration": c_s_vol_av,
            self.domain
            + " electrode "
            + "volume-averaged concentration [mol.m-3]": c_s_vol_av * c_scale,
            self.domain + " electrode average extent of lithiation": c_s_rav,
            "Total lithium concentration in "
            + self.domain.lower()
            + " electrode [mol.m-2]": c_s_vol_av * c_scale * L,
        }

        return variables

    def _get_standard_flux_variables(self, N_s, N_s_xav):
        variables = {
            self.domain + " particle flux": N_s,
            "X-averaged " + self.domain.lower() + " particle flux": N_s_xav,
        }

        return variables

    def set_events(self, variables):
        c_s_surf = variables[self.domain + " particle surface concentration"]
        tol = 1e-4

        self.events.append(
            pybamm.Event(
                "Minumum " + self.domain.lower() + " particle surface concentration",
                pybamm.min(c_s_surf) - tol,
                pybamm.EventType.TERMINATION,
            )
        )

        self.events.append(
            pybamm.Event(
                "Maximum " + self.domain.lower() + " particle surface concentration",
                (1 - tol) - pybamm.max(c_s_surf),
                pybamm.EventType.TERMINATION,
            )
        )
