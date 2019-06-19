#
# Base class for particles
#
import pybamm
import autograd.numpy as np


class BaseParticle(pybamm.BaseSubModel):
    """Base class for molar conservation in particles.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'

    *Extends:* :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param)
        self._domain = domain

    def _get_standard_concentration_variables(self, c_s, c_s_xav):

        c_s_surf = pybamm.surf(c_s, set_domain=True)

        c_s_surf_av = pybamm.average(c_s_surf)

        if self._domain == "Negative":
            c_scale = self.param.c_n_max
        elif self._domain == "Positive":
            c_scale = self.param.c_p_max

        variables = {
            self._domain + " particle concentration": c_s,
            self._domain + " particle concentration [mol.m-3]": c_s * c_scale,
            "X-average " + self._domain.lower() + " particle concentration": c_s_xav,
            "X-average "
            + self._domain.lower()
            + " particle concentration [mol.m-3]": c_s_xav * c_scale,
            self._domain + " particle surface concentration": c_s_surf,
            self._domain
            + " particle surface concentration [mol.m-3]": c_scale * c_s_surf,
            "Average "
            + self._domain.lower()
            + " particle surface concentration": c_s_surf_av,
            "Average "
            + self._domain.lower()
            + " particle surface concentration [mol.m-3]": c_scale * c_s_surf_av,
        }

        return variables

    def _get_standard_flux_variables(self, N_s, N_s_xav):

        variables = {
            self._domain + " particle flux": N_s,
            "X-average " + self._domain.lower() + " particle flux": N_s_xav,
        }

        return variables

    def _get_standard_ocp_variables(self, c_s):

        c_s_surf = pybamm.surf(c_s, set_domain=True)

        if self._domain == "Negative":
            ocp = self.param.U_n(c_s_surf)
            ocp_dim = self.param.U_n_ref + self.param.potential_scale * ocp
            dudT = self.param.dUdT_n(c_s_surf)

        elif self._domain == "Positive":
            ocp = self.param.U_p(c_s_surf)
            ocp_dim = self.param.U_p_ref + self.param.potential_scale * ocp
            dudT = self.param.dUdT_p(c_s_surf)

        else:
            pybamm.DomainError

        ocp_av = pybamm.average(ocp)
        ocp_av_dim = pybamm.average(ocp_dim)
        dudT_av = pybamm.average(dudT)

        variables = {
            self._domain + " electrode open circuit potential": ocp,
            self._domain + " electrode open circuit potential [V]": ocp_dim,
            "Average "
            + self._domain.lower()
            + " electrode open circuit potential": ocp_av,
            "Average "
            + self._domain.lower()
            + " electrode open circuit potential [V]": ocp_av_dim,
            self._domain + " electrode entropic change": dudT,
            "Average " + self._domain.lower() + " electrode entropic change": dudT_av,
        }

        return variables

    def _flux_law(self, c):
        raise NotImplementedError

    def _unpack(self, variables):
        raise NotImplementedError

    def set_initial_conditions(self, variables):

        c, _, _ = self._unpack(variables)

        if self._domain == "Negative":
            c_init = self.param.c_n_init

        elif self._domain == "Positive":
            c_init = self.param.c_p_init

        else:
            pybamm.DomainError

        self.initial_conditions = {c: c_init}

    def set_events(self, variables):

        #
        c_s_surf = variables[self._domain + " particle surface concentration"]
        tol = 0.01
        self.events = [
            pybamm.Function(np.min, c_s_surf) - tol,
            (1 - tol) - pybamm.Function(np.max, c_s_surf),
        ]

