#
# Class for particles with Fickian diffusion and x-dependence
#
import pybamm
from .base_fickian import BaseFickian


class FickianDiffusion(BaseFickian):
    """
    Class for molar conservation in particles, employing Fick's law, and allowing
    variation in the electrode domain. I.e., the concentration varies with r
    (internal coordinate) and x (electrode coordinate).

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

    def get_fundamental_variables(self):
        if self.domain == "Negative":
            c_s = pybamm.standard_variables.c_s_n

        elif self.domain == "Positive":
            c_s = pybamm.standard_variables.c_s_p

        variables = self._get_standard_concentration_variables(c_s)

        return variables

    def get_coupled_variables(self, variables):
        c_s = variables[self.domain + " particle concentration"]
        T = pybamm.PrimaryBroadcast(
            variables[self.domain + " electrode temperature"],
            [self.domain.lower() + " particle"],
        )

        D_eff = self._get_effective_diffusivity(c_s, T)
        N_s = -D_eff * pybamm.grad(c_s)

        variables.update(self._get_standard_flux_variables(N_s, N_s))
        variables.update(self._get_standard_diffusivity_variables(D_eff))
        variables.update(self._get_total_concentration_variables(variables))

        return variables

    def set_rhs(self, variables):
        c_s = variables[self.domain + " particle concentration"]
        N_s = variables[self.domain + " particle flux"]
        R = variables[self.domain + " particle radius"]

        if self.domain == "Negative":
            self.rhs = {c_s: -(1 / (R ** 2 * self.param.C_n)) * pybamm.div(N_s)}

        elif self.domain == "Positive":
            self.rhs = {c_s: -(1 / (R ** 2 * self.param.C_p)) * pybamm.div(N_s)}

    def set_boundary_conditions(self, variables):

        c_s = variables[self.domain + " particle concentration"]
        D_eff = variables[self.domain + " effective diffusivity"]
        j = variables[self.domain + " electrode interfacial current density"]
        R = variables[self.domain + " particle radius"]

        if self.domain == "Negative":
            rbc = (
                -self.param.C_n
                * j
                * R
                / self.param.a_R_n
                / self.param.gamma_n
                / pybamm.surf(D_eff)
            )

        elif self.domain == "Positive":
            rbc = (
                -self.param.C_p
                * j
                * R
                / self.param.a_R_p
                / self.param.gamma_p
                / pybamm.surf(D_eff)
            )

        self.boundary_conditions = {
            c_s: {"left": (pybamm.Scalar(0), "Neumann"), "right": (rbc, "Neumann")}
        }

    def set_initial_conditions(self, variables):
        c_s = variables[self.domain + " particle concentration"]
        if self.domain == "Negative":
            c_init = self.param.c_n_init
        elif self.domain == "Positive":
            c_init = self.param.c_p_init
        self.initial_conditions = {c_s: c_init}
