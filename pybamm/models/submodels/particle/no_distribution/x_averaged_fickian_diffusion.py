#
# Class for a single x-averaged particle with Fickian diffusion
#
import pybamm

from .base_fickian import BaseFickian


class XAveragedFickianDiffusion(BaseFickian):
    """
    Class for molar conservation in a single x-averaged particle, employing Fick's
    law. I.e., the concentration varies with r (internal spherical coordinate)
    but not x (electrode coordinate).

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
            c_s_xav = pybamm.standard_variables.c_s_n_xav
            c_s = pybamm.SecondaryBroadcast(c_s_xav, ["negative electrode"])

        elif self.domain == "Positive":
            c_s_xav = pybamm.standard_variables.c_s_p_xav
            c_s = pybamm.SecondaryBroadcast(c_s_xav, ["positive electrode"])

        variables = self._get_standard_concentration_variables(c_s, c_s_xav=c_s_xav)

        return variables

    def get_coupled_variables(self, variables):
        c_s_xav = variables[
            "X-averaged " + self.domain.lower() + " particle concentration"
        ]
        T_xav = pybamm.PrimaryBroadcast(
            variables["X-averaged " + self.domain.lower() + " electrode temperature"],
            [self.domain.lower() + " particle"],
        )

        D_eff_xav = self._get_effective_diffusivity(c_s_xav, T_xav)
        N_s_xav = -D_eff_xav * pybamm.grad(c_s_xav)

        D_eff = pybamm.SecondaryBroadcast(
            D_eff_xav, [self._domain.lower() + " electrode"]
        )
        N_s = pybamm.SecondaryBroadcast(N_s_xav, [self._domain.lower() + " electrode"])

        variables.update(self._get_standard_flux_variables(N_s, N_s_xav))
        variables.update(self._get_standard_diffusivity_variables(D_eff))
        variables.update(self._get_total_concentration_variables(variables))

        return variables

    def set_rhs(self, variables):
        c_s_xav = variables[
            "X-averaged " + self.domain.lower() + " particle concentration"
        ]
        N_s_xav = variables["X-averaged " + self.domain.lower() + " particle flux"]

        if self.domain == "Negative":
            self.rhs = {c_s_xav: -(1 / self.param.C_n) * pybamm.div(N_s_xav)}

        elif self.domain == "Positive":
            self.rhs = {c_s_xav: -(1 / self.param.C_p) * pybamm.div(N_s_xav)}

    def set_boundary_conditions(self, variables):
        c_s_xav = variables[
            "X-averaged " + self.domain.lower() + " particle concentration"
        ]
        D_eff_xav = variables[
            "X-averaged " + self.domain.lower() + " effective diffusivity"
        ]
        j_xav = variables[
            "X-averaged "
            + self.domain.lower()
            + " electrode interfacial current density"
        ]

        if self.domain == "Negative":
            rbc = (
                -self.param.C_n
                * j_xav
                / self.param.a_R_n
                / self.param.gamma_n
                / pybamm.surf(D_eff_xav)
            )

        elif self.domain == "Positive":
            rbc = (
                -self.param.C_p
                * j_xav
                / self.param.a_R_p
                / self.param.gamma_p
                / pybamm.surf(D_eff_xav)
            )

        self.boundary_conditions = {
            c_s_xav: {"left": (pybamm.Scalar(0), "Neumann"), "right": (rbc, "Neumann")}
        }

    def set_initial_conditions(self, variables):
        """
        For single or x-averaged particle models, initial conditions can't depend on x
        so we take the x-average of the supplied initial conditions.
        """
        c_s_xav = variables[
            "X-averaged " + self.domain.lower() + " particle concentration"
        ]

        if self.domain == "Negative":
            c_init = pybamm.x_average(self.param.c_n_init)
        elif self.domain == "Positive":
            c_init = pybamm.x_average(self.param.c_p_init)

        self.initial_conditions = {c_s_xav: c_init}
