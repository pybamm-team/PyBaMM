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
    phase : str
        Phase of the particle

    **Extends:** :class:`pybamm.particle.BaseParticle`
    """

    def __init__(self, param, domain, options, phase):
        super().__init__(param, domain, options, phase)

    def get_fundamental_variables(self):
        domain = self.domain.lower()
        c_s_xav = pybamm.Variable(
            f"X-averaged {domain} {self.phase_name}particle concentration",
            domain=f"{domain} {self.phase_name}particle",
            auxiliary_domains={"secondary": "current collector"},
            bounds=(0, 1),
        )
        c_s = pybamm.SecondaryBroadcast(c_s_xav, [f"{domain} electrode"])
        variables = self._get_standard_concentration_variables(c_s, c_s_xav=c_s_xav)

        return variables

    def get_coupled_variables(self, variables):
        domain = self.domain.lower()
        phase_name = self.phase_name

        c_s_xav = variables[f"X-averaged {domain} {phase_name}particle concentration"]
        T_xav = pybamm.PrimaryBroadcast(
            variables[f"X-averaged {domain} electrode temperature"],
            [f"{domain} {phase_name}particle"],
        )

        D_eff_xav = self._get_effective_diffusivity(c_s_xav, T_xav)
        N_s_xav = -D_eff_xav * pybamm.grad(c_s_xav)

        D_eff = pybamm.SecondaryBroadcast(D_eff_xav, [f"{domain} electrode"])
        N_s = pybamm.SecondaryBroadcast(N_s_xav, [f"{domain} electrode"])

        variables.update(self._get_standard_flux_variables(N_s, N_s_xav))
        variables.update(self._get_standard_diffusivity_variables(D_eff))
        variables.update(self._get_total_concentration_variables(variables))

        return variables

    def set_rhs(self, variables):
        domain = self.domain.lower()
        phase_name = self.phase_name
        c_s_xav = variables[f"X-averaged {domain} {phase_name}particle concentration"]
        N_s_xav = variables[f"X-averaged {domain} {phase_name}particle flux"]

        self.rhs = {c_s_xav: -(1 / self.phase_param.C_diff) * pybamm.div(N_s_xav)}

    def set_boundary_conditions(self, variables):
        phase_param = self.phase_param
        domain = self.domain.lower()
        phase_name = self.phase_name
        c_s_xav = variables[f"X-averaged {domain} {phase_name}particle concentration"]
        D_eff_xav = variables[f"X-averaged {domain} {phase_name}effective diffusivity"]
        j_xav = variables[
            f"X-averaged {domain} electrode {phase_name}interfacial current density"
        ]

        rbc = (
            -phase_param.C_diff
            * j_xav
            / phase_param.a_R
            / phase_param.gamma
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
        domain = self.domain.lower()
        phase_name = self.phase_name

        c_s_xav = variables[f"X-averaged {domain} {phase_name}particle concentration"]

        c_init = pybamm.x_average(self.phase_param.c_init)

        self.initial_conditions = {c_s_xav: c_init}
