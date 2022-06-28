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
    phase : str
        Phase of the particle

    **Extends:** :class:`pybamm.particle.BaseParticle`
    """

    def __init__(self, param, domain, options, phase):
        super().__init__(param, domain, options, phase)

    def get_fundamental_variables(self):
        domain = self.domain.lower()

        c_s = pybamm.Variable(
            f"{self.domain} {self.phase_name}particle concentration",
            domain=f"{domain} {self.phase_name}particle",
            auxiliary_domains={
                "secondary": f"{domain} electrode",
                "tertiary": "current collector",
            },
            bounds=(0, 1),
        )

        variables = self._get_standard_concentration_variables(c_s)

        return variables

    def get_coupled_variables(self, variables):
        c_s = variables[f"{self.domain} {self.phase_name}particle concentration"]
        T = pybamm.PrimaryBroadcast(
            variables[f"{self.domain} electrode temperature"],
            [f"{self.domain.lower()} {self.phase_name}particle"],
        )

        D_eff = self._get_effective_diffusivity(c_s, T)
        N_s = -D_eff * pybamm.grad(c_s)

        variables.update(self._get_standard_flux_variables(N_s, N_s))
        variables.update(self._get_standard_diffusivity_variables(D_eff))
        variables.update(self._get_total_concentration_variables(variables))

        return variables

    def set_rhs(self, variables):
        c_s = variables[f"{self.domain} {self.phase_name}particle concentration"]
        N_s = variables[f"{self.domain} {self.phase_name}particle flux"]
        R = variables[f"{self.domain} {self.phase_name}particle radius"]

        self.rhs = {c_s: -(1 / (R ** 2 * self.phase_param.C_diff)) * pybamm.div(N_s)}

    def set_boundary_conditions(self, variables):
        phase_param = self.phase_param

        c_s = variables[f"{self.domain} {self.phase_name}particle concentration"]
        D_eff = variables[f"{self.domain} {self.phase_name}effective diffusivity"]
        j = variables[
            f"{self.domain} electrode {self.phase_name}interfacial current density"
        ]
        R = variables[f"{self.domain} {self.phase_name}particle radius"]

        rbc = (
            -phase_param.C_diff
            * j
            * R
            / phase_param.a_R
            / phase_param.gamma
            / pybamm.surf(D_eff)
        )

        self.boundary_conditions = {
            c_s: {"left": (pybamm.Scalar(0), "Neumann"), "right": (rbc, "Neumann")}
        }

    def set_initial_conditions(self, variables):
        c_s = variables[f"{self.domain} {self.phase_name}particle concentration"]
        c_init = self.phase_param.c_init
        self.initial_conditions = {c_s: c_init}
