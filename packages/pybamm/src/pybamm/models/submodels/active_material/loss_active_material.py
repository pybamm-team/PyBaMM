#
# Class for varying active material volume fraction
#
import pybamm

from .base_active_material import BaseModel


class LossActiveMaterial(BaseModel):
    """Submodel for varying active material volume fraction from :footcite:t:`Ai2019`
    and :footcite:t:`Reniers2019`.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    options : dict
        Additional options to pass to the model
    x_average : bool
        Whether to use x-averaged variables (SPM, SPMe, etc) or full variables (DFN)

    """

    def __init__(self, param, domain, options, x_average, phase):
        super().__init__(param, domain, options=options, phase=phase)
        pybamm.citations.register("Reniers2019")
        self.x_average = x_average

    def build(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        electrode_domains = {
            "primary": f"{domain} electrode",
            "secondary": "current collector",
        }

        eps_solid_init = self.phase_param.epsilon_s
        if self.x_average is True:
            eps_solid_xav = pybamm.Variable(
                f"X-averaged {domain} electrode {phase_name}active material volume fraction",
                domain="current collector",
            )
            eps_solid = pybamm.PrimaryBroadcast(eps_solid_xav, f"{domain} electrode")
            self.initial_conditions[eps_solid_xav] = pybamm.x_average(eps_solid_init)
        else:
            eps_solid = pybamm.Variable(
                f"{Domain} electrode {phase_name}active material volume fraction",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
            self.initial_conditions[eps_solid] = eps_solid_init
        lli_name = (
            f"Loss of lithium due to loss of {phase_name}active material "
            f"in {domain} electrode [mol]"
        )
        lli_due_to_lam = pybamm.Variable(lli_name)
        self.initial_conditions[lli_due_to_lam] = pybamm.Scalar(0)

        variables = self._get_standard_active_material_variables(eps_solid)
        variables[lli_name] = lli_due_to_lam
        variables.update(
            self._get_standard_active_material_change_variables(
                self._get_active_material_loss_rate()
            )
        )
        self.variables.update(variables)

        if self.x_average is True:
            eps_solid_state = eps_solid_xav
            deps_solid_dt = variables[
                f"X-averaged {domain} electrode {phase_name}active material "
                "volume fraction change [s-1]"
            ]
        else:
            eps_solid_state = eps_solid
            deps_solid_dt = variables[
                f"{Domain} electrode {phase_name}active material volume fraction change [s-1]"
            ]

        # Loss of lithium due to loss of active material, eq 37 of Sulzer et al.,
        # J. Electrochem. Soc. 168.12 (2021): 120531
        c_s_rav = self.coupled_variable(
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]",
            domains=electrode_domains,
        )
        V = self.domain_param.L * self.param.A_cc
        # minus sign because eps_solid is decreasing and LLI measures positive
        self.rhs[lli_due_to_lam] = -V * pybamm.x_average(c_s_rav * deps_solid_dt)
        self.rhs[eps_solid_state] = deps_solid_dt

    def _get_active_material_loss_rate(self):
        """Rate of change of the active material volume fraction [s-1]."""
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        lam_option = getattr(getattr(self.options, domain), self.phase)[
            "loss of active material"
        ]
        if self.x_average is True:
            prefix = f"X-averaged {domain}"
            domains = {"primary": "current collector"}
        else:
            prefix = Domain
            domains = {
                "primary": f"{domain} electrode",
                "secondary": "current collector",
            }
        T = self.coupled_variable(
            f"{prefix} electrode temperature [K]", domains=domains
        )

        deps_solid_dt = 0
        if "stress" in lam_option:
            # loss of active material driven by particle surface stress
            stress_critical = self.phase_param.stress_critical
            m_LAM = self.phase_param.m_LAM
            stress_t_surf = self.coupled_variable(
                f"{prefix} {phase_name}particle surface tangential stress [Pa]",
                domains=domains,
            )
            # the radial stress vanishes at the particle surface, so it is a scalar
            stress_r_surf = self.coupled_variable(
                f"{prefix} {phase_name}particle surface radial stress [Pa]"
            )
            # compute hydrostatic stress
            stress_h_surf = (stress_r_surf + 2 * stress_t_surf) / 3
            # separate compressive and tensile stresses
            stress_h_surf_compressive = stress_h_surf * (stress_h_surf < 0)
            stress_h_surf_tensile = stress_h_surf * (stress_h_surf > 0)

            if "asymmetric stress" in lam_option:
                pybamm.citations.register("Pannala2024")
                # semi-empirical model for stress-driven LAM that includes both
                # compressive and tensile stresses
                beta_LAM_compressive = self.phase_param.beta_LAM(
                    T, direction="compressive"
                )
                beta_LAM_tensile = self.phase_param.beta_LAM(T, direction="tensile")
                j_stress_LAM = (
                    -beta_LAM_compressive
                    * (abs(stress_h_surf_compressive) / stress_critical) ** m_LAM
                    - beta_LAM_tensile
                    * (abs(stress_h_surf_tensile) / stress_critical) ** m_LAM
                )
            else:
                beta_LAM = self.phase_param.beta_LAM(T)
                # assuming that only tensile stress contributes and that the minimum
                # (tensile) hydrostatic stress is zero for full cycles
                stress_h_surf_min = stress_h_surf * 0
                j_stress_LAM = (
                    -beta_LAM
                    * ((stress_h_surf_tensile - stress_h_surf_min) / stress_critical)
                    ** m_LAM
                )
            deps_solid_dt += j_stress_LAM

        if "reaction" in lam_option:
            a_j_sei = self.coupled_variable(
                f"{prefix} electrode {phase_name}SEI volumetric interfacial current "
                "density [A.m-3]",
                domains=domains,
            )
            deps_solid_dt += self.phase_param.beta_LAM_sei * a_j_sei / self.param.F

        if "current" in lam_option:
            # loss of active material driven by current
            deps_solid_dt += self.domain_param.LAM_rate_current(
                self.param.current_density_with_time, T
            )

        return deps_solid_dt
