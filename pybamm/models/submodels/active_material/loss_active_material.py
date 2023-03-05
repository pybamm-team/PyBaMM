#
# Class for varying active material volume fraction, driven by stress
#
import pybamm

from .base_active_material import BaseModel


class LossActiveMaterial(BaseModel):
    """Submodel for varying active material volume fraction from [1]_ and [2]_.

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

    **Extends:** :class:`pybamm.active_material.BaseModel`

    References
    ----------
    .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2019). Electrochemical
           Thermal-Mechanical Modelling of Stress Inhomogeneity in Lithium-Ion Pouch
           Cells. Journal of The Electrochemical Society, 167(1), 013512.
    .. [2] Reniers, J. M., Mulder, G., & Howey, D. A. (2019). Review and performance
           comparison of mechanical-chemical degradation models for lithium-ion
           batteries. Journal of The Electrochemical Society, 166(14), A3189.
    """

    def __init__(self, param, domain, options, x_average):
        super().__init__(param, domain, options=options)
        pybamm.citations.register("Reniers2019")
        self.x_average = x_average

    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain

        if self.x_average is True:
            eps_solid_xav = pybamm.Variable(
                f"X-averaged {domain} electrode active material volume fraction",
                domain="current collector",
            )
            eps_solid = pybamm.PrimaryBroadcast(eps_solid_xav, f"{domain} electrode")
        else:
            eps_solid = pybamm.Variable(
                f"{Domain} electrode active material volume fraction",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
        variables = self._get_standard_active_material_variables(eps_solid)
        lli_due_to_lam = pybamm.Variable(
            "Loss of lithium due to loss of active material "
            f"in {domain} electrode [mol]"
        )
        variables.update(
            {
                "Loss of lithium due to loss of active material "
                f"in {domain} electrode [mol]": lli_due_to_lam
            }
        )
        return variables

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain

        deps_solid_dt = 0
        lam_option = getattr(self.options, self.domain)["loss of active material"]
        if "stress" in lam_option:
            # obtain the rate of loss of active materials (LAM) by stress
            # This is loss of active material model by mechanical effects
            if self.x_average is True:
                stress_t_surf = variables[
                    f"X-averaged {domain} particle surface tangential stress [Pa]"
                ]
                stress_r_surf = variables[
                    f"X-averaged {domain} particle surface radial stress [Pa]"
                ]
            else:
                stress_t_surf = variables[
                    f"{Domain} particle surface tangential stress [Pa]"
                ]
                stress_r_surf = variables[
                    f"{Domain} particle surface radial stress [Pa]"
                ]

            beta_LAM = self.domain_param.beta_LAM
            stress_critical = self.domain_param.stress_critical
            m_LAM = self.domain_param.m_LAM

            stress_h_surf = (stress_r_surf + 2 * stress_t_surf) / 3
            # compressive stress make no contribution
            stress_h_surf *= stress_h_surf > 0
            # assuming the minimum hydrostatic stress is zero for full cycles
            stress_h_surf_min = stress_h_surf * 0
            j_stress_LAM = (
                -beta_LAM
                * ((stress_h_surf - stress_h_surf_min) / stress_critical) ** m_LAM
            )
            deps_solid_dt += j_stress_LAM

        if "reaction" in lam_option:
            beta_LAM_sei = self.domain_param.beta_LAM_sei
            if self.domain == "negative":
                if self.x_average is True:
                    a_j_sei = variables[
                        "X-averaged negative electrode SEI "
                        "volumetric interfacial current density [A.m-3]"
                    ]
                else:
                    a_j_sei = variables[
                        "Negative electrode SEI volumetric "
                        "interfacial current density [A.m-3]"
                    ]
            else:
                # No SEI in the positive electrode so no reaction-driven LAM
                # until other reactions are implemented
                a_j_sei = 0

            j_stress_reaction = beta_LAM_sei * a_j_sei / self.param.F
            deps_solid_dt += j_stress_reaction
        variables.update(
            self._get_standard_active_material_change_variables(deps_solid_dt)
        )
        return variables

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain

        if self.x_average is True:
            eps_solid = variables[
                f"X-averaged {domain} electrode active material volume fraction"
            ]
            deps_solid_dt = variables[
                f"X-averaged {domain} electrode active material "
                "volume fraction change [s-1]"
            ]
        else:
            eps_solid = variables[f"{Domain} electrode active material volume fraction"]
            deps_solid_dt = variables[
                f"{Domain} electrode active material volume fraction change [s-1]"
            ]

        # Loss of lithium due to loss of active material
        # See eq 37 in "Sulzer, Valentin, et al. "Accelerated battery lifetime
        # simulations using adaptive inter-cycle extrapolation algorithm."
        # Journal of The Electrochemical Society 168.12 (2021): 120531.
        lli_due_to_lam = variables[
            "Loss of lithium due to loss of active material "
            f"in {domain} electrode [mol]"
        ]
        # Multiply by mol.m-3 * m3 to get mol
        c_s_av = variables[f"Average {domain} particle concentration [mol.m-3]"]
        V = self.domain_param.L * self.param.A_cc

        self.rhs = {
            # minus sign because eps_solid is decreasing and LLI measures positive
            lli_due_to_lam: -c_s_av * V * pybamm.x_average(deps_solid_dt),
            eps_solid: deps_solid_dt,
        }

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain

        eps_solid_init = self.domain_param.prim.epsilon_s

        if self.x_average is True:
            eps_solid_xav = variables[
                f"X-averaged {domain} electrode active material volume fraction"
            ]
            self.initial_conditions = {eps_solid_xav: pybamm.x_average(eps_solid_init)}
        else:
            eps_solid = variables[f"{Domain} electrode active material volume fraction"]
            self.initial_conditions = {eps_solid: eps_solid_init}

        lli_due_to_lam = variables[
            "Loss of lithium due to loss of active material "
            f"in {domain} electrode [mol]"
        ]
        self.initial_conditions[lli_due_to_lam] = pybamm.Scalar(0)
