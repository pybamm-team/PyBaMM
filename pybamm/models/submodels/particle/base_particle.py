#
# Base class for particles
#
import pybamm


class BaseParticle(pybamm.BaseSubModel):
    """
    Base class for molar conservation in particles.

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

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain, options=None, phase=None):
        super().__init__(param, domain, options=options, phase=phase)

    def _get_standard_concentration_variables(
        self, c_s, c_s_xav=None, c_s_rav=None, c_s_av=None, c_s_surf=None
    ):
        """
        All particle submodels must provide the particle concentration as an argument
        to this method. Some submodels solve for quantities other than the concentration
        itself, for example the 'XAveragedFickianDiffusion' models solves for the
        x-averaged concentration. In such cases the variables being solved for (set in
        'get_fundamental_variables') must also be passed as keyword arguments. If not
        passed as keyword arguments, the various average concentrations and surface
        concentration are computed automatically from the particle concentration.
        """
        Domain = self.domain
        domain = Domain.lower()
        phase_name = self.phase_name

        # Get surface concentration if not provided as fundamental variable to
        # solve for
        c_s_surf = c_s_surf or pybamm.surf(c_s)
        c_s_surf_av = pybamm.x_average(c_s_surf)

        c_scale = self.phase_param.c_max

        # Get average concentration(s) if not provided as fundamental variable to
        # solve for
        c_s_xav = c_s_xav or pybamm.x_average(c_s)
        c_s_rav = c_s_rav or pybamm.r_average(c_s)
        c_s_av = c_s_av or pybamm.r_average(c_s_xav)

        variables = {
            f"{Domain} {phase_name}particle concentration": c_s,
            f"{Domain} {phase_name}particle concentration [mol.m-3]": c_s * c_scale,
            f"{Domain} {phase_name}particle concentration [mol.m-3]": c_s * c_scale,
            f"X-averaged {domain} {phase_name}particle concentration": c_s_xav,
            f"X-averaged {domain} {phase_name}particle concentration [mol.m-3]": c_s_xav
            * c_scale,
            f"R-averaged {domain} {phase_name}particle concentration": c_s_rav,
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]": c_s_rav
            * c_scale,
            f"Average {domain} {phase_name}particle concentration": c_s_av,
            f"Average {domain} {phase_name}particle concentration [mol.m-3]": c_s_av
            * c_scale,
            f"{Domain} {phase_name}particle " "surface concentration": c_s_surf,
            f"{Domain} particle surface concentration [mol.m-3]": c_scale * c_s_surf,
            f"X-averaged {domain} {phase_name}particle "
            "surface concentration": c_s_surf_av,
            f"X-averaged {domain} {phase_name}particle "
            "surface concentration [mol.m-3]": c_scale * c_s_surf_av,
            f"{Domain} electrode extent of lithiation": c_s_rav,
            f"X-averaged {domain} electrode extent of lithiation": c_s_av,
            f"Minimum {domain} {phase_name}particle concentration": pybamm.min(c_s),
            f"Maximum {domain} {phase_name}particle concentration": pybamm.max(c_s),
            f"Minimum {domain} {phase_name}particle concentration [mol.m-3]"
            "": pybamm.min(c_s) * c_scale,
            f"Maximum {domain} {phase_name}particle concentration [mol.m-3]"
            "": pybamm.max(c_s) * c_scale,
            f"Minimum {domain} {phase_name}particle "
            "surface concentration": pybamm.min(c_s_surf),
            f"Maximum {domain} {phase_name}particle "
            "surface concentration": pybamm.max(c_s_surf),
            f"Minimum {domain} {phase_name}particle "
            "surface concentration [mol.m-3]": pybamm.min(c_s_surf) * c_scale,
            f"Maximum {domain} {phase_name}particle "
            "surface concentration [mol.m-3]": pybamm.max(c_s_surf) * c_scale,
        }

        return variables

    def _get_total_concentration_variables(self, variables):
        Domain = self.domain
        domain = Domain.lower()
        phase = self.phase
        phase_name = self.phase_name

        c_s_rav = variables[f"R-averaged {domain} {phase_name}particle concentration"]
        eps_s = variables[
            f"{Domain} electrode {phase_name}active material volume fraction"
        ]
        eps_s_av = pybamm.x_average(eps_s)
        c_s_vol_av = pybamm.x_average(eps_s * c_s_rav) / eps_s_av
        c_scale = self.phase_param.c_max
        L = self.domain_param.L
        A = self.param.A_cc

        variables.update(
            {
                f"{Domain} electrode {phase_name}SOC": c_s_vol_av,
                f"{Domain} electrode {phase_name}volume-averaged "
                "concentration": c_s_vol_av,
                f"{Domain} electrode {phase_name}volume-averaged "
                "concentration [mol.m-3]": c_s_vol_av * c_scale,
                f"Total lithium in {phase} phase in {domain} electrode [mol]"
                "": pybamm.yz_average(c_s_vol_av * eps_s_av) * c_scale * L * A,
            }
        )
        return variables

    def _get_standard_flux_variables(self, N_s, N_s_xav):
        Domain = self.domain
        domain = Domain.lower()
        phase_name = self.phase_name

        variables = {
            f"{Domain} {phase_name}particle flux": N_s,
            f"X-averaged {domain} {phase_name}particle flux": N_s_xav,
        }

        return variables
