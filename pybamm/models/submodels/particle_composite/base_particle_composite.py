#
# Base class for particles
#
import pybamm


class BaseParticleComposite(pybamm.BaseSubModel):
    """
    Base class for molar conservation in particles.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        # reaction = "composite particle"
        super().__init__(param, domain)

    def _get_standard_concentration_variables(
        self, c_s, c_s_xav=None, c_s_rav=None, c_s_av=None, c_s_surf=None, phase=None
    ):
        """
        All particle submodels must provide the particle concentration as an argument
        to this method. Some submodels solve for quantities other than the concentration
        itself, for example the 'FickianSingleParticle' models solves for the x-averaged
        concentration. In such cases the variables being solved for (set in
        'get_fundamental_variables') must also be passed as keyword arguments. If not
        passed as keyword arguments, the various average concentrations and surface
        concentration are computed automatically from the particle concentration.
        """

        # Get surface concentration if not provided as fundamental variable to
        # solve for
        c_s_surf = c_s_surf or pybamm.surf(c_s)
        c_s_surf_av = pybamm.x_average(c_s_surf)

        if self.domain == "Negative":
            if phase == "phase 1":
                c_scale = self.param.c_n_p1_max
                p_name = " of phase 1"
            elif phase == "phase 2":
                c_scale = self.param.c_n_p2_max
                p_name = " of phase 2"
            else:
                c_scale = self.param.c_n_max
                p_name = ""
        elif self.domain == "Positive":
            if phase == "phase 1":
                c_scale = self.param.c_p_p1_max
                p_name = " of phase 1"
            elif phase == "phase 2":
                c_scale = self.param.c_p_p2_max
                p_name = " of phase 2"
            else:
                c_scale = self.param.c_p_max
                p_name = ""
            # Only composite anode is considered

        # Get average concentration(s) if not provided as fundamental variable to
        # solve for
        c_s_xav = c_s_xav or pybamm.x_average(c_s)
        c_s_rav = c_s_rav or pybamm.r_average(c_s)
        c_s_av = c_s_av or pybamm.r_average(c_s_xav)

        variables = {
            f"{self.domain} particle concentration{p_name}": c_s,
            f"{self.domain} particle concentration{p_name} [mol.m-3]": c_s * c_scale,
            f"{self.domain} particle concentration{p_name} [mol.m-3]": c_s * c_scale,
            f"X-averaged {self.domain.lower()} particle concentration{p_name}": c_s_xav,
            f"X-averaged {self.domain.lower()} particle concentration{p_name} [mol.m-3]": c_s_xav
            * c_scale,
            f"R-averaged {self.domain.lower()} particle concentration{p_name}": c_s_rav,
            f"R-averaged {self.domain.lower()} particle concentration{p_name} [mol.m-3]": c_s_rav
            * c_scale,
            f"Average {self.domain.lower()} particle concentration{p_name}": c_s_av,
            f"Average {self.domain.lower()} particle concentration{p_name} [mol.m-3]": c_s_av
            * c_scale,
            f"{self.domain} particle surface concentration{p_name}": c_s_surf,
            f"{self.domain} particle surface concentration{p_name} [mol.m-3]": c_scale
            * c_s_surf,
            f"X-averaged {self.domain.lower()} particle surface concentration{p_name}": c_s_surf_av,
            f"X-averaged {self.domain.lower()} particle surface concentration{p_name} [mol.m-3]": c_scale
            * c_s_surf_av,
            f"{self.domain} electrode extent of lithiation{p_name}": c_s_rav,
            f"X-averaged {self.domain.lower()} electrode extent of lithiation{p_name}": c_s_av,
            f"Minimum {self.domain.lower()} particle concentration{p_name}": pybamm.min(
                c_s
            ),
            f"Maximum {self.domain.lower()} particle concentration{p_name}": pybamm.max(
                c_s
            ),
            f"Minimum {self.domain.lower()} particle concentration{p_name} [mol.m-3]": pybamm.min(
                c_s
            )
            * c_scale,
            f"Maximum {self.domain.lower()} particle concentration{p_name} [mol.m-3]": pybamm.max(
                c_s
            )
            * c_scale,
            f"Minimum {self.domain.lower()} particle surface concentration{p_name}": pybamm.min(
                c_s_surf
            ),
            f"Maximum {self.domain.lower()} particle surface concentration{p_name}": pybamm.max(
                c_s_surf
            ),
            f"Minimum {self.domain.lower()} particle surface concentration{p_name} [mol.m-3]": pybamm.min(
                c_s_surf
            )
            * c_scale,
            f"Maximum {self.domain.lower()} particle surface concentration{p_name} [mol.m-3]": pybamm.max(
                c_s_surf
            )
            * c_scale,
        }

        return variables

    def _get_total_concentration_variables(self, variables, phase=None):
        if self.domain == "Negative":
            if phase == "phase 1":
                c_scale = self.param.c_n_p1_max
                p_name = " of phase 1"
            elif phase == "phase 2":
                c_scale = self.param.c_n_p2_max
                p_name = " of phase 2"
            else:
                c_scale = self.param.c_n_max
                p_name = ""
            L = self.param.L_n
        elif self.domain == "Positive":
            if phase == "phase 1":
                c_scale = self.param.c_p_p1_max
                p_name = " of phase 1"
            elif phase == "phase 2":
                c_scale = self.param.c_p_p2_max
                p_name = " of phase 2"
            else:
                c_scale = self.param.c_p_max
                p_name = ""
            L = self.param.L_n
        c_s_rav = variables[
            f"R-averaged {self.domain.lower} particle concentration{p_name}"
        ]
        eps_s = variables[
            f"{self.domain} electrode active material volume fraction{p_name}"
        ]
        c_s_vol_av = pybamm.x_average(eps_s * c_s_rav)

        A = self.param.A_cc

        variables.update(
            {
                f"{self.domain} electrode volume-averaged concentration{p_name}": c_s_vol_av,
                f"{self.domain} electrode volume-averaged concentration{p_name} [mol.m-3]": c_s_vol_av
                * c_scale,
                f"Total lithium in {self.domain.lower()} electrode [mol]{p_name}": c_s_vol_av
                * c_scale
                * L
                * A,
            }
        )
        return variables

    def _get_standard_flux_variables(self, N_s, N_s_xav, phase=None):
        if self.domain == "Negative":
            if phase == "phase 1":
                p_name = " of phase 1"
            elif phase == "phase 2":
                p_name = " of phase 2"
            else:
                p_name = ""
        elif self.domain == "Positive":
            if phase == "phase 1":
                p_name = " of phase 1"
            elif phase == "phase 2":
                p_name = " of phase 2"
            else:
                p_name = ""
        variables = {
            f"{self.domain} particle flux{p_name}": N_s,
            f"X-averaged {self.domain.lower()} particle flux{p_name}": N_s_xav,
        }
        return variables

    def set_events(self, variables, phase=None):
        if self.domain == "Negative":
            if phase == "phase 1":
                p_name = " of phase 1"
            elif phase == "phase 2":
                p_name = " of phase 2"
            else:
                p_name = ""
        elif self.domain == "Positive":
            if phase == "phase 1":
                p_name = " of phase 1"
            elif phase == "phase 2":
                p_name = " of phase 2"
            else:
                p_name = ""
        c_s_surf = variables[f"{self.domain} particle surface concentration{p_name}"]
        tol = 1e-4

        self.events.append(
            pybamm.Event(
                f"Minumum {self.domain.lower()} particle surface concentration{p_name}",
                pybamm.min(c_s_surf) - tol,
                pybamm.EventType.TERMINATION,
            )
        )

        self.events.append(
            pybamm.Event(
                f"Maximum {self.domain.lower()} particle surface concentration{p_name}",
                (1 - tol) - pybamm.max(c_s_surf),
                pybamm.EventType.TERMINATION,
            )
        )
