#
# Class for single particle with polynomial concentration profile
#
import pybamm

from .polynomial_profile import PolynomialProfile


class XAveragedPolynomialProfile(PolynomialProfile):
    """
    Class for molar conservation in a single x-averaged particle employing Fick's law,
    with an assumed polynomial concentration profile in r. Model equations from
    :footcite:t:`Subramanian2005`.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")

    """

    def __init__(self, param, domain, options, phase="primary"):
        super().__init__(param, domain, options, phase)
        self.x_average = True

    def build(self):
        domain = self.domain
        F = self.param.F
        j_xav = self._coupled_interfacial_current_density()
        R = self.coupled_variable(
            f"X-averaged {domain} particle radius [m]", domain="current collector"
        )

        # For all orders we solve an equation for the average concentration
        if self.size_distribution:
            c_s_av_distribution = pybamm.Variable(
                f"Average {domain} particle concentration distribution [mol.m-3]",
                domains=self._domains(f"{domain} particle size"),
                bounds=(0, self.phase_param.c_max),
                scale=self.phase_param.c_max,
            )
            R_dist, f_v_dist, variables = self._get_size_distribution()
            variables.update(
                self._get_standard_concentration_distribution_variables(
                    c_s_av_distribution
                )
            )
            # Size-average with the volume-weighted distribution, since the
            # concentration is volume-based (e.g. for the total lithium)
            c_s_av = pybamm.Integral(f_v_dist * c_s_av_distribution, R_dist)
        else:
            c_s_av = pybamm.Variable(
                f"Average {domain} particle concentration [mol.m-3]",
                domain="current collector",
                bounds=(0, self.phase_param.c_max),
                scale=self.phase_param.c_max,
            )
            variables = {}
        variables[f"Average {domain} particle concentration [mol.m-3]"] = c_s_av

        if self.name == "quartic profile":
            # For the fourth order polynomial we also solve for the average
            # concentration gradient q = dc/dr (called the flux in Subramanian2005)
            q_s_av = pybamm.Variable(
                f"Average {domain} particle concentration gradient [mol.m-4]",
                domain="current collector",
                scale=self.phase_param.c_max / self.phase_param.R_typ,
            )
            variables[f"Average {domain} particle concentration gradient [mol.m-4]"] = (
                q_s_av
            )

        if self.name != "uniform profile":
            T_av = self._coupled_temperature()
            current = self.coupled_variable("Total current density [A.m-2]")
            D_eff_av = self._get_effective_diffusivity(c_s_av, T_av, current)
            # The average interfacial current density is written out explicitly,
            # since the interface submodel needs the surface concentration first
            i_boundary_cc = self.coupled_variable(
                "Current collector current density [A.m-2]",
                domain="current collector",
            )
            a_av = self.coupled_variable(
                f"X-averaged {domain} electrode surface area to volume ratio [m-1]",
                domain="current collector",
            )
            sgn = 1 if domain == "negative" else -1
            j_xav_explicit = sgn * i_boundary_cc / (a_av * self.domain_param.L)

        # Set surface concentration based on polynomial order
        if self.name == "uniform profile":
            # The concentration is uniform so the surface value is equal to
            # the average
            c_s_surf_xav = c_s_av
        elif self.name == "quadratic profile":
            # D should be evaluated at the surface concentration, but using the
            # average keeps the SPM(e) a system of ODEs rather than DAEs
            c_s_surf_xav = c_s_av - (j_xav_explicit * R / 5 / F / D_eff_av)
        elif self.name == "quartic profile":
            # eq 31 of Subramanian2005 (see the notes for the quadratic profile)
            c_s_surf_xav = c_s_av + R / 35 * (
                8 * q_s_av - (j_xav_explicit / F / D_eff_av)
            )

        # Set concentration depending on polynomial order. The spatial variable r
        # has no "electrode" domain since c_s_xav doesn't depend on x
        r = pybamm.SpatialVariable(
            f"r_{domain[0]}",
            domain=[f"{domain} particle"],
            auxiliary_domains={"secondary": "current collector"},
            coord_sys="spherical polar",
        )
        zero = pybamm.PrimaryBroadcast(0, "current collector")
        if self.name == "uniform profile":
            # The concentration is uniform
            A, B, C = c_s_av, zero, zero
        elif self.name == "quadratic profile":
            # The concentration is given by c = A + B*r**2
            # eqs 11-12 in Subramanian2005
            A = 5 / 2 * c_s_av - 3 / 2 * c_s_surf_xav
            B = (5 / 2) * (c_s_surf_xav - c_s_av)
            C = zero
        elif self.name == "quartic profile":
            # The concentration is given by c = A + B*r**2 + C*r**4
            # eqs 24-26 in Subramanian2005
            A = 39 / 4 * c_s_surf_xav - 3 * q_s_av * R - 35 / 4 * c_s_av
            B = -35 * c_s_surf_xav + 10 * q_s_av * R + 35 * c_s_av
            C = 105 / 4 * c_s_surf_xav - 7 * q_s_av * R - 105 / 4 * c_s_av
        A, B, C = (
            pybamm.PrimaryBroadcast(X, [f"{domain} particle"]) for X in (A, B, C)
        )
        c_s_xav = A + B * r**2 / R**2 + C * r**4 / R**4
        c_s = pybamm.SecondaryBroadcast(c_s_xav, [f"{domain} electrode"])
        c_s_surf = pybamm.PrimaryBroadcast(c_s_surf_xav, [f"{domain} electrode"])

        # Set flux based on polynomial order
        if self.name == "uniform profile":
            # The flux is zero since there is no concentration gradient
            N_s_xav = pybamm.FullBroadcastToEdges(
                0, f"{domain} particle", "current collector"
            )
        else:
            T_xav = pybamm.PrimaryBroadcast(T_av, [f"{domain} particle"])
            D_eff_xav = self._get_effective_diffusivity(c_s_xav, T_xav, current)
            D_eff = pybamm.SecondaryBroadcast(D_eff_xav, [f"{domain} electrode"])
            variables.update(self._get_standard_diffusivity_variables(D_eff))
        if self.name == "quadratic profile":
            # The flux may be computed directly from the polynomial for c
            N_s_xav = -D_eff_xav * 5 * (c_s_surf_xav - c_s_av) * r / R**2
        elif self.name == "quartic profile":
            # The flux may be computed directly from the polynomial for c
            N_s_xav = -D_eff_xav * (
                (-70 * c_s_surf_xav + 20 * q_s_av * R + 70 * c_s_av) * r / R**2
                + (105 * c_s_surf_xav - 28 * q_s_av * R - 105 * c_s_av) * (r**3 / R**4)
            )
        N_s = pybamm.SecondaryBroadcast(N_s_xav, [f"{domain} electrode"])

        variables.update(
            self._get_standard_concentration_variables(
                c_s, c_s_av=c_s_av, c_s_surf=c_s_surf
            )
        )
        variables.update(self._get_standard_flux_variables(N_s))
        self.variables.update(variables)

        # Initial conditions can't depend on x or r, so take their averages
        c_init = pybamm.x_average(pybamm.r_average(self.phase_param.c_init))
        if self.size_distribution:
            # eq 15 of Subramanian2005, equivalent to dcdt = -i_cc / (eps * F * L)
            R_xav = variables[f"X-averaged {domain} particle sizes [m]"]
            self.rhs = {c_s_av_distribution: -3 * j_xav / F / R_xav}
            self.initial_conditions = {
                c_s_av_distribution: pybamm.PrimaryBroadcast(
                    c_init, f"{domain} particle size"
                )
            }
        else:
            # `pybamm.source` multiplies the scalar source term by the correct mass
            # matrix for 2D current collectors with the finite element method
            self.rhs = {c_s_av: pybamm.source(-3 * j_xav / F / R, c_s_av)}
            self.initial_conditions = {c_s_av: c_init}

        if self.name == "quartic profile":
            # eq 30 of Subramanian2005: an extra ODE for the average particle
            # concentration gradient
            dqdt = (
                -30 * pybamm.surf(D_eff_xav) * q_s_av / R**2 - 45 / 2 * j_xav / F / R**2
            )
            self.rhs[q_s_av] = pybamm.source(dqdt, q_s_av)
            self.initial_conditions[q_s_av] = 0
