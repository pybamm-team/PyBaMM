#
# Class for many particles with polynomial concentration profile
#
import pybamm

from .base_particle import BaseParticle


class PolynomialProfile(BaseParticle):
    """
    Class for molar conservation in particles employing Fick's
    law, assuming a polynomial concentration profile in r, and allowing variation
    in the electrode domain. Model equations from :footcite:t:`Subramanian2005`.

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
        self.name = getattr(self.options, self.domain)["particle"]
        if self.name == "Fickian diffusion":
            raise ValueError(
                "Particle type must be 'uniform profile', "
                "'quadratic profile' or 'quartic profile'"
            )

        pybamm.citations.register("Subramanian2005")

    def build(self):
        domain, Domain = self.domain_Domain
        j = self._coupled_interfacial_current_density()
        c_init = pybamm.r_average(self.phase_param.c_init)

        if self.size_distribution:
            # Only the uniform profile is implemented for size distributions
            c_s_rav_distribution = pybamm.Variable(
                f"R-averaged {domain} particle concentration distribution [mol.m-3]",
                domains=self._domains(f"{domain} particle size"),
                bounds=(0, self.phase_param.c_max),
                scale=self.phase_param.c_max,
            )
            R, f_v_dist, variables = self._get_size_distribution()
            variables.update(
                self._get_standard_concentration_distribution_variables(
                    c_s_rav_distribution
                )
            )
            # Size-average with the volume-weighted distribution, since the
            # concentration is volume-based (e.g. for the total lithium)
            c_s_rav = pybamm.Integral(f_v_dist * c_s_rav_distribution, R)
            c_s = pybamm.PrimaryBroadcast(c_s_rav, [f"{domain} particle"])
            variables.update(
                self._get_standard_concentration_variables(
                    c_s, c_s_rav=c_s_rav, c_s_surf=c_s_rav
                )
            )
            variables.update(self._get_standard_flux_variables(self._zero_flux()))
            self.variables.update(variables)

            self.rhs = {c_s_rav_distribution: -3 * j / self.param.F / R}
            self.initial_conditions = {
                c_s_rav_distribution: pybamm.PrimaryBroadcast(
                    c_init, [f"{domain} particle size"]
                )
            }
            return

        T = self._broadcast_to_particle(self._coupled_temperature())
        current = self.coupled_variable("Total current density [A.m-2]")
        R = self.coupled_variable(
            f"{Domain} particle radius [m]", domains=self._domains()
        )

        # For all orders we solve an equation for the average concentration
        c_s_rav = pybamm.Variable(
            f"R-averaged {domain} particle concentration [mol.m-3]",
            domains=self._domains(),
            bounds=(0, self.phase_param.c_max),
            scale=self.phase_param.c_max,
        )
        variables = {}
        if self.name == "uniform profile":
            # The concentration is uniform so the surface value is equal to
            # the average
            c_s_surf = c_s_rav
        else:
            # We solve an equation for the surface concentration, so it is
            # a variable in the model
            c_s_surf = pybamm.Variable(
                f"{Domain} particle surface concentration [mol.m-3]",
                domains=self._domains(),
                bounds=(0, self.phase_param.c_max),
                scale=self.phase_param.c_max,
            )
        if self.name == "quartic profile":
            # For the fourth order polynomial approximation we also solve an
            # equation for the average concentration gradient. Note: in the original
            # paper this quantity is referred to as the flux, but here we make the
            # distinction between the flux defined as N = -D*dc/dr and the
            # concentration gradient q = dc/dr
            q_s_rav = pybamm.Variable(
                f"R-averaged {domain} particle concentration gradient [mol.m-4]",
                domains=self._domains(),
                scale=self.phase_param.c_max / self.phase_param.R_typ,
            )
            variables[
                f"R-averaged {domain} particle concentration gradient [mol.m-4]"
            ] = q_s_rav

        # Set concentration depending on polynomial order
        R_param = self.phase_param.R
        zero = pybamm.FullBroadcast(0, f"{domain} electrode", "current collector")
        if self.name == "uniform profile":
            # The concentration is uniform
            A, B, C = c_s_rav, zero, zero
        elif self.name == "quadratic profile":
            # The concentration is given by c = A + B*r**2
            # eqs 11-12 in Subramanian2005
            A = 5 / 2 * c_s_rav - 3 / 2 * c_s_surf
            B = 5 / 2 * (c_s_surf - c_s_rav)
            C = zero
        else:  # "quartic profile"
            # The concentration is given by c = A + B*r**2 + C*r**4
            # eqs 24-26 in Subramanian2005
            A = 39 / 4 * c_s_surf - 3 * q_s_rav * R_param - 35 / 4 * c_s_rav
            B = -35 * c_s_surf + 10 * q_s_rav * R_param + 35 * c_s_rav
            C = 105 / 4 * c_s_surf - 7 * q_s_rav * R_param - 105 / 4 * c_s_rav
        A, B, C = (
            pybamm.PrimaryBroadcast(X, [f"{domain} particle"]) for X in (A, B, C)
        )
        r = pybamm.SpatialVariable(
            f"r_{domain[0]}",
            domains=self._domains(f"{domain} particle"),
            coord_sys="spherical polar",
        )
        c_s = A + B * r**2 / R_param**2 + C * r**4 / R_param**4
        variables.update(
            self._get_standard_concentration_variables(
                c_s, c_s_rav=c_s_rav, c_s_surf=c_s_surf
            )
        )

        D_eff = self._get_effective_diffusivity(c_s, T, current)
        variables.update(self._get_standard_diffusivity_variables(D_eff))

        # Set flux depending on polynomial order
        if self.name == "uniform profile":
            # The flux is zero since there is no concentration gradient
            N_s = self._zero_flux()
        elif self.name == "quadratic profile":
            # The flux may be computed directly from the polynomial for c
            N_s = -D_eff * 5 * (c_s_surf - c_s_rav) * r / R**2
        elif self.name == "quartic profile":
            # The flux may be computed directly from the polynomial for c
            N_s = -D_eff * (
                (-70 * c_s_surf + 20 * q_s_rav * R + 70 * c_s_rav) * r / R**2
                + (105 * c_s_surf - 28 * q_s_rav * R - 105 * c_s_rav) * r**3 / R**4
            )
        variables.update(self._get_standard_flux_variables(N_s))
        self.variables.update(variables)

        self.rhs = {c_s_rav: -3 * j / self.param.F / R}
        self.initial_conditions = {c_s_rav: c_init}

        c_max = self.phase_param.c_max
        D_c_max_scale = self.phase_param.D(c_max, self.param.T_ref) * c_max
        if self.name == "quadratic profile":
            # We solve an algebraic equation for the surface concentration
            self.algebraic = {
                c_s_surf: (
                    pybamm.surf(D_eff) * (c_s_surf - c_s_rav) + j * R / self.param.F / 5
                )
                / D_c_max_scale
            }
        elif self.name == "quartic profile":
            # eq 30 of Subramanian2005: an extra ODE for the average concentration
            # gradient
            self.rhs[q_s_rav] = (
                -30 * pybamm.r_average(D_eff) * q_s_rav / R**2
                - 45 / 2 * j / self.param.F / R**2
            )
            self.initial_conditions[q_s_rav] = 0
            # eq 31 of Subramanian2005: the algebraic equation for the surface
            # concentration accounts for the average concentration gradient
            D_c_max_over_R_scale = D_c_max_scale / self.phase_param.R_typ
            self.algebraic = {
                c_s_surf: (
                    pybamm.surf(D_eff) * (35 / R * (c_s_surf - c_s_rav) - 8 * q_s_rav)
                    + j / self.param.F
                )
                / D_c_max_over_R_scale
            }
        if self.name != "uniform profile":
            # Initial guess for the algebraic solver
            self.initial_conditions[c_s_surf] = c_init

    def _zero_flux(self):
        """Particle flux [mol.m-2.s-1] of a uniform concentration profile."""
        domain = self.domain
        return pybamm.FullBroadcastToEdges(
            0,
            [f"{domain} particle"],
            auxiliary_domains={
                "secondary": f"{domain} electrode",
                "tertiary": "current collector",
            },
        )
