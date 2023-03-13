#
# Class for many particles with polynomial concentration profile
#
import pybamm

from .base_particle import BaseParticle


class PolynomialProfile(BaseParticle):
    """
    Class for molar conservation in particles employing Fick's
    law, assuming a polynomial concentration profile in r, and allowing variation
    in the electrode domain. Model equations from [1]_.

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

    References
    ----------
    .. [1] VR Subramanian, VD Diwakar and D Tapriyal. “Efficient Macro-Micro Scale
           Coupled Modeling of Batteries”. Journal of The Electrochemical Society,
           152(10):A2002-A2008, 2005
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

    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain

        variables = {}
        # For all orders we solve an equation for the average concentration
        if self.size_distribution is False:
            c_s_rav = pybamm.Variable(
                f"R-averaged {domain} particle concentration [mol.m-3]",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
                bounds=(0, self.phase_param.c_max),
                scale=self.phase_param.c_max,
            )
            r = pybamm.SpatialVariable(
                f"r_{domain[0]}",
                domain=[f"{domain} particle"],
                auxiliary_domains={
                    "secondary": f"{domain} electrode",
                    "tertiary": "current collector",
                },
                coord_sys="spherical polar",
            )
            R = self.phase_param.R
        else:
            c_s_rav_distribution = pybamm.Variable(
                f"R-averaged {domain} particle concentration distribution [mol.m-3]",
                domain=f"{domain} particle size",
                auxiliary_domains={
                    "secondary": f"{domain} electrode",
                    "tertiary": "current collector",
                },
                bounds=(0, self.phase_param.c_max),
                scale=self.phase_param.c_max,
            )
            r = pybamm.SpatialVariable(
                f"r_{domain[0]}",
                domain=[f"{domain} particle"],
                auxiliary_domains={
                    "secondary": f"{domain} particle size",
                    "tertiary": f"{domain} electrode",
                    "quaternary": "current collector",
                },
                coord_sys="spherical polar",
            )
            R = pybamm.SpatialVariable(
                f"R_{domain[0]}",
                domain=[f"{domain} particle size"],
                auxiliary_domains={
                    "secondary": f"{domain} electrode",
                    "tertiary": "current collector",
                },
                coord_sys="cartesian",
            )

            variables = self._get_distribution_variables(R)

            # Standard concentration distribution variables (size-dependent)
            variables.update(
                self._get_standard_concentration_distribution_variables(
                    c_s_rav_distribution
                )
            )

            # Standard size-averaged variables. Average concentrations using
            # the volume-weighted distribution since they are volume-based
            # quantities. Necessary for output variables "Total lithium in
            # negative electrode [mol]", etc, to be calculated correctly
            f_v_dist = variables[
                f"{Domain} volume-weighted particle-size distribution [m-1]"
            ]
            c_s_rav = pybamm.Integral(f_v_dist * c_s_rav_distribution, R)
            c_s_surf = c_s_rav
            c_s = pybamm.PrimaryBroadcast(c_s_rav, [f"{domain} particle"])

            variables.update(
                self._get_standard_concentration_variables(
                    c_s, c_s_rav=c_s_rav, c_s_surf=c_s_surf
                )
            )

            return variables

        if self.name == "uniform profile":
            # The concentration is uniform so the surface value is equal to
            # the average
            c_s_surf = c_s_rav
        elif self.name in ["quadratic profile", "quartic profile"]:
            # We solve an equation for the surface concentration, so it is
            # a variable in the model
            c_s_surf = pybamm.Variable(
                f"{Domain} particle surface concentration [mol.m-3]",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
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
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
                scale=self.phase_param.c_max / self.phase_param.R_typ,
            )
            variables.update(
                {
                    f"R-averaged {domain} particle "
                    "concentration gradient [mol.m-4]": q_s_rav
                }
            )

        # Set concentration depending on polynomial order
        if self.name == "uniform profile":
            # The concentration is uniform
            A = c_s_rav
            B = pybamm.FullBroadcast(0, f"{domain} electrode", "current collector")
            C = pybamm.FullBroadcast(0, f"{domain} electrode", "current collector")
        elif self.name == "quadratic profile":
            # The concentration is given by c = A + B*r**2
            # eqs 11-12 in Subramanian2005
            A = 5 / 2 * c_s_rav - 3 / 2 * c_s_surf
            B = 5 / 2 * (c_s_surf - c_s_rav)
            C = pybamm.FullBroadcast(0, f"{domain} electrode", "current collector")
        elif self.name == "quartic profile":
            # The concentration is given by c = A + B*r**2 + C*r**4
            # eqs 24-26 in Subramanian2005
            A = 39 / 4 * c_s_surf - 3 * q_s_rav * R - 35 / 4 * c_s_rav
            B = -35 * c_s_surf + 10 * q_s_rav + 35 * c_s_rav
            C = 105 / 4 * c_s_surf - 7 * q_s_rav * R - 105 / 4 * c_s_rav
        A = pybamm.PrimaryBroadcast(A, [f"{domain} particle"])
        B = pybamm.PrimaryBroadcast(B, [f"{domain} particle"])
        C = pybamm.PrimaryBroadcast(C, [f"{domain} particle"])

        c_s = A + B * r**2 / R**2 + C * r**4 / R**4

        variables.update(
            self._get_standard_concentration_variables(
                c_s, c_s_rav=c_s_rav, c_s_surf=c_s_surf
            )
        )

        return variables

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain

        if self.size_distribution is False:
            c_s = variables[f"{Domain} particle concentration [mol.m-3]"]
            c_s_rav = variables[f"R-averaged {domain} particle concentration [mol.m-3]"]
            c_s_surf = variables[f"{Domain} particle surface concentration [mol.m-3]"]
            T = pybamm.PrimaryBroadcast(
                variables[f"{Domain} electrode temperature [K]"], [f"{domain} particle"]
            )
            D_eff = self._get_effective_diffusivity(c_s, T)
            r = pybamm.SpatialVariable(
                f"r_{domain[0]}",
                domain=[f"{domain} particle"],
                auxiliary_domains={
                    "secondary": f"{domain} electrode",
                    "tertiary": "current collector",
                },
                coord_sys="spherical polar",
            )
            R = variables[f"{Domain} particle radius [m]"]
            variables.update(self._get_standard_diffusivity_variables(D_eff))
        else:
            # only uniform concentration implemented, no need to calculate D_eff
            pass

        # Set flux depending on polynomial order
        if self.name == "uniform profile":
            # The flux is zero since there is no concentration gradient
            N_s = pybamm.FullBroadcastToEdges(
                0,
                [f"{domain} particle"],
                auxiliary_domains={
                    "secondary": f"{domain} electrode",
                    "tertiary": "current collector",
                },
            )
        elif self.name == "quadratic profile":
            # The flux may be computed directly from the polynomial for c
            N_s = -D_eff * 5 * (c_s_surf - c_s_rav) * r / R**2
        elif self.name == "quartic profile":
            q_s_rav = variables[
                f"R-averaged {domain} particle concentration gradient [mol.m-4]"
            ]
            # The flux may be computed directly from the polynomial for c
            N_s = -D_eff * (
                (-70 * c_s_surf + 20 * q_s_rav * R + 70 * c_s_rav) * r / R**2
                + (105 * c_s_surf - 28 * q_s_rav * R - 105 * c_s_rav) * r**3 / R**4
            )

        variables.update(self._get_standard_flux_variables(N_s))

        return variables

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain

        if self.size_distribution is False:
            c_s_rav = variables[f"R-averaged {domain} particle concentration [mol.m-3]"]
            j = variables[f"{Domain} electrode interfacial current density [A.m-2]"]
            R = variables[f"{Domain} particle radius [m]"]
        else:
            c_s_rav = variables[
                f"R-averaged {domain} particle concentration distribution [mol.m-3]"
            ]
            j = variables[
                f"{Domain} electrode interfacial current density distribution [A.m-2]"
            ]
            R = variables[f"{Domain} particle sizes [m]"]

        self.rhs = {c_s_rav: -3 * j / self.param.F / R}

        if self.name == "quartic profile":
            # We solve an extra ODE for the average particle flux
            q_s_rav = variables[
                f"R-averaged {domain} particle concentration gradient [mol.m-4]"
            ]
            c_s_rav = variables[f"R-averaged {domain} particle concentration [mol.m-3]"]
            D_eff = variables[f"{Domain} particle effective diffusivity [m2.s-1]"]

            # eq 30 of Subramanian2005
            self.rhs.update(
                {
                    q_s_rav: -30 * pybamm.r_average(D_eff) * q_s_rav / R**2
                    - 45 / 2 * j / self.param.F / R**2
                }
            )

    def set_algebraic(self, variables):
        if self.name == "uniform profile":
            # No algebraic equations since we only solve for the average concentration
            return

        domain, Domain = self.domain_Domain

        c_s_surf = variables[f"{Domain} particle surface concentration [mol.m-3]"]
        c_s_rav = variables[f"R-averaged {domain} particle concentration [mol.m-3]"]
        D_eff = variables[f"{Domain} particle effective diffusivity [m2.s-1]"]
        j = variables[f"{Domain} electrode interfacial current density [A.m-2]"]
        R = variables[f"{Domain} particle radius [m]"]

        c_max = self.phase_param.c_max
        T_ref = self.param.T_ref
        D_c_max_scale = self.phase_param.D(c_max, T_ref) * c_max

        if self.name == "quadratic profile":
            # We solve an algebraic equation for the surface concentration
            self.algebraic = {
                c_s_surf: (
                    pybamm.surf(D_eff) * (c_s_surf - c_s_rav) + j * R / self.param.F / 5
                )
                / D_c_max_scale
            }

        elif self.name == "quartic profile":
            # We solve a different algebraic equation for the surface concentration
            # that accounts for the average concentration gradient inside the particle
            q_s_rav = variables[
                f"R-averaged {domain} particle concentration gradient [mol.m-4]"
            ]
            # eq 31 of Subramanian2005
            D_c_max_over_R_scale = D_c_max_scale / self.phase_param.R_typ
            self.algebraic = {
                c_s_surf: (
                    pybamm.surf(D_eff) * (35 / R * (c_s_surf - c_s_rav) - 8 * q_s_rav)
                    + j / self.param.F
                )
                / D_c_max_over_R_scale
            }

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain

        c_init = pybamm.r_average(self.phase_param.c_init)

        if self.size_distribution is False:
            c_s_rav = variables[f"R-averaged {domain} particle concentration [mol.m-3]"]
        else:
            c_s_rav = variables[
                f"R-averaged {domain} particle concentration distribution [mol.m-3]"
            ]
            c_init = pybamm.PrimaryBroadcast(c_init, [f"{domain} particle size"])

        self.initial_conditions = {c_s_rav: c_init}

        if self.name in ["quadratic profile", "quartic profile"]:
            # We also need to provide an initial condition (initial guess for the
            # algebraic solver) for the surface concentration
            c_s_surf = variables[f"{Domain} particle surface concentration [mol.m-3]"]
            self.initial_conditions.update({c_s_surf: c_init})
        if self.name == "quartic profile":
            # We also need to provide an initial condition for the average
            # concentration gradient
            q_s_rav = variables[
                f"R-averaged {domain} particle concentration gradient [mol.m-4]"
            ]
            self.initial_conditions.update({q_s_rav: 0})
