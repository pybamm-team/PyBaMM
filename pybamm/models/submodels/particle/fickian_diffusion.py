#
# Class for particles with Fickian diffusion
#
import pybamm
from .base_particle import BaseParticle


class FickianDiffusion(BaseParticle):
    """
    Class for molar conservation in particles, employing Fick's law

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
    x_average : bool
        Whether the particle concentration is averaged over the x-direction
    """

    def __init__(self, param, domain, options, phase="primary", x_average=False):
        super().__init__(param, domain, options, phase)
        self.x_average = x_average

    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        variables = {}
        if self.size_distribution is False:
            if self.x_average is False:
                c_s = pybamm.Variable(
                    f"{Domain} {phase_name}particle concentration [mol.m-3]",
                    f"{domain} {phase_name}particle",
                    auxiliary_domains={
                        "secondary": f"{domain} electrode",
                        "tertiary": "current collector",
                    },
                    bounds=(0, self.phase_param.c_max),
                    scale=self.phase_param.c_max,
                )
                c_s.print_name = f"c_s_{domain[0]}"
            else:
                c_s_xav = pybamm.Variable(
                    f"X-averaged {domain} {phase_name}particle concentration [mol.m-3]",
                    f"{domain} {phase_name}particle",
                    auxiliary_domains={"secondary": "current collector"},
                    bounds=(0, self.phase_param.c_max),
                    scale=self.phase_param.c_max,
                )
                c_s_xav.print_name = f"c_s_{domain[0]}_xav"
                c_s = pybamm.SecondaryBroadcast(c_s_xav, f"{domain} electrode")
        else:
            if self.x_average is False:
                c_s_distribution = pybamm.Variable(
                    f"{Domain} {phase_name}particle "
                    "concentration distribution [mol.m-3]",
                    domain=f"{domain} {phase_name}particle",
                    auxiliary_domains={
                        "secondary": f"{domain} {phase_name}particle size",
                        "tertiary": f"{domain} electrode",
                        "quaternary": "current collector",
                    },
                    bounds=(0, self.phase_param.c_max),
                    scale=self.phase_param.c_max,
                )
                R = pybamm.SpatialVariable(
                    f"R_{domain[0]}",
                    domain=[f"{domain} {phase_name}particle size"],
                    auxiliary_domains={
                        "secondary": f"{domain} electrode",
                        "tertiary": "current collector",
                    },
                    coord_sys="cartesian",
                )
                variables = self._get_distribution_variables(R)
                f_v_dist = variables[
                    f"{Domain} volume-weighted {phase_name}"
                    "particle-size distribution [m-1]"
                ]
            else:
                c_s_distribution = pybamm.Variable(
                    f"X-averaged {domain} {phase_name}particle "
                    "concentration distribution [mol.m-3]",
                    domain=f"{domain} {phase_name}particle",
                    auxiliary_domains={
                        "secondary": f"{domain} {phase_name}particle size",
                        "tertiary": "current collector",
                    },
                    bounds=(0, self.phase_param.c_max),
                    scale=self.phase_param.c_max,
                )
                R = pybamm.SpatialVariable(
                    f"R_{domain[0]}",
                    domain=[f"{domain} {phase_name}particle size"],
                    auxiliary_domains={"secondary": "current collector"},
                    coord_sys="cartesian",
                )
                variables = self._get_distribution_variables(R)
                f_v_dist = variables[
                    f"X-averaged {domain} volume-weighted {phase_name}"
                    "particle-size distribution [m-1]"
                ]

            # Standard concentration distribution variables (size-dependent)
            variables.update(
                self._get_standard_concentration_distribution_variables(
                    c_s_distribution
                )
            )
            # Standard size-averaged variables. Average concentrations using
            # the volume-weighted distribution since they are volume-based
            # quantities. Necessary for output variables "Total lithium in
            # negative electrode [mol]", etc, to be calculated correctly
            c_s = pybamm.Integral(f_v_dist * c_s_distribution, R)
            if self.x_average is True:
                c_s = pybamm.SecondaryBroadcast(c_s, [f"{domain} electrode"])

        variables.update(self._get_standard_concentration_variables(c_s))

        return variables

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        param = self.param

        if self.size_distribution is False:
            if self.x_average is False:
                c_s = variables[
                    f"{Domain} {phase_name}particle concentration [mol.m-3]"
                ]
                T = pybamm.PrimaryBroadcast(
                    variables[f"{Domain} electrode temperature [K]"],
                    [f"{domain} {phase_name}particle"],
                )
                R_nondim = variables[f"{Domain} {phase_name}particle radius"]
                j = variables[
                    f"{Domain} electrode {phase_name}"
                    "interfacial current density [A.m-2]"
                ]
            else:
                c_s = variables[
                    f"X-averaged {domain} {phase_name}particle concentration [mol.m-3]"
                ]
                T = pybamm.PrimaryBroadcast(
                    variables[f"X-averaged {domain} electrode temperature [K]"],
                    [f"{domain} {phase_name}particle"],
                )
                R_nondim = 1
                j = variables[
                    f"X-averaged {domain} electrode {phase_name}"
                    "interfacial current density [A.m-2]"
                ]
            R_broad_nondim = R_nondim
        else:
            R_nondim = variables[f"{Domain} {phase_name}particle sizes"]
            R_broad_nondim = pybamm.PrimaryBroadcast(
                R_nondim, [f"{domain} {phase_name}particle"]
            )
            if self.x_average is False:
                c_s = variables[
                    f"{Domain} {phase_name}particle "
                    "concentration distribution [mol.m-3]"
                ]

                # broadcast T to "particle size" domain then again into "particle"
                T = pybamm.PrimaryBroadcast(
                    variables[f"{Domain} electrode temperature [K]"],
                    [f"{domain} {phase_name}particle size"],
                )
                T = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}particle"])
                j = variables[
                    f"{Domain} electrode {phase_name}interfacial "
                    "current density distribution [A.m-2]"
                ]
            else:
                c_s = variables[
                    f"X-averaged {domain} {phase_name}particle "
                    "concentration distribution [mol.m-3]"
                ]

                # broadcast to "particle size" domain then again into "particle"
                T = pybamm.PrimaryBroadcast(
                    variables[f"X-averaged {domain} electrode temperature [K]"],
                    [f"{domain} {phase_name}particle size"],
                )
                T = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}particle"])
                j = variables[
                    f"X-averaged {domain} electrode {phase_name}interfacial "
                    "current density distribution [A.m-2]"
                ]

        D_eff = self._get_effective_diffusivity(c_s, T)
        N_s = -D_eff * pybamm.grad(c_s)

        variables.update(
            {
                f"{Domain} {phase_name}particle rhs [mol.m-3.s-1]": -(
                    1 / (R_broad_nondim**2)
                )
                * pybamm.div(N_s),
                f"{Domain} {phase_name}particle bc [mol.m-2]": -j
                * R_nondim
                / param.F
                / pybamm.surf(D_eff),
            }
        )

        if self.size_distribution is True:
            # Size-dependent flux variables
            variables.update(
                self._get_standard_diffusivity_distribution_variables(D_eff)
            )
            variables.update(self._get_standard_flux_distribution_variables(N_s))
            # Size-averaged flux variables
            R = variables[f"{Domain} {phase_name}particle sizes [m]"]
            f_a_dist = self.phase_param.f_a_dist(R)
            D_eff = pybamm.Integral(f_a_dist * D_eff, R)
            N_s = pybamm.Integral(f_a_dist * N_s, R)

        if self.x_average is True:
            D_eff = pybamm.SecondaryBroadcast(D_eff, [f"{domain} electrode"])
            N_s = pybamm.SecondaryBroadcast(N_s, [f"{domain} electrode"])

        variables.update(self._get_standard_diffusivity_variables(D_eff))
        variables.update(self._get_standard_flux_variables(N_s))

        return variables

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        if self.size_distribution is False:
            if self.x_average is False:
                c_s = variables[
                    f"{Domain} {phase_name}particle concentration [mol.m-3]"
                ]
            else:
                c_s = variables[
                    f"X-averaged {domain} {phase_name}particle concentration [mol.m-3]"
                ]
        else:
            if self.x_average is False:
                c_s = variables[
                    f"{Domain} {phase_name}particle "
                    "concentration distribution [mol.m-3]"
                ]
            else:
                c_s = variables[
                    f"X-averaged {domain} {phase_name}particle "
                    "concentration distribution [mol.m-3]"
                ]
        self.rhs = {c_s: variables[f"{Domain} {phase_name}particle rhs [mol.m-3.s-1]"]}

    def set_boundary_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        if self.size_distribution is False:
            if self.x_average is False:
                c_s = variables[
                    f"{Domain} {phase_name}particle concentration [mol.m-3]"
                ]
            else:
                c_s = variables[
                    f"X-averaged {domain} {phase_name}particle concentration [mol.m-3]"
                ]
        else:
            if self.x_average is False:
                c_s = variables[
                    f"{Domain} {phase_name}particle "
                    "concentration distribution [mol.m-3]"
                ]
            else:
                c_s = variables[
                    f"X-averaged {domain} {phase_name}particle "
                    "concentration distribution [mol.m-3]"
                ]

        rbc = variables[f"{Domain} {phase_name}particle bc [mol.m-2]"]
        self.boundary_conditions = {
            c_s: {"left": (pybamm.Scalar(0), "Neumann"), "right": (rbc, "Neumann")}
        }

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        c_init = self.phase_param.c_init
        if self.size_distribution is False:
            if self.x_average is False:
                c_s = variables[
                    f"{Domain} {phase_name}particle concentration [mol.m-3]"
                ]
            else:
                c_s = variables[
                    f"X-averaged {domain} {phase_name}particle concentration [mol.m-3]"
                ]
                c_init = pybamm.x_average(c_init)
        else:
            if self.x_average is False:
                c_s = variables[
                    f"{Domain} {phase_name}particle "
                    "concentration distribution [mol.m-3]"
                ]
                c_init = pybamm.SecondaryBroadcast(
                    c_init, f"{domain} {phase_name}particle size"
                )
            else:
                c_s = variables[
                    f"X-averaged {domain} {phase_name}particle "
                    "concentration distribution [mol.m-3]"
                ]

                c_init = pybamm.SecondaryBroadcast(
                    pybamm.x_average(c_init), f"{domain} {phase_name}particle size"
                )
        self.initial_conditions = {c_s: c_init}
