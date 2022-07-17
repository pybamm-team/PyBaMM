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
    x_average : bool
        Whether the particle concentration is averaged over the x-direction

    **Extends:** :class:`pybamm.particle.BaseParticle`
    """

    def __init__(self, param, domain, options, x_average):
        super().__init__(param, domain, options)
        self.x_average = x_average

    def get_fundamental_variables(self):
        Domain = self.domain
        domain = Domain.lower()
        phase_name = self.phase_name

        variables = {}
        if self.size_distribution is False:
            if self.x_average is False:
                c_s = pybamm.Variable(
                    f"{Domain} {phase_name}particle concentration",
                    f"{domain} {phase_name}particle",
                    auxiliary_domains={
                        "secondary": f"{domain} electrode",
                        "tertiary": "current collector",
                    },
                    bounds=(0, 1),
                )
                c_s.print_name = f"c_s_{domain[0]}"
            else:
                c_s_xav = pybamm.Variable(
                    f"X-averaged {domain} {phase_name}particle concentration",
                    f"{domain} {phase_name}particle",
                    auxiliary_domains={"secondary": "current collector"},
                    bounds=(0, 1),
                )
                c_s_xav.print_name = f"c_s_{domain[0]}_xav"
                c_s = pybamm.SecondaryBroadcast(c_s_xav, f"{domain} electrode")
        else:
            if self.x_average is False:
                c_s_distribution = pybamm.Variable(
                    f"{Domain} {phase_name}particle concentration distribution",
                    domain=f"{domain} {phase_name}particle",
                    auxiliary_domains={
                        "secondary": f"{domain} {phase_name}particle size",
                        "tertiary": f"{domain} electrode",
                        "quaternary": "current collector",
                    },
                    bounds=(0, 1),
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
                    f"{Domain} volume-weighted particle-size distribution"
                ]
            else:
                c_s_distribution = pybamm.Variable(
                    f"X-averaged {domain} {phase_name}particle "
                    "concentration distribution",
                    domain=f"{domain} {phase_name}particle",
                    auxiliary_domains={
                        "secondary": f"{domain} {phase_name}particle size",
                        "tertiary": "current collector",
                    },
                    bounds=(0, 1),
                )
                R = pybamm.SpatialVariable(
                    f"R_{domain[0]}",
                    domain=[f"{domain} {phase_name}particle size"],
                    auxiliary_domains={"secondary": "current collector"},
                    coord_sys="cartesian",
                )
                variables = self._get_distribution_variables(R)
                f_v_dist = variables[
                    f"X-averaged {domain} volume-weighted particle-size distribution"
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
        Domain = self.domain
        domain = Domain.lower()
        phase_name = self.phase_name

        if self.size_distribution is False:
            if self.x_average is False:
                c_s = variables[f"{Domain} {phase_name}particle concentration"]
                T = pybamm.PrimaryBroadcast(
                    variables[f"{Domain} electrode temperature"],
                    [f"{domain} {phase_name}particle"],
                )
            else:
                c_s = variables[
                    f"X-averaged {domain} {phase_name}particle concentration"
                ]
                T = pybamm.PrimaryBroadcast(
                    variables[f"X-averaged {domain} electrode temperature"],
                    [f"{domain} {phase_name}particle"],
                )
        else:
            if self.x_average is False:
                c_s = variables[
                    f"{Domain} {phase_name}particle concentration distribution"
                ]

                # broadcast T to "particle size" domain then again into "particle"
                T = pybamm.PrimaryBroadcast(
                    variables[f"{Domain} electrode temperature"],
                    [f"{domain} {phase_name}particle size"],
                )
                T = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}particle"])
            else:
                c_s = variables[
                    f"X-averaged {domain} {phase_name}particle "
                    "concentration distribution"
                ]

                # broadcast to "particle size" domain then again into "particle"
                T = pybamm.PrimaryBroadcast(
                    variables[f"X-averaged {domain} electrode temperature"],
                    [f"{domain} {phase_name}particle size"],
                )
                T = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}particle"])

        D_eff = self._get_effective_diffusivity(c_s, T)
        N_s = -D_eff * pybamm.grad(c_s)

        if self.size_distribution is True:
            # Size-dependent flux variables
            variables.update(self._get_standard_flux_distribution_variables(N_s))
            R = variables[f"{Domain} {phase_name}particle sizes"]
            f_a_dist = self.phase_param.f_a_dist(R)
            # Size-averaged flux variables (perform area-weighted avg manually as flux
            # evals on edges)
            N_s = pybamm.Integral(f_a_dist * N_s, R)

            # Volume-weighted average for effective diffusivity
            variables.update(
                self._get_standard_diffusivity_distribution_variables(D_eff)
            )

        if self.x_average is True:
            D_eff = pybamm.SecondaryBroadcast(D_eff, [f"{domain} electrode"])
            N_s = pybamm.SecondaryBroadcast(N_s, [f"{domain} electrode"])

        if self.size_distribution is False:
            # Save diffusivity variables for the no-size-distrbution case
            # (they were saved earlier for the size-distribution case)
            variables.update(self._get_standard_diffusivity_variables(D_eff))

        variables.update(self._get_standard_flux_variables(N_s))
        variables.update(self._get_total_concentration_variables(variables))

        return variables

    def set_rhs(self, variables):
        Domain = self.domain
        domain = Domain.lower()
        phase_name = self.phase_name

        if self.size_distribution is False:
            if self.x_average is False:
                c_s = variables[f"{Domain} {phase_name}particle concentration"]
                N_s = variables[f"{Domain} {phase_name}particle flux"]
                R = variables[f"{Domain} {phase_name}particle radius"]
            else:
                c_s = variables[
                    f"X-averaged {domain} {phase_name}particle concentration"
                ]
                N_s = variables[f"X-averaged {domain} {phase_name}particle flux"]
                R = 1
        else:
            if self.x_average is False:
                c_s = variables[
                    f"{Domain} {phase_name}particle concentration distribution"
                ]
                N_s = variables[f"{Domain} {phase_name}particle flux distribution"]
            else:
                c_s = variables[
                    f"X-averaged {domain} {phase_name}particle "
                    "concentration distribution"
                ]

                N_s = variables[
                    f"X-averaged {domain} {phase_name}particle flux distribution"
                ]
            # Spatial variable R, broadcast into particle
            R_spatial_variable = variables[f"{Domain} {phase_name}particle sizes"]
            R = pybamm.PrimaryBroadcast(
                R_spatial_variable, [f"{domain} {phase_name}particle"]
            )

        self.rhs = {c_s: -(1 / (R ** 2 * self.phase_param.C_diff)) * pybamm.div(N_s)}

    def set_boundary_conditions(self, variables):
        Domain = self.domain
        domain = Domain.lower()
        phase_name = self.phase_name
        phase_param = self.phase_param

        if self.size_distribution is False:
            if self.x_average is False:
                c_s = variables[f"{Domain} {phase_name}particle concentration"]
                D_eff = variables[
                    f"{Domain} effective {phase_name} particle diffusivity"
                ]
                j = variables[f"{Domain} electrode interfacial current density"]
                R = variables[f"{Domain} {phase_name}particle radius"]
            else:
                c_s = variables[
                    f"X-averaged {domain} {phase_name}particle concentration"
                ]
                D_eff = variables[
                    f"X-averaged {domain} effective {phase_name} particle diffusivity"
                ]
                j = variables[
                    f"X-averaged {domain} electrode interfacial current density"
                ]
                R = 1
        else:
            if self.x_average is False:
                c_s = variables[
                    f"{Domain} {phase_name}particle concentration distribution"
                ]
                D_eff = variables[
                    f"{Domain} effective {phase_name} particle diffusivity distribution"
                ]
                j = variables[
                    f"{Domain} electrode interfacial current density distribution"
                ]
            else:
                c_s = variables[
                    f"X-averaged {domain} {phase_name}particle "
                    "concentration distribution"
                ]
                D_eff = variables[
                    f"X-averaged {domain} effective {phase_name} particle "
                    "diffusivity distribution"
                ]
                j = variables[
                    f"X-averaged {domain} electrode interfacial "
                    "current density distribution"
                ]
            R = variables[f"{Domain} {phase_name}particle sizes"]

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
        Domain = self.domain
        domain = Domain.lower()
        phase_name = self.phase_name

        c_init = self.phase_param.c_init
        if self.size_distribution is False:
            if self.x_average is False:
                c_s = variables[f"{Domain} {phase_name}particle concentration"]
            else:
                c_s = variables[
                    f"X-averaged {domain} {phase_name}particle concentration"
                ]
                c_init = pybamm.x_average(c_init)
        else:
            if self.x_average is False:
                c_s = variables[
                    f"{Domain} {phase_name}particle concentration distribution"
                ]
                c_init = pybamm.SecondaryBroadcast(
                    c_init, f"{domain} {phase_name}particle size"
                )
            else:
                c_s = variables[
                    f"X-averaged {domain} {phase_name}particle "
                    "concentration distribution"
                ]

                c_init = pybamm.SecondaryBroadcast(
                    pybamm.x_average(c_init), f"{domain} {phase_name}particle size"
                )
        self.initial_conditions = {c_s: c_init}
