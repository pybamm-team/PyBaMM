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

        variables = {}
        if self.size_distribution is False:
            if self.x_average is False:
                c_s = pybamm.Variable(
                    f"{Domain} particle concentration",
                    f"{domain} particle",
                    auxiliary_domains={
                        "secondary": f"{domain} electrode",
                        "tertiary": "current collector",
                    },
                    bounds=(0, 1),
                )
                c_s.print_name = f"c_s_{domain[0]}"
            else:
                c_s_xav = pybamm.Variable(
                    f"X-averaged {domain} particle concentration",
                    f"{domain} particle",
                    auxiliary_domains={"secondary": "current collector"},
                    bounds=(0, 1),
                )
                c_s_xav.print_name = f"c_s_{domain[0]}_xav"
                c_s = pybamm.SecondaryBroadcast(c_s_xav, f"{domain} electrode")
        else:
            if self.x_average is False:
                c_s_distribution = pybamm.Variable(
                    f"{Domain} particle concentration distribution",
                    domain=f"{domain} particle",
                    auxiliary_domains={
                        "secondary": f"{domain} particle size",
                        "tertiary": f"{domain} electrode",
                        "quaternary": "current collector",
                    },
                    bounds=(0, 1),
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
                f_v_dist = variables[
                    f"{Domain} volume-weighted particle-size distribution"
                ]
            else:
                c_s_distribution = pybamm.Variable(
                    f"X-averaged {domain} particle concentration distribution",
                    domain=f"{domain} particle",
                    auxiliary_domains={
                        "secondary": f"{domain} particle size",
                        "tertiary": "current collector",
                    },
                    bounds=(0, 1),
                )
                R = pybamm.SpatialVariable(
                    f"R_{domain[0]}",
                    domain=[f"{domain} particle size"],
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
        domain_param = self.domain_param

        if self.size_distribution is False:
            if self.x_average is False:
                c_s = variables[f"{Domain} particle concentration"]
                T = pybamm.PrimaryBroadcast(
                    variables[f"{Domain} electrode temperature"], [f"{domain} particle"]
                )
                R = variables[f"{Domain} particle radius"]
                j = variables[f"{Domain} electrode interfacial current density"]
            else:
                c_s = variables[f"X-averaged {domain} particle concentration"]
                T = pybamm.PrimaryBroadcast(
                    variables[f"X-averaged {domain} electrode temperature"],
                    [f"{domain} particle"],
                )
                R = 1
                j = variables[
                    f"X-averaged {domain} electrode interfacial current density"
                ]
            R_broad = R
        else:
            R = variables[f"{Domain} particle sizes"]
            R_broad = pybamm.PrimaryBroadcast(R, [f"{domain} particle"])
            if self.x_average is False:
                c_s = variables[f"{Domain} particle concentration distribution"]

                # broadcast T to "particle size" domain then again into "particle"
                T = pybamm.PrimaryBroadcast(
                    variables[f"{Domain} electrode temperature"],
                    [f"{domain} particle size"],
                )
                T = pybamm.PrimaryBroadcast(T, [f"{domain} particle"])
                j = variables[
                    f"{Domain} electrode interfacial current density distribution"
                ]
            else:
                c_s = variables[
                    f"X-averaged {domain} particle concentration distribution"
                ]

                # broadcast to "particle size" domain then again into "particle"
                T = pybamm.PrimaryBroadcast(
                    variables[f"X-averaged {domain} electrode temperature"],
                    [f"{domain} particle size"],
                )
                T = pybamm.PrimaryBroadcast(T, [f"{domain} particle"])
                j = variables[
                    f"X-averaged {domain} electrode interfacial "
                    "current density distribution"
                ]

        D_eff = self._get_effective_diffusivity(c_s, T)
        N_s = -D_eff * pybamm.grad(c_s)

        variables.update(
            {
                f"{Domain} particle rhs": -(1 / (R_broad ** 2 * domain_param.C_diff))
                * pybamm.div(N_s),
                f"{Domain} particle bc": -domain_param.C_diff
                * j
                * R
                / domain_param.a_R
                / domain_param.gamma
                / pybamm.surf(D_eff),
            }
        )

        if self.size_distribution is True:
            # Size-dependent flux variables
            variables.update(self._get_standard_flux_distribution_variables(N_s))
            f_a_dist = self.domain_param.f_a_dist(R)
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

        if self.size_distribution is False:
            if self.x_average is False:
                c_s = variables[f"{Domain} particle concentration"]
            else:
                c_s = variables[f"X-averaged {domain} particle concentration"]
        else:
            if self.x_average is False:
                c_s = variables[f"{Domain} particle concentration distribution"]
            else:
                c_s = variables[
                    f"X-averaged {domain} particle concentration distribution"
                ]
        self.rhs = {c_s: variables[f"{Domain} particle rhs"]}

    def set_boundary_conditions(self, variables):
        Domain = self.domain
        domain = Domain.lower()

        if self.size_distribution is False:
            if self.x_average is False:
                c_s = variables[f"{Domain} particle concentration"]
            else:
                c_s = variables[f"X-averaged {domain} particle concentration"]
        else:
            if self.x_average is False:
                c_s = variables[f"{Domain} particle concentration distribution"]
            else:
                c_s = variables[
                    f"X-averaged {domain} particle concentration distribution"
                ]

        rbc = variables[f"{Domain} particle bc"]
        self.boundary_conditions = {
            c_s: {"left": (pybamm.Scalar(0), "Neumann"), "right": (rbc, "Neumann")}
        }

    def set_initial_conditions(self, variables):
        Domain = self.domain
        domain = Domain.lower()

        c_init = self.domain_param.c_init
        if self.size_distribution is False:
            if self.x_average is False:
                c_s = variables[f"{Domain} particle concentration"]
            else:
                c_s = variables[f"X-averaged {domain} particle concentration"]
                c_init = pybamm.x_average(c_init)
        else:
            if self.x_average is False:
                c_s = variables[f"{Domain} particle concentration distribution"]
                c_init = pybamm.SecondaryBroadcast(c_init, f"{domain} particle size")
            else:
                c_s = variables[
                    f"X-averaged {domain} particle concentration distribution"
                ]

                c_init = pybamm.SecondaryBroadcast(
                    pybamm.x_average(c_init), f"{domain} particle size"
                )
        self.initial_conditions = {c_s: c_init}
