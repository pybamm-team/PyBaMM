#
# Class for cracking
#
import numpy as np

import pybamm

from .base_mechanics import BaseMechanics


class CrackPropagation(BaseMechanics):
    """
    Cracking behaviour in electrode particles. See :footcite:t:`Ai2019` for mechanical
    model (thickness change) and :footcite:t:`Deshpande2012` for cracking model.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    x_average : bool
        Whether to use x-averaged variables (SPM, SPMe, etc) or full variables (DFN)
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")

    """

    def __init__(self, param, domain, x_average, options, phase="primary"):
        super().__init__(param, domain, options, phase)
        self.x_average = x_average

        pybamm.citations.register("Ai2019")
        pybamm.citations.register("Deshpande2012")

    def build(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        l_cr_0 = self.phase_param.l_cr_0
        if self.x_average:
            if self.size_distribution:
                l_cr_av_dist = pybamm.Variable(
                    f"X-averaged {domain} {phase_name}particle crack length distribution [m]",
                    domains={
                        "primary": f"{domain} particle size",
                        "secondary": "current collector",
                    },
                    scale=l_cr_0,
                )
                l_cr_dist = pybamm.SecondaryBroadcast(
                    l_cr_av_dist, f"{domain} electrode"
                )
                l_cr_av = pybamm.size_average(l_cr_av_dist)
                l_cr_state = l_cr_av_dist
                l_cr_init = pybamm.PrimaryBroadcast(l_cr_0, f"{domain} particle size")
            else:
                l_cr_av = pybamm.Variable(
                    f"X-averaged {domain} {phase_name}particle crack length [m]",
                    domain="current collector",
                    scale=l_cr_0,
                )
                l_cr_state = l_cr_av
                l_cr_init = l_cr_0
            l_cr = pybamm.PrimaryBroadcast(l_cr_av, f"{domain} electrode")
        else:
            if self.size_distribution:
                l_cr_dist = pybamm.Variable(
                    f"{Domain} {phase_name}particle crack length distribution [m]",
                    domains={
                        "primary": f"{domain} particle size",
                        "secondary": f"{domain} electrode",
                        "tertiary": "current collector",
                    },
                    scale=l_cr_0,
                )
                l_cr = pybamm.size_average(l_cr_dist)
                l_cr_state = l_cr_dist
                l_cr_init = pybamm.PrimaryBroadcast(
                    pybamm.PrimaryBroadcast(l_cr_0, f"{domain} electrode"),
                    f"{domain} particle size",
                )
            else:
                l_cr = pybamm.Variable(
                    f"{Domain} {phase_name}particle crack length [m]",
                    domain=f"{domain} electrode",
                    auxiliary_domains={"secondary": "current collector"},
                    scale=l_cr_0,
                )
                l_cr_state = l_cr
                l_cr_init = pybamm.PrimaryBroadcast(l_cr_0, f"{domain} electrode")

        variables = self._get_standard_variables(l_cr)
        variables.update(self._get_standard_surface_variables(l_cr))
        variables.update(self._get_mechanical_results())
        if self.size_distribution:
            variables.update(self._get_standard_size_distribution_variables(l_cr_dist))
            variables.update(self._get_mechanical_size_distribution_results())
            stress_t_surf = variables[
                f"{Domain} {phase_name}particle surface tangential stress distribution [Pa]"
            ]
            l_cr_growing = l_cr_dist
        else:
            stress_t_surf = variables[
                f"{Domain} {phase_name}particle surface tangential stress [Pa]"
            ]
            l_cr_growing = l_cr

        T = self._electrode_temperature()
        k_cr = self.phase_param.k_cr(T)
        m_cr = self.phase_param.m_cr
        b_cr = self.phase_param.b_cr
        # compressive stress will not lead to crack propagation
        dK_SIF = (
            stress_t_surf
            * b_cr
            * pybamm.sqrt(np.pi * l_cr_growing)
            * (stress_t_surf >= 0)
        )
        dl_cr = k_cr * (dK_SIF**m_cr) / 3600  # divide by 3600 to replace t0_cr
        dl_cr_av = pybamm.x_average(dl_cr)
        variables.update(
            {
                f"{Domain} {phase_name}particle cracking rate [m.s-1]": dl_cr,
                f"X-averaged {domain} {phase_name}particle cracking rate [m.s-1]": dl_cr_av,
            }
        )
        self.variables.update(variables)

        self.rhs[l_cr_state] = dl_cr_av if self.x_average else dl_cr
        self.initial_conditions[l_cr_state] = l_cr_init
        l_cr_event = variables[
            self._build_var_name("particle crack length", "[m]", self.x_average)
        ]
        self.events.append(
            pybamm.Event(
                f"{domain} {phase_name} particle crack length larger than particle radius",
                1 - pybamm.max(l_cr_event) / self.phase_param.R_typ,
            )
        )
