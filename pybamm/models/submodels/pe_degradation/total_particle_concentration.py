#
# Class to calculate total lithium concentrations in the core
#
import pybamm
from .base_phase_transition import BasePhaseTransition


class TotalConcentration(BasePhaseTransition):
    """
    Class to calculate total lithium concentrations in the core

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str, optional
        The domain of the model (default is "Positive")
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")
    """

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase = self.phase
        phase_name = self.phase_name
        phase_param = self.phase_param

        c_c_rav = variables[
            f"R-averaged {domain} {phase_name}core "
            "lithium concentration [mol.m-3]"
        ]
        eps_s = variables[
            f"{Domain} electrode {phase_name}active material volume fraction"
        ]
        eps_s_av = pybamm.x_average(eps_s)

        lam_pe_av = variables[
            "X-averaged loss of active material due to PE phase transition"
        ]

        # total lithium in the core, concentration in shell not taken into account
        c_c_vol_av = (
            pybamm.x_average(eps_s * c_c_rav) / eps_s_av
        )
        # total cyclable lithium in the core
        c_c_vol_av_cyc = (
            pybamm.x_average(eps_s * (c_c_rav - phase_param.c_bott)) / eps_s_av
        )
        c_scale = self.phase_param.c_max
        # Positive electrode thickness [m]
        L = self.domain_param.L
        # Area of current collector
        A = self.param.A_cc

        variables.update(
            {
                f"{Domain} electrode {phase_name}stoichiometry"
                "": c_c_vol_av / c_scale,
                f"{Domain} electrode {phase_name}volume-averaged "
                "concentration": c_c_vol_av / c_scale,
                 f"{Domain} electrode {phase_name}volume-averaged "
                "concentration [mol.m-3]": c_c_vol_av,
                f"Total lithium in {phase} phase in {domain} electrode [mol]"
                "": pybamm.yz_average(c_c_vol_av * eps_s_av) * L * A * (
                    1 - lam_pe_av # remove degraded shell fraction
                    ),
                f"Total cyclable lithium in {phase} phase in "
                f"{domain} electrode [mol]": pybamm.yz_average(
                    c_c_vol_av_cyc * eps_s_av * (1 - lam_pe_av)
                ) * L * A,
            }
        )
        return variables
