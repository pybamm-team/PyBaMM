#
# Class to calculate total particle concentrations
#
import pybamm
from .base_particle import BaseParticle


class TotalConcentration(BaseParticle):
    """
    Class to calculate total particle concentrations

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

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase = self.phase
        phase_name = self.phase_name

        c_s_rav = variables[
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]"
        ]
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
                f"{Domain} electrode {phase_name}stoichiometry": c_s_vol_av / c_scale,
                f"{Domain} electrode {phase_name}volume-averaged "
                "concentration": c_s_vol_av / c_scale,
                f"{Domain} electrode {phase_name}volume-averaged "
                "concentration [mol.m-3]": c_s_vol_av,
                f"Total lithium in {phase} phase in {domain} electrode [mol]"
                "": pybamm.yz_average(c_s_vol_av * eps_s_av) * L * A,
            }
        )
        return variables
