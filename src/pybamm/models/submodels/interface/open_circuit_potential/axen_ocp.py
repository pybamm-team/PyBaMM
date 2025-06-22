#
# from Axen 2022
#
import pybamm

from . import BaseHysteresisOpenCircuitPotential


class AxenOpenCircuitPotential(BaseHysteresisOpenCircuitPotential):
    """
    Class for single-state hysteresis model based on the implementation in :footcite:t:`Axen2022`. The hysteresis state variable $h$ is governed by an ODE which depends on the local surface stoichiometry and volumetric interfacial current density.
    """

    def __init__(self, param, domain, reaction, options, phase="primary"):
        super().__init__(param, domain, reaction, options=options, phase=phase)
        pybamm.citations.register("Axen2022")

    def get_fundamental_variables(self):
        return self._get_hysteresis_state_variables()

    def get_coupled_variables(self, variables):
        return self._get_coupled_variables(variables)

    def set_rhs(self, variables):
        _, Domain = self.domain_Domain
        phase_name = self.phase_name

        i_vol = variables[
            f"{Domain} electrode {phase_name}volumetric interfacial current density [A.m-3]"
        ]
        sto_surf = variables[f"{Domain} {phase_name}particle surface stoichiometry"]
        T = variables[f"{Domain} electrode temperature [K]"]
        h = variables[f"{Domain} electrode {phase_name}hysteresis state"]

        c_max = self.phase_param.c_max
        epsl = self.phase_param.epsilon_s
        gamma_lith = self.phase_param.hysteresis_decay(sto_surf, T, "lithiation")
        gamma_delith = self.phase_param.hysteresis_decay(sto_surf, T, "delithiation")

        i_vol_sign = pybamm.sign(i_vol)
        signed_h = 1 - i_vol_sign * h
        gamma = (
            0.5 * (1 - i_vol_sign) * gamma_lith + 0.5 * (1 + i_vol_sign) * gamma_delith
        )

        dhdt = gamma * i_vol / (self.param.F * c_max * epsl) * signed_h

        self.rhs[h] = dhdt
