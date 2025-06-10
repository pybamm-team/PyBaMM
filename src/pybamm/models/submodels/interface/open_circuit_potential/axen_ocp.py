#
# from Axen 2022
#
import pybamm

from . import BaseHysteresisOpenCircuitPotential


class AxenOpenCircuitPotential(BaseHysteresisOpenCircuitPotential):
    """
    Class for open-circuit potential (OCP) with hysteresis based on the approach outlined by Axen et al. https://doi.org/10.1016/j.est.2022.103985.
    This approach tracks the evolution of the hysteresis using an ODE system scaled by the volumetric interfacial current density and
    a decay rate parameter.

    The original method defines the open-circuit potential as OCP = OCP_delithiation + (OCP_lithiation - OCP_delithiation) * h,
    where h is the hysteresis state. By allowing h to vary between 0 and 1, the OCP approaches OCP_lithiation when h tends to 1
    and OCP_delithiation when h tends to 0. These interval limits are achieved by defining the ODE system as dh/dt = S_lith * (1 - h)
    for lithiation and dh/dt = S_delith * h for delithiation, where S_lith and S_delith are rate coefficients.

    The original implementation is modified to ensure h varies between -1 and 1, thus assuming an unified framework within pybamm.
    This new variation interval is obtained with the ODE system dh/dt = S_lith * (1 + h) for lithiation and dh/dt = S_delith * (1 - h)
    for delithiation. Now, OCP = OCP_delithiation + (OCP_lithiation - OCP_delithiation) * (1 - h) / 2, implying that the OCP approaches
    OCP_lithiation when h tends to -1 and OCP_delithiation when h tends to 1.
    """

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
        K_lith = self.phase_param.hysteresis_decay(sto_surf, T, "lithiation")
        K_delith = self.phase_param.hysteresis_decay(sto_surf, T, "delithiation")

        S_lith = i_vol * K_lith / (self.param.F * c_max * epsl)
        S_delith = i_vol * K_delith / (self.param.F * c_max * epsl)

        i_vol_sign = pybamm.sign(i_vol)
        signed_h = 1 - i_vol_sign * h
        rate_coefficient = (
            0.5 * (1 - i_vol_sign) * S_lith + 0.5 * (1 + i_vol_sign) * S_delith
        )

        dhdt = rate_coefficient * signed_h

        self.rhs[h] = dhdt
