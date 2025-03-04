#
# from Wycisk 2022
#
import pybamm
from . import BaseHysteresisOpenCircuitPotential


class WyciskOpenCircuitPotential(BaseHysteresisOpenCircuitPotential):
    """
    Class for open-circuit potential with hysteresis based on the approach outlined by Wycisk :footcite:t:'Wycisk2022'.
    This approach employs a differential capacity hysteresis state variable. The decay and switching of the hysteresis state
    is tunable via two additional parameters. The hysteresis state is updated based on the current and the differential capacity.
    """

    def get_fundamental_variables(self):
        return self._get_hysteresis_state_variables()

    def get_coupled_variables(self, variables):
        _, Domain = self.domain_Domain
        phase_name = self.phase_name

        variables = self._get_coupled_variables(variables)

        if self.reaction == "lithium-ion main":
            # determine dQ/dU
            sto_surf, T = self._get_stoichiometry_and_temperature(variables)
            T_bulk = pybamm.xyz_average(pybamm.size_average(T))
            if phase_name == "":
                Q_mag = variables[f"{Domain} electrode capacity [A.h]"]
            else:
                Q_mag = variables[
                    f"{Domain} electrode {phase_name}phase capacity [A.h]"
                ]

            dU = self.phase_param.U(sto_surf, T_bulk).diff(sto_surf)
            dQdU = Q_mag / dU
            variables[
                f"{Domain} electrode {phase_name}differential capacity [A.s.V-1]"
            ] = dQdU

        return variables

    def set_rhs(self, variables):
        _, Domain = self.domain_Domain
        phase_name = self.phase_name

        i_surf = variables[
            f"{Domain} electrode {phase_name}interfacial current density [A.m-2]"
        ]
        sto_surf = variables[f"{Domain} {phase_name}particle surface stoichiometry"]
        T = variables[f"{Domain} electrode temperature [K]"]
        h = variables[f"{Domain} electrode {phase_name}hysteresis state"]

        # check if composite or not
        if phase_name != "":
            Q_cell = variables[f"{Domain} electrode {phase_name}phase capacity [A.h]"]
        else:
            Q_cell = variables[f"{Domain} electrode capacity [A.h]"]

        dQdU = variables[
            f"{Domain} electrode {phase_name}differential capacity [A.s.V-1]"
        ]
        dQdU = dQdU.orphans[0]
        K = self.phase_param.hysteresis_decay(sto_surf, T)
        K_x = self.phase_param.hysteresis_switch

        i_surf_sign = pybamm.sign(i_surf)
        signed_h = 1 - i_surf_sign * h
        rate_coefficient = K * (i_surf / (Q_cell * (dQdU**K_x)))
        dhdt = rate_coefficient * signed_h

        self.rhs[h] = dhdt
