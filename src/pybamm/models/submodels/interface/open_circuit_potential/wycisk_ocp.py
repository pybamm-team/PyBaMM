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
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        i_surf = variables[
            f"{Domain} electrode {phase_name}interfacial current density [A.m-2]"
        ]
        # check if composite or not
        if phase_name != "":
            Q_cell = variables[f"{Domain} electrode {phase_name}phase capacity [A.h]"]
        else:
            Q_cell = variables[f"{Domain} electrode capacity [A.h]"]

        dQdU = variables[
            f"{Domain} electrode {phase_name}differential capacity [A.s.V-1]"
        ]
        dQdU = dQdU.orphans[0]
        K = self.phase_param.hysteresis_decay()
        K_x = self.phase_param.hysteresis_switch
        h = variables[f"{Domain} electrode {phase_name}hysteresis state"]

        dhdt = (
            K * (i_surf / (Q_cell * (dQdU**K_x))) * (1 - pybamm.sign(i_surf) * h)
        )  #! current is backwards for a halfcell
        self.rhs[h] = dhdt

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        h = variables[f"{Domain} electrode {phase_name}hysteresis state"]
        self.initial_conditions[h] = self.phase_param.h_init
