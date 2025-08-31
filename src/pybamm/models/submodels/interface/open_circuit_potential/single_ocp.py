#
# Single open-circuit potential, same for charge and discharge
#
import pybamm

from . import BaseOpenCircuitPotential


class SingleOpenCircuitPotential(BaseOpenCircuitPotential):
    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        if self.reaction == "lithium-ion main":
            sto_surf, sto_bulk, T, T_bulk = self._get_stoichiometry_and_temperature(
                variables
            )

            ocp_surf = self.phase_param.U(sto_surf, T)
            ocp_bulk = self.phase_param.U(sto_bulk, T_bulk)

            dUdT = self.phase_param.dUdT(sto_surf)

            variables.update(
                {
                    f"{Domain} electrode {phase_name}equilibrium open-circuit potential [V]": ocp_surf,
                    f"X-averaged {domain} electrode {phase_name}equilibrium open-circuit potential [V]": pybamm.x_average(
                        ocp_surf
                    ),
                }
            )

        elif self.reaction == "lithium metal plating":
            T = variables[f"{Domain} electrode temperature [K]"]
            ocp_surf = 0 * T
            ocp_bulk = pybamm.Scalar(0)
            dUdT = 0 * T
        elif self.reaction == "lead-acid main":
            c_e = variables[f"{Domain} electrolyte concentration [mol.m-3]"]
            # If c_e was broadcast, take only the orphan
            if isinstance(c_e, pybamm.Broadcast):
                c_e = c_e.orphans[0]
            ocp_surf = self.phase_param.U(c_e, self.param.T_init)
            dUdT = pybamm.Scalar(0)

            c_e_av = variables["X-averaged electrolyte concentration [mol.m-3]"]
            ocp_bulk = self.phase_param.U(c_e_av, self.param.T_init)

        elif self.reaction == "lead-acid oxygen":
            ocp_surf = self.param.U_Ox
            ocp_bulk = self.param.U_Ox
            dUdT = pybamm.Scalar(0)

        variables.update(self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT))
        return variables
