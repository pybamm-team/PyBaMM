#
# Single open-circuit potential, same for charge and discharge
#
import pybamm

from . import BaseOpenCircuitPotential


class SingleOpenCircuitPotential(BaseOpenCircuitPotential):
    def build(self):
        if self.reaction == "lithium-ion main":
            sto_surf, sto_bulk, T, T_bulk = self._get_stoichiometry_and_temperature()

            ocp_surf = self.phase_param.U(sto_surf, T)
            ocp_bulk = self.phase_param.U(sto_bulk, T_bulk)

            dUdT = self.phase_param.dUdT(sto_surf)

        elif self.reaction == "lithium metal plating":
            T = self._coupled_electrode_variable("electrode temperature [K]")
            ocp_surf = 0 * T
            ocp_bulk = pybamm.Scalar(0)
            dUdT = 0 * T
        elif self.reaction == "lead-acid main":
            c_e = self._coupled_electrode_variable(
                "electrolyte concentration [mol.m-3]", self.x_average
            )
            ocp_surf = self.phase_param.U(c_e, self.param.T_init)
            dUdT = pybamm.Scalar(0)

            c_e_av = self.coupled_variable(
                "X-averaged electrolyte concentration [mol.m-3]",
                domain="current collector",
            )
            ocp_bulk = self.phase_param.U(c_e_av, self.param.T_init)

        elif self.reaction == "lead-acid oxygen":
            ocp_surf = self.param.U_Ox
            ocp_bulk = self.param.U_Ox
            dUdT = pybamm.Scalar(0)

        variables = self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT)
        if self.reaction == "lithium-ion main":
            self._alias_ocp_as_equilibrium(variables)
        self.variables.update(variables)
