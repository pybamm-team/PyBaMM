#
# Single open-circuit potential, same for charge and discharge
#
import pybamm
from . import BaseOpenCircuitPotential


class RK_Open_Circuit_Potential(BaseOpenCircuitPotential):
    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        if self.reaction == "lithium-ion main":
            T = variables[f"{Domain} electrode temperature [K]"]
            # For "particle-size distribution" models, take distribution version
            # of c_s_surf that depends on particle size.
            domain_options = getattr(self.options, domain)
            if domain_options["particle size"] == "distribution":
                sto_surf = variables[
                    f"{Domain} {phase_name}particle surface stoichiometry distribution"
                ]
                # If variable was broadcast, take only the orphan
                if isinstance(sto_surf, pybamm.Broadcast) and isinstance(
                    T, pybamm.Broadcast
                ):
                    sto_surf = sto_surf.orphans[0]
                    T = T.orphans[0]
                T = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}particle size"])
            else:
                sto_surf = variables[
                    f"{Domain} {phase_name}particle surface stoichiometry"
                ]
                # If variable was broadcast, take only the orphan
                if isinstance(sto_surf, pybamm.Broadcast) and isinstance(
                    T, pybamm.Broadcast
                ):
                    sto_surf = sto_surf.orphans[0]
                    T = T.orphans[0]

            inputs = {
                f"{Domain} particle stoichiometry": sto_surf,
                f"{Domain} electrode temperature [k]": T,
            }

            OCP_surf = pybamm.FunctionParameter(
                f"{Domain} electrode RK_OCP [V]", inputs
            )

            dUdT = self.phase_param.dUdT(sto_surf)

            # Bulk OCP is from the average SOC and temperature
            sto_bulk = variables[f"{Domain} electrode {phase_name}stoichiometry"]
            T_bulk = pybamm.xyzs_average(T)
            ocp_bulk = self.phase_param.U(sto_bulk, T_bulk)

        variables.update(self._get_standard_ocp_variables(OCP_surf, ocp_bulk, dUdT))
        return variables
