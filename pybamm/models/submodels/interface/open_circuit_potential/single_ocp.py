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
            T = variables[f"{Domain} electrode temperature"]
            # For "particle-size distribution" models, take distribution version
            # of c_s_surf that depends on particle size.
            if self.options["particle size"] == "distribution":
                c_s_surf = variables[
                    f"{Domain} {phase_name}particle surface concentration distribution"
                ]
                # If variable was broadcast, take only the orphan
                if isinstance(c_s_surf, pybamm.Broadcast) and isinstance(
                    T, pybamm.Broadcast
                ):
                    c_s_surf = c_s_surf.orphans[0]
                    T = T.orphans[0]
                T = pybamm.PrimaryBroadcast(T, [f"{domain} particle size"])
            else:
                c_s_surf = variables[
                    f"{Domain} {phase_name}particle surface concentration"
                ]
                # If variable was broadcast, take only the orphan
                if isinstance(c_s_surf, pybamm.Broadcast) and isinstance(
                    T, pybamm.Broadcast
                ):
                    c_s_surf = c_s_surf.orphans[0]
                    T = T.orphans[0]

            ocp = self.phase_param.U(c_s_surf, T)
            dUdT = self.phase_param.dUdT(c_s_surf)
        elif self.reaction == "lithium metal plating":
            T = variables[f"{Domain} electrode temperature"]
            ocp = self.param.n.U_ref
            dUdT = 0 * T
        elif self.reaction == "lead-acid main":
            c_e = variables[f"{Domain} electrolyte concentration"]
            # If c_e was broadcast, take only the orphan
            if isinstance(c_e, pybamm.Broadcast):
                c_e = c_e.orphans[0]
            ocp = self.phase_param.U(c_e, self.param.T_init)
            dUdT = pybamm.Scalar(0)

        elif self.reaction == "lead-acid oxygen":
            ocp = self.phase_param.U_Ox
            dUdT = pybamm.Scalar(0)

        variables.update(self._get_standard_ocp_variables(ocp, dUdT))
        return variables
