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
            T = variables[f"{Domain} electrode temperature [K]"]
            # For "particle-size distribution" models, take distribution version
            # of c_s_surf that depends on particle size.
            domain_options = getattr(self.options, domain)
            if domain_options["particle size"] == "distribution":
                sto = variables[
                    f"{Domain} {phase_name}particle surface stoichiometry distribution"
                ]
                # If variable was broadcast, take only the orphan
                if isinstance(sto, pybamm.Broadcast) and isinstance(
                    T, pybamm.Broadcast
                ):
                    sto = sto.orphans[0]
                    T = T.orphans[0]
                T = pybamm.PrimaryBroadcast(T, [f"{domain} particle size"])
            else:
                sto = variables[f"{Domain} {phase_name}particle surface stoichiometry"]
                # If variable was broadcast, take only the orphan
                if isinstance(sto, pybamm.Broadcast) and isinstance(
                    T, pybamm.Broadcast
                ):
                    sto = sto.orphans[0]
                    T = T.orphans[0]

            ocp = self.phase_param.U(sto, T)
            dUdT = self.phase_param.dUdT(sto)
        elif self.reaction == "lithium metal plating":
            T = variables[f"{Domain} electrode temperature [K]"]
            ocp = 0 * T
            dUdT = 0 * T
        elif self.reaction == "lead-acid main":
            c_e = variables[f"{Domain} electrolyte concentration [mol.m-3]"]
            # If c_e was broadcast, take only the orphan
            if isinstance(c_e, pybamm.Broadcast):
                c_e = c_e.orphans[0]
            ocp = self.phase_param.U(c_e, self.param.T_init)
            dUdT = pybamm.Scalar(0)

        elif self.reaction == "lead-acid oxygen":
            ocp = self.param.U_Ox
            dUdT = pybamm.Scalar(0)

        variables.update(self._get_standard_ocp_variables(ocp, dUdT))
        return variables
