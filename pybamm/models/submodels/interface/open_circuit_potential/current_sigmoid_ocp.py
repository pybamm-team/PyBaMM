#
# Different OCPs for charge and discharge, based on current
#
import pybamm
from . import BaseOpenCircuitPotential


class CurrentSigmoidOpenCircuitPotential(BaseOpenCircuitPotential):
    def get_coupled_variables(self, variables):
        current = variables["Total current density [A.m-2]"]
        k = 100
        m_lith = pybamm.sigmoid(current, 0, k)  # for lithation (current < 0)
        m_delith = 1 - m_lith  # for delithiation (current > 0)

        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        if self.reaction == "lithium-ion main":
            T = variables[f"{Domain} electrode temperature [K]"]
            # For "particle-size distribution" models, take distribution version
            # of sto_surf that depends on particle size.
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
                T = pybamm.PrimaryBroadcast(T, [f"{domain} particle size"])
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

            U_lith = self.phase_param.U(sto_surf, T, "lithiation")
            U_delith = self.phase_param.U(sto_surf, T, "delithiation")
            ocp = m_lith * U_lith + m_delith * U_delith
            dUdT = self.phase_param.dUdT(sto_surf)

        variables.update(self._get_standard_ocp_variables(ocp, dUdT))
        return variables
