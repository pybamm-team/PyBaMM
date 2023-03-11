#
# Different OCPs for charge and discharge, based on current
#
import pybamm
from . import BaseOpenCircuitPotential


class CurrentSigmoidOpenCircuitPotential(BaseOpenCircuitPotential):
    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        current = variables["Total current density [A.m-2]"]
        k = 100

        if Domain == "Positive":
            lithiation_current = current
        elif Domain == "Negative":
            lithiation_current = -current

        m_lith = pybamm.sigmoid(0, lithiation_current, k)  # lithiation_current > 0
        m_delith = 1 - m_lith  # lithiation_current < 0

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
            ocp_surf = m_lith * U_lith + m_delith * U_delith
            dUdT = self.phase_param.dUdT(sto_surf)

            # Bulk OCP is from the average SOC and temperature
            sto_bulk = variables[f"{Domain} electrode {phase_name}stoichiometry"]
            T_bulk = pybamm.xyz_average(pybamm.size_average(T))
            U_bulk_lith = self.phase_param.U(sto_bulk, T_bulk, "lithiation")
            U_bulk_delith = self.phase_param.U(sto_bulk, T_bulk, "delithiation")
            ocp_bulk = m_lith * U_bulk_lith + m_delith * U_bulk_delith

        variables.update(self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT))
        return variables
