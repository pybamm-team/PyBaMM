#
# Different OCPs for charge and discharge, based on current
#
import pybamm
from . import BaseOpenCircuitPotential


class CurrentSigmoidOpenCircuitPotential(BaseOpenCircuitPotential):
    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        current = variables["Total current density"]
        k = 100

        if Domain == "Positive":
            lithiation_current = current
        elif Domain == "Negative":
            lithiation_current = -current

        m_lith = pybamm.sigmoid(0, lithiation_current, k)  # lithiation_current > 0
        m_delith = 1 - m_lith  # lithiation_current < 0

        phase_name = self.phase_name

        if self.reaction == "lithium-ion main":
            T = variables[f"{Domain} electrode temperature"]
            # For "particle-size distribution" models, take distribution version
            # of c_s_surf that depends on particle size.
            domain_options = getattr(self.options, domain)
            if domain_options["particle size"] == "distribution":
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

            U_lith = self.phase_param.U(c_s_surf, T, "lithiation")
            U_delith = self.phase_param.U(c_s_surf, T, "delithiation")
            ocp = m_lith * U_lith + m_delith * U_delith
            dUdT = self.phase_param.dUdT(c_s_surf)

        variables.update(self._get_standard_ocp_variables(ocp, dUdT))
        return variables
