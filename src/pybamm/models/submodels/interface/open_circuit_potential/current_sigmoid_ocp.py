#
# Different OCPs for charge and discharge, based on current
#
import pybamm
from . import BaseOpenCircuitPotential


class CurrentSigmoidOpenCircuitPotential(BaseOpenCircuitPotential):
    def build(self, submodels):
        domain, Domain = self.domain_Domain
        current = pybamm.CoupledVariable("Total current density [A.m-2]")
        self.coupled_variables.update({current.name: current})
        k = 100

        if Domain == "Positive":
            lithiation_current = current
        elif Domain == "Negative":
            lithiation_current = -current

        m_lith = pybamm.sigmoid(0, lithiation_current, k)  # lithiation_current > 0
        m_delith = 1 - m_lith  # lithiation_current < 0

        phase_name = self.phase_name

        if self.reaction == "lithium-ion main":
            T = pybamm.CoupledVariable(
                f"{Domain} electrode temperature [K]",
                f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
            self.coupled_variables.update({T.name: T})
            # For "particle-size distribution" models, take distribution version
            # of sto_surf that depends on particle size.
            domain_options = getattr(self.options, domain)
            if domain_options["particle size"] == "distribution":
                sto_surf = pybamm.CoupledVariable(
                    f"{Domain} {phase_name}particle surface stoichiometry distribution",
                    f"{domain} particle size",
                    auxiliary_domains={
                        "secondary": f"{domain} electrode",
                        "tertiary": "current collector",
                    },
                )
                self.coupled_variables.update({sto_surf.name: sto_surf})
                # If variable was broadcast, take only the orphan
                # if isinstance(sto_surf, pybamm.Broadcast) and isinstance(
                #    T, pybamm.Broadcast
                # ):
                #    sto_surf = sto_surf.orphans[0]
                #    T = T.orphans[0]
                T = pybamm.PrimaryBroadcast(T, [f"{domain} particle size"])
            else:
                sto_surf = pybamm.CoupledVariable(
                    f"{Domain} {phase_name}particle surface stoichiometry",
                    f"{domain} electrode",
                    auxiliary_domains={"secondary": "current collector"},
                )
                self.coupled_variables.update({sto_surf.name: sto_surf})
                # If variable was broadcast, take only the orphan
                # if isinstance(sto_surf, pybamm.Broadcast) and isinstance(
                #    T, pybamm.Broadcast
                # ):
                #    sto_surf = sto_surf.orphans[0]
                #    T = T.orphans[0]

            U_lith = self.phase_param.U(sto_surf, T, "lithiation")
            U_delith = self.phase_param.U(sto_surf, T, "delithiation")
            ocp_surf = m_lith * U_lith + m_delith * U_delith
            dUdT = self.phase_param.dUdT(sto_surf)

            # Bulk OCP is from the average SOC and temperature
            sto_bulk = pybamm.CoupledVariable(
                f"{Domain} electrode {phase_name}stoichiometry",
                "current collector",
            )
            self.coupled_variables.update({sto_bulk.name: sto_bulk})
            T_bulk = pybamm.xyz_average(pybamm.size_average(T))
            U_bulk_lith = self.phase_param.U(sto_bulk, T_bulk, "lithiation")
            U_bulk_delith = self.phase_param.U(sto_bulk, T_bulk, "delithiation")
            ocp_bulk = m_lith * U_bulk_lith + m_delith * U_bulk_delith

        variables = self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT)
        self.variables.update(variables)
