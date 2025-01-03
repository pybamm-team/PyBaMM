#
# Single open-circuit potential, same for charge and discharge
#
import pybamm
from . import BaseOpenCircuitPotential


class SingleOpenCircuitPotential(BaseOpenCircuitPotential):
    def build(self, submodels):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        if self.reaction == "lithium-ion main":
            T = pybamm.CoupledVariable(
                f"{Domain} electrode temperature [K]",
                f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
            # For "particle-size distribution" models, take distribution version
            # of c_s_surf that depends on particle size.
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
                if isinstance(sto_surf, pybamm.Broadcast) and isinstance(
                    T, pybamm.Broadcast
                ):
                    sto_surf = sto_surf.orphans[0]
                    T = T.orphans[0]
                T = pybamm.PrimaryBroadcast(T, [f"{domain} particle size"])
            else:
                sto_surf = pybamm.CoupledVariable(
                    f"{Domain} {phase_name}particle surface stoichiometry",
                    f"{domain} electrode",
                    auxiliary_domains={"secondary": "current collector"},
                )
                self.coupled_variables.update({sto_surf.name: sto_surf})
                # If variable was broadcast, take only the orphan
                if isinstance(sto_surf, pybamm.Broadcast) and isinstance(
                    T, pybamm.Broadcast
                ):
                    sto_surf = sto_surf.orphans[0]
                    T = T.orphans[0]

            ocp_surf = self.phase_param.U(sto_surf, T)
            dUdT = self.phase_param.dUdT(sto_surf)

            # Bulk OCP is from the average SOC and temperature
            sto_bulk = pybamm.CoupledVariable(
                f"{Domain} electrode {phase_name}stoichiometry",
                "current collector",
            )
            self.coupled_variables.update({sto_bulk.name: sto_bulk})
            T_bulk = pybamm.xyz_average(pybamm.size_average(T))
            ocp_bulk = self.phase_param.U(sto_bulk, T_bulk)
        elif self.reaction == "lithium metal plating":
            T = pybamm.CoupledVariable(
                f"{Domain} electrode temperature [K]",
                "current collector",
            )
            ocp_surf = 0 * T
            ocp_bulk = pybamm.Scalar(0)
            dUdT = 0 * T
        elif self.reaction == "lead-acid main":
            c_e = pybamm.CoupledVariable(
                f"{Domain} electrolyte concentration [mol.m-3]",
                f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
            self.coupled_variables.update({c_e.name: c_e})
            # If c_e was broadcast, take only the orphan
            if isinstance(c_e, pybamm.Broadcast):
                c_e = c_e.orphans[0]
            ocp_surf = self.phase_param.U(c_e, self.param.T_init)
            dUdT = pybamm.Scalar(0)

            # Bulk OCP is from the average concentration and temperature
            c_e_av = pybamm.CoupledVariable(
                "X-averaged electrolyte concentration [mol.m-3]",
                "current collector",
            )
            self.coupled_variables.update({c_e_av.name: c_e_av})
            ocp_bulk = self.phase_param.U(c_e_av, self.param.T_init)

        elif self.reaction == "lead-acid oxygen":
            ocp_surf = self.param.U_Ox
            ocp_bulk = self.param.U_Ox
            dUdT = pybamm.Scalar(0)
        variables = self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT)
        self.variables.update(variables)
