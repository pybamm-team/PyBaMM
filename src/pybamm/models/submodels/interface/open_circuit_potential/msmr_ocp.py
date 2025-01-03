#
# Open-circuit potential from the Multi-Species Multi-Reaction framework
#
import pybamm
from . import BaseOpenCircuitPotential


class MSMROpenCircuitPotential(BaseOpenCircuitPotential):
    """
    Class for open-circuit potential within the Multi-Species Multi-Reaction
    framework :footcite:t:`Baker2018`. The thermodynamic model is presented in
    :footcite:t:`Verbrugge2017`, along with parameter values for a number of
    substitutional materials.
    """

    def build(self, submodels):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        if self.reaction == "lithium-ion main":
            T = pybamm.CoupledVariable(
                f"{Domain} electrode temperature [K]",
                f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
            self.coupled_variables.update({T.name: T})
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
                ocp_surf = pybamm.CoupledVariable(
                    f"{Domain} {phase_name}particle surface potential distribution [V]",
                    f"{domain} particle size",
                    auxiliary_domains={
                        "secondary": f"{domain} electrode",
                        "tertiary": "current collector",
                    },
                )
                self.coupled_variables.update({ocp_surf.name: ocp_surf})
                # If variable was broadcast, take only the orphan
                if (
                    isinstance(sto_surf, pybamm.Broadcast)
                    and isinstance(ocp_surf, pybamm.Broadcast)
                    and isinstance(T, pybamm.Broadcast)
                ):
                    sto_surf = sto_surf.orphans[0]
                    ocp_surf = ocp_surf.orphans[0]
                    T = T.orphans[0]
                T = pybamm.PrimaryBroadcast(T, [f"{domain} particle size"])
            else:
                sto_surf = pybamm.CoupledVariable(
                    f"{Domain} {phase_name}particle surface stoichiometry",
                    f"{domain} electrode",
                    auxiliary_domains={"secondary": "current collector"},
                )
                self.coupled_variables.update({sto_surf.name: sto_surf})
                ocp_surf = pybamm.CoupledVariable(
                    f"{Domain} {phase_name}particle surface potential [V]",
                    f"{domain} electrode",
                    auxiliary_domains={"secondary": "current collector"},
                )
                self.coupled_variables.update({ocp_surf.name: ocp_surf})
                # If variable was broadcast, take only the orphan
                if (
                    isinstance(sto_surf, pybamm.Broadcast)
                    and isinstance(ocp_surf, pybamm.Broadcast)
                    and isinstance(T, pybamm.Broadcast)
                ):
                    sto_surf = sto_surf.orphans[0]
                    ocp_surf = ocp_surf.orphans[0]
                    T = T.orphans[0]

            ocp_bulk = pybamm.CoupledVariable(
                f"Average {domain} {phase_name}particle potential [V]",
                "current collector",
            )
            self.coupled_variables.update({ocp_bulk.name: ocp_bulk})
            dUdT = self.phase_param.dUdT(sto_surf)

        variables = self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT)
        self.variables.update(variables)
