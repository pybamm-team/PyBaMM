#
# Open-circuit potential from the Multi-Species Multi-Reaction framework
#
import pybamm
from . import BaseOpenCircuitPotential


class MSMROpenCircuitPotential(BaseOpenCircuitPotential):
    """
    Class for open-circuit potential within the Multi-Species Multi-Reaction
    framework [1]_.

    References
    ----------
    .. [1] DR Baker and MW Verbrugge. "Multi-species, multi-reaction model for porous
           intercalation electrodes: Part I. Model formulation and a perturbation
           solution for low-scan-rate, linear-sweep voltammetry of a spinel lithium
           manganese oxide electrode." Journal of The Electrochemical Society,
           165(16):A3952, 2019
    """

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
                ocp_surf = variables[
                    f"{Domain} {phase_name}particle surface open-circuit potential "
                    "distribution [V]"
                ]
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
                sto_surf = variables[
                    f"{Domain} {phase_name}particle surface stoichiometry"
                ]
                ocp_surf = variables[
                    f"{Domain} {phase_name}particle surface open-circuit potential [V]"
                ]
                # If variable was broadcast, take only the orphan
                if (
                    isinstance(sto_surf, pybamm.Broadcast)
                    and isinstance(ocp_surf, pybamm.Broadcast)
                    and isinstance(T, pybamm.Broadcast)
                ):
                    sto_surf = sto_surf.orphans[0]
                    ocp_surf = ocp_surf.orphans[0]
                    T = T.orphans[0]

            ocp_bulk = variables[
                f"Average {domain} {phase_name}particle open-circuit potential [V]"
            ]
            dUdT = self.phase_param.dUdT(sto_surf)

        variables.update(self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT))
        return variables
