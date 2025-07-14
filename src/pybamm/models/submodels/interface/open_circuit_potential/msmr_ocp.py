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

    def __init__(
        self, param, domain, reaction, options, phase="primary", x_average=False
    ):
        super().__init__(
            param, domain, reaction, options=options, phase=phase, x_average=x_average
        )
        pybamm.citations.register("Baker2018")

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
                    f"{Domain} {phase_name}particle surface potential distribution [V]"
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
                    f"{Domain} {phase_name}particle surface potential [V]"
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

            ocp_bulk = variables[f"Average {domain} {phase_name}particle potential [V]"]
            dUdT = self.phase_param.dUdT(sto_surf)

        variables.update(self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT))
        variables.update(
            {
                f"{Domain} electrode {phase_name}equilibrium open-circuit potential [V]": ocp_surf,
                f"X-averaged {domain} electrode {phase_name}equilibrium open-circuit potential [V]": pybamm.x_average(
                    ocp_surf
                ),
            }
        )
        return variables
