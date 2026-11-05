#
# Different OCPs for charge and discharge, based on current
#
import pybamm

from . import BaseHysteresisOpenCircuitPotential


class CurrentSigmoidOpenCircuitPotential(BaseHysteresisOpenCircuitPotential):
    def __init__(
        self, param, domain, reaction, options, phase="primary", x_average=False
    ):
        super().__init__(
            param, domain, reaction, options=options, phase=phase, x_average=x_average
        )
        pybamm.citations.register("Ai2022")

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        domain_options = getattr(self.options, domain)
        phase_name = self.phase_name

        # Set hysteresis state
        current = variables["Total current density [A.m-2]"]
        k = 100

        if Domain == "Positive":
            lithiation_current = current
        elif Domain == "Negative":
            lithiation_current = -current

        h_x_av = 1 - 2 * pybamm.sigmoid(0, lithiation_current, k)
        h = pybamm.FullBroadcast(
            h_x_av,
            [f"{domain} electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        variables.update(
            {
                f"{Domain} electrode {phase_name}hysteresis state": h,
                f"X-averaged {domain} electrode {phase_name}hysteresis state": h_x_av,
            }
        )
        if domain_options["particle size"] == "distribution":
            h_dist = pybamm.PrimaryBroadcast(h, [f"{domain} {phase_name}particle size"])
            h_dist_x_av = pybamm.x_average(h_dist)
            variables.update(
                {
                    f"{Domain} electrode {phase_name}hysteresis state distribution": h_dist,
                    f"X-averaged {domain} electrode {phase_name}hysteresis state distribution": h_dist_x_av,
                }
            )

        # Get single-state hysteresis variables
        variables.update(self._get_coupled_variables(variables))

        return variables

    def set_initial_conditions(self, variables):
        pass
