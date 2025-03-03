#
# Different OCPs for charge and discharge, based on current
#
import pybamm
from . import BaseOpenCircuitPotential


class CurrentSigmoidOpenCircuitPotential(BaseOpenCircuitPotential):
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

        if self.reaction == "lithium-ion main":
            # Get delithiation and lithiation OCPs
            sto_surf, T = self._get_stoichiometry_and_temperature(variables)
            U_lith = self.phase_param.U(sto_surf, T, "lithiation")
            U_delith = self.phase_param.U(sto_surf, T, "delithiation")
            U_delith_x_av = pybamm.x_average(U_delith)

            H = U_lith - U_delith
            H_x_av = pybamm.x_average(H)
            variables.update(
                {
                    f"{Domain} electrode {phase_name}OCP hysteresis [V]": H,
                    f"X-averaged {domain} electrode {phase_name}OCP hysteresis [V]": H_x_av,
                }
            )

            # Get correct hysteresis state variable
            if domain_options["particle size"] == "distribution":
                h = variables[
                    f"{Domain} electrode {phase_name}hysteresis state distribution"
                ]
                h_x_av = variables[
                    f"X-averaged {domain} electrode {phase_name}hysteresis state distribution"
                ]
            else:
                h = variables[f"{Domain} electrode {phase_name}hysteresis state"]
                h_x_av = variables[
                    f"X-averaged {domain} electrode {phase_name}hysteresis state"
                ]

            # check if psd
            if domain_options["particle size"] == "distribution":
                # should always be true
                if f"{domain} particle size" in sto_surf.domains["primary"]:
                    # check if MPM Model
                    if "current collector" in sto_surf.domains["secondary"]:
                        ocp_surf = U_delith_x_av + H_x_av * (1 - h_x_av) / 2
                    # must be DFN with PSD model
                    elif (
                        f"{domain} electrode" in sto_surf.domains["secondary"]
                        or f"{domain} {phase_name}particle size"
                        in sto_surf.domains["primary"]
                    ):
                        ocp_surf = U_delith + H * (1 - h) / 2
            # must not be a psd
            else:
                ocp_surf = U_delith + H * (1 - h) / 2

            # Size average
            U_delith_s_av = pybamm.size_average(U_delith_x_av)
            H_s_av = pybamm.size_average(H_x_av)
            h_s_av = pybamm.size_average(h_x_av)

            ocp_bulk = U_delith_s_av + H_s_av * (1 - h_s_av) / 2
            dUdT = self.phase_param.dUdT(sto_surf)

        variables.update(self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT))
        return variables
