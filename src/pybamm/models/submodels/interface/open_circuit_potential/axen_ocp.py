#
# from Axen 2022
#
import pybamm
from . import BaseOpenCircuitPotential


class AxenOpenCircuitPotential(BaseOpenCircuitPotential):
    """
    Class for open-circuit potential (OCP) with hysteresis based on the approach outlined by Axen et al. https://doi.org/10.1016/j.est.2022.103985.
    This approach tracks the evolution of the hysteresis using an ODE system scaled by the volumetric interfacial current density and
    a decay rate parameter.

    The original method defines the open-circuit potential as OCP = OCP_delithiation + (OCP_lithiation - OCP_delithiation) * h,
    where h is the hysteresis state. By allowing h to vary between 0 and 1, the OCP approaches OCP_lithiation when h tends to 1
    and OCP_delithiation when h tends to 0. These interval limits are achieved by defining the ODE system as dh/dt = S_lith * (1 - h)
    for lithiation and dh/dt = S_delith * h for delithiation, where S_lith and S_delith are rate coefficients.

    The original implementation is modified to ensure h varies between -1 and 1, thus assuming an unified framework within pybamm.
    This new variation interval is obtained with the ODE system dh/dt = S_lith * (1 + h) for lithiation and dh/dt = S_delith * (1 - h)
    for delithiation. Now, OCP = OCP_delithiation + (OCP_lithiation - OCP_delithiation) * (1 - h) / 2, implying that the OCP approaches
    OCP_lithiation when h tends to -1 and OCP_delithiation when h tends to 1.
    """

    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        h = pybamm.Variable(
            f"{Domain} electrode {phase_name}hysteresis state",
            domains={
                "primary": f"{domain} electrode",
                "secondary": "current collector",
            },
        )
        return {
            f"{Domain} electrode {phase_name}hysteresis state": h,
        }

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        if self.reaction == "lithium-ion main":
            T = variables[f"{Domain} electrode temperature [K]"]
            h = variables[f"{Domain} electrode {phase_name}hysteresis state"]

            # For "particle-size distribution" models, take distribution version
            # of c_s_surf that depends on particle size.
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
                T = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}particle size"])
                h = pybamm.PrimaryBroadcast(h, [f"{domain} {phase_name}particle size"])
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

            variables[
                f"{Domain} electrode {phase_name}hysteresis state distribution"
            ] = h

            # Bulk OCP is from the average SOC and temperature
            lith_ref = self.phase_param.U(sto_surf, T, "lithiation")
            delith_ref = self.phase_param.U(sto_surf, T, "delithiation")
            H = lith_ref - delith_ref
            variables[f"{Domain} electrode {phase_name}OCP hysteresis [V]"] = H

            delith_ref_x_av = pybamm.x_average(delith_ref)
            H_x_av = pybamm.x_average(H)
            h_x_av = pybamm.x_average(h)
            variables[f"X-averaged {domain} electrode {phase_name}hysteresis state"] = (
                h_x_av
            )

            # check if psd
            if domain_options["particle size"] == "distribution":
                # should always be true
                if f"{domain} particle size" in sto_surf.domains["primary"]:
                    # check if MPM Model
                    if "current collector" in sto_surf.domains["secondary"]:
                        ocp_surf = delith_ref_x_av + H_x_av * (1 - h_x_av) / 2
                    # must be DFN with PSD model
                    elif (
                        f"{domain} electrode" in sto_surf.domains["secondary"]
                        or f"{domain} {phase_name}particle size"
                        in sto_surf.domains["primary"]
                    ):
                        ocp_surf = delith_ref + H * (1 - h) / 2
            # must not be a psd
            else:
                ocp_surf = delith_ref + H * (1 - h) / 2

            delith_ref_s_av = pybamm.size_average(delith_ref_x_av)
            H_s_av = pybamm.size_average(H_x_av)
            h_s_av = pybamm.size_average(h_x_av)

            ocp_bulk = delith_ref_s_av + H_s_av * (1 - h_s_av) / 2

            dUdT = self.phase_param.dUdT(sto_surf)

        variables.update(self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT))
        return variables

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        i_vol = variables[
            f"{Domain} electrode {phase_name}volumetric interfacial current density [A.m-3]"
        ]
        c_max = self.phase_param.c_max
        epsl = self.phase_param.epsilon_s

        K_lith = self.phase_param.hysteresis_decay("lithiation")
        K_delith = self.phase_param.hysteresis_decay("delithiation")
        h = variables[f"{Domain} electrode {phase_name}hysteresis state"]

        S_lith = i_vol * K_lith / (self.param.F * c_max * epsl)
        S_delith = i_vol * K_delith / (self.param.F * c_max * epsl)

        i_vol_sign = pybamm.sign(i_vol)
        signed_h = 1 - i_vol_sign * h
        rate_coefficient = (
            0.5 * (1 - i_vol_sign) * S_lith + 0.5 * (1 + i_vol_sign) * S_delith
        )

        dhdt = rate_coefficient * signed_h

        self.rhs[h] = dhdt

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        h = variables[f"{Domain} electrode {phase_name}hysteresis state"]
        self.initial_conditions[h] = self.phase_param.h_init
