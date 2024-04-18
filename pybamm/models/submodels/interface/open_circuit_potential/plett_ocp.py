#
# from Wycisk 2022
#
import pybamm
from . import BaseOpenCircuitPotential


class PlettOpenCircuitPotential(BaseOpenCircuitPotential):
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
        phase = self.phase

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
                T = pybamm.PrimaryBroadcast(T, [f"{domain} particle size"])
                h = pybamm.PrimaryBroadcast(h, [f"{domain} particle size"])
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

            ocp_surf_eq = self.phase_param.U(sto_surf, T)
            dUdT = self.phase_param.dUdT(sto_surf)

            # Bulk OCP is from the average SOC and temperature
            sto_bulk = variables[f"{Domain} electrode {phase_name}stoichiometry"]
            T_bulk = pybamm.xyz_average(pybamm.size_average(T))
            ocp_bulk_eq = self.phase_param.U(sto_bulk, T_bulk)

            # # get dQ, change in capacity with respect to stoichiometry
            # c_max = self.phase_param.c_max
            # epsilon_s_av = self.phase_param.epsilon_s_av
            # V_electrode = self.param.A_cc * self.domain_param.L
            # Li_max = c_max * V_electrode * epsilon_s_av
            # Q_max = Li_max * self.param.F / 3600
            # dQ = Q_max * sto_bulk

            c_scale = self.phase_param.c_max
            variables[f"Total lithium in {phase} phase in {domain} electrode [mol]"] = (
                sto_bulk * c_scale
            )  # c_s_vol * L * A

            H = self.phase_param.H(sto_surf)
            variables[f"{Domain} electrode {phase_name}equilibrium OCP [V]"] = (
                ocp_surf_eq
            )
            variables[f"{Domain} electrode {phase_name}bulk equilibrium OCP [V]"] = (
                ocp_bulk_eq
            )
            variables[f"{Domain} electrode {phase_name}OCP hysteresis [V]"] = H

            dU = self.phase_param.U(sto_surf, T_bulk).diff(sto_surf)
            dQ = self.phase_param.Q(sto_surf).diff(sto_surf)
            dQdU = dQ / dU
            variables[
                f"{Domain} electrode {phase_name}differential capacity [A.s.V-1]"
            ] = dQdU
            variables[
                f"{Domain} electrode {phase_name}hysteresis state distribution"
            ] = h
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
                        ocp_surf = ocp_surf_eq + H_x_av * h_x_av
                    # must be DFN with PSD model
                    elif f"{domain} electrode" in sto_surf.domains["secondary"]:
                        ocp_surf = ocp_surf_eq + H * h
                    else:
                        raise ValueError(
                            'Model type not implementted with open-circuit potential as "Plett"'
                        )
                else:
                    raise ValueError(
                        'Model type not implementted with open-circuit potential as "Plett"'
                    )
            # must not be a psd
            else:
                ocp_surf = ocp_surf_eq + H * h
            H_s_av = pybamm.size_average(H_x_av)
            h_s_av = pybamm.size_average(h_x_av)

            ocp_bulk = ocp_bulk_eq + H_s_av * h_s_av

        variables.update(self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT))
        return variables

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        current = self.param.current_with_time
        current = variables[
            f"{Domain} electrode {phase_name}interfacial current density [A.m-2]"
        ]
        Q_cell = variables[f"{Domain} electrode capacity [A.h]"]
        dQdU = variables[
            f"{Domain} electrode {phase_name}differential capacity [A.s.V-1]"
        ]
        dQdU = dQdU.orphans[0]
        K = self.phase_param.K
        K_x = self.phase_param.K_x
        h = variables[f"{Domain} electrode {phase_name}hysteresis state"]

        dhdt = (
            K * (current / (Q_cell * (dQdU**K_x))) * (1 - pybamm.sign(current) * h)
        )  #! current is backwards for a halfcell
        self.rhs[h] = dhdt

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        h = variables[f"{Domain} electrode {phase_name}hysteresis state"]
        self.initial_conditions[h] = pybamm.Scalar(0)
