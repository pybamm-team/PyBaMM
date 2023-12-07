#
# from Wycisk 2022
#
import pybamm
from . import BaseOpenCircuitPotential


class PlettOpenCircuitPotential(BaseOpenCircuitPotential):
    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        h = pybamm.Variable(f'{Domain} electrode {phase_name}hysteresis state',domains={
            'primary':f'{domain} electrode',
            'secondary':'current collector',
        })
        return {f'{Domain} electrode {phase_name}hysteresis state':h,}

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        phase = self.phase

        if self.reaction == "lithium-ion main":
            T = variables[f"{Domain} electrode temperature [K]"]
            h = variables[f'{Domain} electrode {phase_name}hysteresis state']
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
                h = pybamm.PrimaryBroadcast(h, [f'{domain} particle size'])
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

            c_s_rav = variables[
            f"R-averaged {domain} {phase_name}particle concentration distribution"
            ]
            # eps_s = variables[
            #     f"{Domain} electrode {phase_name}active material volume fraction"
            # ]
            # eps_s_av = pybamm.x_average(eps_s)
            # c_s_vol_av = pybamm.x_average(eps_s * c_s_rav) / eps_s_av
            # c_s_vol = eps_s * c_s_rav
            c_scale = self.phase_param.c_max
            # L = self.domain_param.L
            # A = self.param.A_cc
            sto_rav = c_s_rav / c_scale
            sto_rav = variables[f'R-averaged {domain} particle stoichiometry distribution']
            # sto_xav = variables[f'X-averaged {domain} particle stoichiometry distribution']
            # eps_s = pybamm.PrimaryBroadcast(eps_s,[f'{domain} particle size'])

            #! don't know how to broadcast this
            variables[f"Total lithium in {phase} phase in {domain} electrode [mol]"] =  sto_bulk* c_scale #c_s_vol * L * A


            H = self.phase_param.H(sto_rav)
            variables[f'{Domain} electrode {phase_name}equilibrium OCP [V]'] = ocp_surf_eq
            variables[f'{Domain} electrode {phase_name}bulk equilibrium OCP [V]'] = ocp_bulk_eq
            variables[f'{Domain} electrode {phase_name}OCP hysteresis [V]'] = H
            # dQdU = variables[f'{Domain} electrode {phase_name}differential capacity [A.s.V-1]']

            dU = pybamm.source(1,ocp_bulk_eq.diff(sto_bulk))
            # dU = pybamm.PrimaryBroadcast(dU,[f'{domain} electrode'])
            # dU = pybamm.PrimaryBroadcast(dU,[f'{domain} particle size'])
            # dU = pybamm.FullBroadcast(dU,broadcast_domains={'primary':f'{domain} particle size', 'secondary':f'{domain} electrode'})
            Q = variables[f"Total lithium in {phase} phase in {domain} electrode [mol]"] * pybamm.constants.F / 3600
            # Q = pybamm.PrimaryBroadcast(Q,f'{domain} electrode')
            dQ = Q.diff(sto_bulk)
            dQdU = dQ/dU
            variables[f'{Domain} electrode {phase_name}differential capacity [A.s.V-1]'] = dQdU
            variables[f'{Domain} electrode {phase_name}hysteresis state distribution'] = h
            # H = pybamm.PrimaryBroadcast(H,f'{domain} particle size')
            # h = pybamm.SecondaryBroadcast(h,f'current collector')
            # ocp_surf_eq = pybamm.SecondaryBroadcast(ocp_surf_eq,[f'{domain} electrode'])
            # ocp_bulk_eq = pybamm.PrimaryBroadcast(ocp_bulk_eq,[f'{domain} electrode'])
            # ocp_bulk_eq = pybamm.PrimaryBroadcast(ocp_bulk_eq, [f'{domain} particle size'])
            #! why is ocp surf not in electrode domain? why no x dimension?
            H_x_av = pybamm.x_average(H)
            h_x_av = pybamm.x_average(h)
            ocp_surf = ocp_surf_eq + H_x_av * h_x_av
            H_s_av = pybamm.size_average(H_x_av)
            h_s_av = pybamm.size_average(h_x_av)

            ocp_bulk = ocp_bulk_eq + H_s_av * h_s_av

        variables.update(self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT))
        return variables

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        current = self.param.current_with_time

        Q_cell = variables[f'{Domain} electrode capacity [A.h]']
        dQdU = variables[f'{Domain} electrode {phase_name}differential capacity [A.s.V-1]']
        K = self.phase_param.K
        K_x = self.phase_param.K_x
        h = variables[f'{Domain} electrode {phase_name}hysteresis state distribution']

        dhdt = K * (current/(Q_cell*(dQdU**K_x)))*(1-pybamm.sign(current)*h) #! current is backwards for a halfcell
        self.rhs[h] = dhdt

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        h = variables[f'{Domain} electrode {phase_name}hysteresis state']
        h_av = variables[f'{Domain} electrode {phase_name}hysteresis state distribution']
        self.initial_conditions[h_av] = pybamm.Scalar(0)
        self.initial_conditions[h] = pybamm.Scalar(0)
