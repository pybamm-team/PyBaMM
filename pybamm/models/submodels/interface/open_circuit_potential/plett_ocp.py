#
# from Wycisk 2022
#
import pybamm
from . import BaseOpenCircuitPotential


class PlettOpenCircuitPotential(BaseOpenCircuitPotential):
    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        h = pybamm.Variable(f'{Domain} electrode {phase_name}hysteresis state')
        return {f'{Domain} electrode {phase_name}hysteresis state':h,}

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

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

            H = self.phase_param.H(sto_bulk)
            variables[f'{Domain} electrode {phase_name}equilibrium OCP [V]'] = ocp_surf_eq
            variables[f'{Domain} electrode {phase_name}bulk equilibrium OCP [V]'] = ocp_bulk_eq
            variables[f'{Domain} electrode {phase_name} OCP hysteresis [V]'] = H
            # dQdU = variables[f'{Domain} electrode {phase_name}differential capacity [A.s.V-1]']

            dU = pybamm.source(1,ocp_bulk_eq.diff(sto_bulk))
            Q = self.phase_param.Q(sto_bulk)
            dQ = Q.diff(sto_bulk)
            dQdU = dQ/dU
            variables[f'{Domain} electrode {phase_name}differential capacity [A.s.V-1]'] = dQdU
            ocp_surf = ocp_surf_eq + H * h
            ocp_bulk = ocp_bulk_eq + H * h

        variables.update(self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT))
        return variables

    # def set_algebraic(self, variables):
    #     domain, Domain = self.domain_Domain
    #     phase_name = self.phase_name
    #     dQdU = variables[f'{Domain} electrode {phase_name}differential capacity [A.s.V-1]']

    #     return super().set_algebraic(variables)

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        current = self.param.current_with_time

        Q_cell = variables[f'{Domain} electrode capacity [A.h]']
        dQdU = variables[f'{Domain} electrode {phase_name}differential capacity [A.s.V-1]']
        K = self.phase_param.K
        K_x = self.phase_param.K_x
        h = variables[f'{Domain} electrode {phase_name}hysteresis state']

        dhdt = K * (current/(Q_cell*(dQdU**K_x)))*(1-pybamm.sign(current)*h)
        self.rhs[h] = dhdt

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        h = pybamm.Variable(f'{Domain} electrode {phase_name}hysteresis state')
        self.initial_conditions[h] = pybamm.Scalar(0)

    # def set_boundary_conditions(self, variables):
    #     self.boundary_conditions[]
