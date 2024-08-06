#
# Tanks-in-Series Model
#
import pybamm
from .base_lithium_ion_model import BaseModel


class TanksInSeries(BaseModel):
    """Tanks-in-Series model of a lithium-ion battery from #TODO add citation

    Parameters
    ----------
    name: str, optional
        The name of the model.
    """

    def __init__(self, name="Tanks-in-Series Model"):
        super().__init__({}, name)
        # pybamm.citations.register("link to paper")
        param = self.param

        ######################
        # Variables
        ######################
        # All variables are only-time dependent

        Q = pybamm.Variable("Discharge capacity [A.h]")

        c_s_n = pybamm.Variable("X-averaged negative particle concentration [mol.m-3]")
        c_s_p = pybamm.Variable("X-averaged positive particle concentration [mol.m-3]")

        c_e_n = pybamm.Variable(
            "X-averaged negative electrolyte concentration [mol.m-3]"
        )
        c_e_s = pybamm.Variable(
            "X-averaged separator electrolyte concentration [mol.m-3]"
        )
        c_e_p = pybamm.Variable(
            "X-averaged positive electrolyte concentration [mol.m-3]"
        )

        c_surf_ave_n = pybamm.Variable(
            "X-averaged negative particle surface concentration [mol.m-3]"
        )
        c_surf_ave_p = pybamm.Variable(
            "X-averaged positive particle surface concentration [mol.m-3]"
        )

        q_ave_n = pybamm.Variable(
            "X-averaged negative particle concentration gradient [mol.m-4]"
        )
        q_ave_p = pybamm.Variable(
            "X-averaged positive particle concentration gradient [mol.m-4]"
        )

        phi_e_n = pybamm.Variable("X-averaged negative electrolyte potential [V]")
        phi_e_s = pybamm.Variable("X-averaged separator electrolyte potential [V]")
        phi_e_p = pybamm.Variable("X-averaged positive electrolyte potential [V]")

        phi_s_p = pybamm.Variable("X-averaged positive electrode potential [V]")
        phi_s_n = pybamm.Variable("X-averaged negative electrode potential [V]")

        # Porosity
        eps_p = pybamm.Parameter("Positive electrode porosity")
        eps_n = pybamm.Parameter("Negative electrode porosity")
        eps_sep = pybamm.Parameter("Separator porosity")

        # Isothermal model with constant temperature
        T = param.T_init

        # Current density
        iapp = -param.current_density_with_time
        a_n = 3 * param.n.prim.epsilon_s_av / param.n.prim.R_typ
        a_p = 3 * param.p.prim.epsilon_s_av / param.p.prim.R_typ

        # Interfacial reactions
        sto_surf_n = c_surf_ave_n / param.n.prim.c_max
        j0_n = param.n.prim.j0(c_e_n, c_surf_ave_n, T)
        eta_n = phi_s_n - phi_e_n - param.n.prim.U(sto_surf_n, T)
        alpha_n = 0.5  # param.n.alpha_bv
        Feta_RT_n = param.F * eta_n / (param.R * T)
        # j_n = j0_n * (np.exp((1 - alpha_n) * Feta_RT_n) - np.exp(-alpha_n * Feta_RT_n))
        j_n = 2 * j0_n * pybamm.sinh(param.n.prim.ne / 2 * Feta_RT_n)

        sto_surf_p = c_surf_ave_p / param.p.prim.c_max
        j0_p = param.p.prim.j0(c_e_p, c_surf_ave_p, T)
        eta_p = phi_s_p - phi_e_p - param.p.prim.U(sto_surf_p, T)
        alpha_p = 0.5  # param.p.alpha_bv
        Feta_RT_p = param.F * eta_p / (param.R * T)
        # j_p = j0_p * (np.exp((1 - alpha_p) * Feta_RT_p) - np.exp(-alpha_p * Feta_RT_p))
        j_p = 2 * j0_p * pybamm.sinh(param.p.prim.ne / 2 * Feta_RT_p)

        ######################
        # State of Charge
        ######################

        I = param.current_with_time
        # The `rhs` dictionary contains differential equations, with the key being the
        # variable in the d/dt
        self.rhs[Q] = I / 3600
        # Initial conditions must be provided for the ODEs
        self.initial_conditions[Q] = pybamm.Scalar(0)

        ######################
        # Particles
        ######################

        # Particle spatially discretized with 3-parameter model
        # based on biquadratic profile for particle concentration

        self.rhs[c_s_n] = -3 * j_n / param.n.prim.R_typ
        self.rhs[c_s_p] = -3 * j_p / param.p.prim.R_typ

        self.initial_conditions[c_s_n] = param.n.prim.c_init_av
        self.initial_conditions[c_s_p] = param.p.prim.c_init_av

        self.rhs[q_ave_n] = (
            -30 * param.n.prim.D(c_s_n, T) * q_ave_n / param.n.prim.R_typ**2
            - 45 / 2 * j_n / param.n.prim.R_typ
        )
        self.rhs[q_ave_p] = (
            -30 * param.p.prim.D(c_s_p, T) * q_ave_p / param.p.prim.R_typ**2
            - 45 / 2 * j_p / param.p.prim.R_typ
        )

        self.initial_conditions[q_ave_n] = pybamm.Scalar(0)
        self.initial_conditions[q_ave_p] = pybamm.Scalar(0)

        ######################
        # Boundary values
        ######################

        # 12: boundary between positive electrode and separator
        c_12 = (
            eps_p**param.p.b_e / param.p.L * c_e_p
            + eps_sep**param.s.b_e / param.s.L * c_e_s
        ) / (eps_p**param.p.b_e / param.p.L + eps_sep**param.s.b_e / param.s.L)
        # 23: boundary between separator and negative electrode
        c_23 = (
            eps_n**param.n.b_e / param.n.L * c_e_n
            + eps_sep**param.s.b_e / param.s.L * c_e_s
        ) / (eps_n**param.n.b_e / param.n.L + eps_sep**param.s.b_e / param.s.L)

        # Grouped thickness/porosity variables
        # positive electrode and separator
        leps12 = param.p.L / eps_p**param.p.b_e + param.s.L / eps_sep**param.s.b_e
        # negative electrode and separator
        leps23 = param.n.L / eps_n**param.n.b_e + param.s.L / eps_sep**param.s.b_e

        ######################
        # Electrolyte concentration
        ######################

        self.rhs[c_e_n] = (
            -2 * param.D_e(c_23, T) * (c_e_n - c_e_s) / leps23 / (eps_n * param.n.L)
            - (1 - param.t_plus(c_e_n, T)) * iapp / param.F / eps_n / param.n.L
        )
        self.rhs[c_e_s] = (
            -2 * param.D_e(c_12, T) * (c_e_s - c_e_p) / leps12
            + 2 * param.D_e(c_23, T) * (c_e_n - c_e_s) / leps23
        ) / (eps_sep * param.s.L)
        self.rhs[c_e_p] = (
            2 * param.D_e(c_12, T) * (c_e_s - c_e_p) / leps12 / (eps_p * param.p.L)
            + (1 - param.t_plus(c_e_p, T)) * iapp / param.F / eps_p / param.p.L
        )

        self.initial_conditions[c_e_n] = param.c_e_init
        self.initial_conditions[c_e_s] = param.c_e_init
        self.initial_conditions[c_e_p] = param.c_e_init

        ######################
        # Algebraic Variables
        ######################

        ######################
        # Average Surface Concentration
        ######################

        self.algebraic[c_surf_ave_p] = (
            j_p
            + 35
            * param.p.prim.D(c_surf_ave_p, T)
            / param.p.prim.R_typ
            * (c_surf_ave_p - c_s_p)
            - 8 * param.p.prim.D(c_surf_ave_p, T) * q_ave_p
        )
        self.algebraic[c_surf_ave_n] = (
            j_n
            + 35
            * param.n.prim.D(c_surf_ave_n, T)
            / param.n.prim.R_typ
            * (c_surf_ave_n - c_s_n)
            - 8 * param.n.prim.D(c_surf_ave_n, T) * q_ave_n
        )

        self.initial_conditions[c_surf_ave_p] = param.p.prim.c_init_av
        self.initial_conditions[c_surf_ave_n] = param.n.prim.c_init_av

        ######################
        # Current in the electrolyte
        ######################

        RT_F = param.R * T / param.F
        elec_thermo_p = param.chi(c_12, T) / 2 * param.kappa_e(c_12, T)
        self.algebraic[phi_e_p] = (
            -iapp
            - 2 * param.kappa_e(c_12, T) * (phi_e_s - phi_e_p) / leps12
            + 2 * param.chiRT_over_Fc(c_12, T) * (c_e_s - c_e_p) / leps12
        )
        elec_thermo_n = param.chi(c_23, T) / 2 * param.kappa_e(c_23, T)
        self.algebraic[phi_e_n] = (
            -iapp
            - 2 * param.kappa_e(c_23, T) * (phi_e_n - phi_e_s) / leps23
            + 2 * param.chiRT_over_Fc(c_23, T) * (c_e_n - c_e_s) / leps23
        )
        self.algebraic[phi_e_s] = (
            eps_p**param.p.b_e / param.p.L * phi_e_p
            + eps_sep**param.s.b_e / param.s.L * phi_e_s
        )

        self.initial_conditions[phi_e_p] = pybamm.Scalar(0)
        self.initial_conditions[phi_e_n] = pybamm.Scalar(0)
        self.initial_conditions[phi_e_s] = pybamm.Scalar(0)

        ######################
        # Current in the solid
        ######################

        self.algebraic[phi_s_p] = -iapp / param.F / a_p / param.p.L + j_p
        self.algebraic[phi_s_n] = iapp / param.F / a_n / param.n.L + j_n

        self.initial_conditions[phi_s_p] = param.ocv_init
        self.initial_conditions[phi_s_n] = pybamm.Scalar(0)

        ######################
        # More variables
        ######################

        voltage = phi_s_p - phi_s_n
        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )

        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        self.variables = {
            "X-averaged negative electrolyte concentration [mol.m-3]": c_e_n,
            "X-averaged positive electrolyte concentration [mol.m-3]": c_e_p,
            "X-averaged separator electrolyte concentration [mol.m-3]": c_e_s,
            "X-averaged negative particle concentration [mol.m-3]": c_s_n,
            "X-averaged positive particle concentration [mol.m-3]": c_s_p,
            "X-averaged negative particle surface concentration [mol.m-3]": c_surf_ave_n,
            "X-averaged positive particle surface concentration [mol.m-3]": c_surf_ave_p,
            "X-averaged negative particle concentration gradient [mol.m-4]": q_ave_n,
            "X-averaged positive particle concentration gradient [mol.m-4]": q_ave_p,
            "X-averaged positive electrolyte potential [V]": phi_e_p,
            "X-averaged negative electrolyte potential [V]": phi_e_n,
            "X-averaged separator electrolyte potential [V]": phi_e_s,
            "X-averaged positive electrode potential [V]": phi_s_p,
            "X-averaged negative electrode potential [V]": phi_s_n,
            "Current [A]": I,
            "Current variable [A]": I,  # for compatibility with pybamm.Experiment
            "Voltage [V]": voltage,
            "Battery voltage [V]": voltage * num_cells,
            "Time [s]": pybamm.t,
            "Discharge capacity [A.h]": Q,
            "Positive overpotential": eta_p,
            "Negative overpotential": eta_n,
            "Negative pore wall flux": j_n,
            "Positive pore wall flux": j_p,
        }

        # Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event("Minimum voltage [V]", voltage - param.voltage_low_cut),
            pybamm.Event("Maximum voltage [V]", param.voltage_high_cut - voltage),
        ]
