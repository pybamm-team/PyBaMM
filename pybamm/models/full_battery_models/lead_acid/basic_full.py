#
# Basic lead-acid model
#
import pybamm
from .base_lead_acid_model import BaseModel


class BasicFull(BaseModel):
    """
    Porous electrode model for lead-acid, from [2]_.

    This class differs from the :class:`pybamm.lead_acid.Full` model class in that it
    shows the whole model in a single class. This comes at the cost of flexibility in
    comparing different physical effects, and in general the main DFN class should be
    used instead.

    Parameters
    ----------
    name : str, optional
        The name of the model.

    References
    ----------
    .. [2] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster lead-acid
           battery simulations from porous-electrode theory: Part II. Asymptotic
           analysis. Journal of The Electrochemical Society 166.12 (2019), A2372–A2382..
    """

    def __init__(self, name="Basic full model"):
        super().__init__({}, name)
        # `param` is a class containing all the relevant parameters and functions for
        # this model. These are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.
        param = self.param

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = pybamm.Variable("Discharge capacity [A.h]")
        # Variables that vary spatially are created with a domain
        c_e_n = pybamm.Variable(
            "Negative electrolyte concentration [mol.m-3]",
            domain="negative electrode",
            scale=param.c_e_init,
        )
        c_e_s = pybamm.Variable(
            "Separator electrolyte concentration [mol.m-3]",
            domain="separator",
            scale=param.c_e_init,
        )
        c_e_p = pybamm.Variable(
            "Positive electrolyte concentration [mol.m-3]",
            domain="positive electrode",
            scale=param.c_e_init,
        )
        # Concatenations combine several variables into a single variable, to simplify
        # implementing equations that hold over several domains
        c_e = pybamm.concatenation(c_e_n, c_e_s, c_e_p)

        # Electrolyte potential
        phi_e_n = pybamm.Variable(
            "Negative electrolyte potential [V]",
            domain="negative electrode",
            reference=-param.n.prim.U_init,
        )
        phi_e_s = pybamm.Variable(
            "Separator electrolyte potential [V]",
            domain="separator",
            reference=-param.n.prim.U_init,
        )
        phi_e_p = pybamm.Variable(
            "Positive electrolyte potential [V]",
            domain="positive electrode",
            reference=-param.n.prim.U_init,
        )
        phi_e = pybamm.concatenation(phi_e_n, phi_e_s, phi_e_p)

        # Electrode potential
        phi_s_n = pybamm.Variable(
            "Negative electrode potential [V]", domain="negative electrode"
        )
        phi_s_p = pybamm.Variable(
            "Positive electrode potential [V]",
            domain="positive electrode",
            reference=param.ocv_init,
        )

        # Porosity
        eps_n = pybamm.Variable(
            "Negative electrode porosity", domain="negative electrode"
        )
        eps_s = pybamm.Variable("Separator porosity", domain="separator")
        eps_p = pybamm.Variable(
            "Positive electrode porosity", domain="positive electrode"
        )
        eps = pybamm.concatenation(eps_n, eps_s, eps_p)

        # Constant temperature
        T = param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = param.current_density_with_time

        # transport_efficiency
        tor = pybamm.concatenation(
            eps_n**param.n.b_e, eps_s**param.s.b_e, eps_p**param.p.b_e
        )

        # Interfacial reactions
        F_RT = param.F / (param.R * T)
        Feta_RT_n = F_RT * (phi_s_n - phi_e_n - param.n.prim.U(c_e_n, T))
        j0_n = param.n.prim.j0(c_e_n, T)
        j_n = 2 * j0_n * pybamm.sinh(param.n.prim.ne / 2 * Feta_RT_n)
        j_s = pybamm.PrimaryBroadcast(0, "separator")
        Feta_RT_p = F_RT * (phi_s_p - phi_e_p - param.p.prim.U(c_e_p, T))
        j0_p = param.p.prim.j0(c_e_p, T)
        j_p = 2 * j0_p * pybamm.sinh(param.p.prim.ne / 2 * (Feta_RT_p))

        a_n = pybamm.Parameter("Negative electrode surface area to volume ratio [m-1]")
        a_p = pybamm.Parameter("Positive electrode surface area to volume ratio [m-1]")
        a_j_n = a_n * j_n
        a_j_p = a_p * j_p
        a_j = pybamm.concatenation(a_j_n, j_s, a_j_p)

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
        # Current in the electrolyte
        ######################
        i_e = (param.kappa_e(c_e, T) * tor) * (
            param.chiRT_over_Fc(c_e, T) * pybamm.grad(c_e) - pybamm.grad(phi_e)
        )
        # multiply by Lx**2 to improve conditioning
        self.algebraic[phi_e] = (pybamm.div(i_e) - a_j) * param.L_x**2
        self.boundary_conditions[phi_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[phi_e] = -param.n.prim.U_init

        ######################
        # Current in the solid
        ######################
        i_s_n = -param.n.sigma(T) * (1 - eps_n) ** param.n.b_s * pybamm.grad(phi_s_n)
        sigma_eff_p = param.p.sigma(T) * (1 - eps_p) ** param.p.b_s
        i_s_p = -sigma_eff_p * pybamm.grad(phi_s_p)
        # The `algebraic` dictionary contains differential equations, with the key being
        # the main scalar variable of interest in the equation
        # multiply by Lx**2 to improve conditioning
        self.algebraic[phi_s_n] = (pybamm.div(i_s_n) + a_j_n) * param.L_x**2
        self.algebraic[phi_s_p] = (pybamm.div(i_s_p) + a_j_p) * param.L_x**2
        self.boundary_conditions[phi_s_n] = {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.boundary_conditions[phi_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (i_cell / pybamm.boundary_value(-sigma_eff_p, "right"), "Neumann"),
        }
        # Initial conditions must also be provided for algebraic equations, as an
        # initial guess for a root-finding algorithm which calculates consistent initial
        # conditions
        self.initial_conditions[phi_s_n] = pybamm.Scalar(0)
        self.initial_conditions[phi_s_p] = param.ocv_init

        ######################
        # Porosity
        ######################
        DeltaVsurf = pybamm.concatenation(
            pybamm.PrimaryBroadcast(param.n.DeltaVsurf, "negative electrode"),
            pybamm.PrimaryBroadcast(0, "separator"),
            pybamm.PrimaryBroadcast(param.p.DeltaVsurf, "positive electrode"),
        )
        deps_dt = DeltaVsurf * a_j / param.F
        self.rhs[eps] = deps_dt
        self.initial_conditions[eps] = param.epsilon_init
        self.events.extend(
            [
                pybamm.Event(
                    "Zero negative electrode porosity cut-off", pybamm.min(eps_n)
                ),
                pybamm.Event(
                    "Max negative electrode porosity cut-off", 1 - pybamm.max(eps_n)
                ),
                pybamm.Event(
                    "Zero positive electrode porosity cut-off", pybamm.min(eps_p)
                ),
                pybamm.Event(
                    "Max positive electrode porosity cut-off", 1 - pybamm.max(eps_p)
                ),
            ]
        )

        ######################
        # Electrolyte concentration
        ######################
        N_e = (
            -tor * param.D_e(c_e, T) * pybamm.grad(c_e)
            + param.t_plus(c_e, T) * i_e / param.F
        )
        s = pybamm.concatenation(
            pybamm.PrimaryBroadcast(param.n.prim.s_plus_S, "negative electrode"),
            pybamm.PrimaryBroadcast(0, "separator"),
            pybamm.PrimaryBroadcast(param.p.prim.s_plus_S, "positive electrode"),
        )
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) + s * a_j / param.F - c_e * deps_dt
        )
        self.boundary_conditions[c_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[c_e] = param.c_e_init
        self.events.append(
            pybamm.Event(
                "Zero electrolyte concentration cut-off", pybamm.min(c_e) - 0.002
            )
        )

        ######################
        # (Some) variables
        ######################
        voltage = pybamm.boundary_value(phi_s_p, "right")
        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        self.variables = {
            "Electrolyte concentration [mol.m-3]": c_e,
            "Current [A]": I,
            "Negative electrode potential [V]": phi_s_n,
            "Electrolyte potential [V]": phi_e,
            "Positive electrode potential [V]": phi_s_p,
            "Voltage [V]": voltage,
            "Porosity": eps,
        }
        self.events.extend(
            [
                pybamm.Event("Minimum voltage [V]", voltage - param.voltage_low_cut),
                pybamm.Event("Maximum voltage [V]", param.voltage_high_cut - voltage),
            ]
        )
