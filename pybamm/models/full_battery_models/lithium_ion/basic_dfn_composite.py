#
# Basic Doyle-Fuller-Newman (DFN) Model
#
import pybamm
from .base_lithium_ion_model import BaseModel


class BasicDFNComposite(BaseModel):
    """Doyle-Fuller-Newman (DFN) model of a lithium-ion battery with composite particles
        of graphite and silicon.

    This class differs from the :class:`pybamm.lithium_ion.DFN` model class in that it
    shows the whole model in a single class. This comes at the cost of flexibility in
    comparing different physical effects, and in general the main DFN class should be
    used instead.

    Parameters
    ----------
    name : str, optional
        The name of the model.

    References
    ----------
    ..  W. Ai, N. Kirkaldy, Y. Jiang, G. Offer, H. Wang, B. Wu (2022).
        A composite electrode model for lithium-ion battery with a
        silicon/graphite negative electrode. Journal of Power Sources. 527, 231142.

    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(self, name="Composite graphite/silicon Doyle-Fuller-Newman model"):
        options = {"particle phases": ("2", "1")}
        super().__init__(options, name)
        pybamm.citations.register("Ai2022")
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
            scale=param.c_e_init_av,
        )
        c_e_s = pybamm.Variable(
            "Separator electrolyte concentration [mol.m-3]",
            domain="separator",
            scale=param.c_e_init_av,
        )
        c_e_p = pybamm.Variable(
            "Positive electrolyte concentration [mol.m-3]",
            domain="positive electrode",
            scale=param.c_e_init_av,
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
        # Particle concentrations are variables on the particle domain, but also vary in
        # the x-direction (electrode domain) and so must be provided with auxiliary
        # domains
        c_s_n_p1 = pybamm.Variable(
            "Negative primary particle concentration [mol.m-3]",
            domain="negative primary particle",
            auxiliary_domains={"secondary": "negative electrode"},
            scale=param.n.prim.c_max,
        )
        c_s_n_p2 = pybamm.Variable(
            "Negative secondary particle concentration [mol.m-3]",
            domain="negative secondary particle",
            auxiliary_domains={"secondary": "negative electrode"},
            scale=param.n.sec.c_max,
        )
        c_s_p = pybamm.Variable(
            "Positive particle concentration [mol.m-3]",
            domain="positive particle",
            auxiliary_domains={"secondary": "positive electrode"},
            scale=param.p.prim.c_max,
        )

        # Constant temperature
        T = param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = param.current_density_with_time

        # Porosity
        # Primary broadcasts are used to broadcast scalar quantities across a domain
        # into a vector of the right shape, for multiplying with other vectors
        eps_n = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Negative electrode porosity"), "negative electrode"
        )
        eps_s = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Separator porosity"), "separator"
        )
        eps_p = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Positive electrode porosity"), "positive electrode"
        )
        eps = pybamm.concatenation(eps_n, eps_s, eps_p)

        # Active material volume fraction (eps + eps_s + eps_inactive = 1)
        eps_s_n = pybamm.Parameter(
            "Primary: Negative electrode active material volume fraction"
        ) + pybamm.Parameter(
            "Secondary: Negative electrode active material volume fraction"
        )
        eps_s_p = pybamm.Parameter("Positive electrode active material volume fraction")

        # Tortuosity
        tor = pybamm.concatenation(
            eps_n**param.n.b_e, eps_s**param.s.b_e, eps_p**param.p.b_e
        )

        # Open-circuit potentials
        c_s_surf_n_p1 = pybamm.surf(c_s_n_p1)
        sto_surf_n_p1 = c_s_surf_n_p1 / param.n.prim.c_max
        ocp_n_p1 = param.n.prim.U(sto_surf_n_p1, T)

        c_s_surf_n_p2 = pybamm.surf(c_s_n_p2)
        sto_surf_n_p2 = c_s_surf_n_p2 / param.n.sec.c_max
        k = 100
        m_lith = pybamm.sigmoid(i_cell, 0, k)  # for lithation (current < 0)
        m_delith = 1 - m_lith  # for delithiation (current > 0)
        U_lith = self.param.n.sec.U(sto_surf_n_p2, T, "lithiation")
        U_delith = self.param.n.sec.U(sto_surf_n_p2, T, "delithiation")
        ocp_n_p2 = m_lith * U_lith + m_delith * U_delith

        c_s_surf_p = pybamm.surf(c_s_p)
        sto_surf_p = c_s_surf_p / param.p.prim.c_max
        ocp_p = param.p.prim.U(sto_surf_p, T)

        # Interfacial reactions
        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        F_RT = param.F / (param.R * T)
        j0_n_p1 = param.n.prim.j0(c_e_n, c_s_surf_n_p1, T)
        j_n_p1 = (
            2
            * j0_n_p1
            * pybamm.sinh(param.n.prim.ne / 2 * F_RT * (phi_s_n - phi_e_n - ocp_n_p1))
        )
        j0_n_p2 = param.n.sec.j0(c_e_n, c_s_surf_n_p2, T)
        j_n_p2 = (
            2
            * j0_n_p2
            * pybamm.sinh(param.n.sec.ne / 2 * F_RT * (phi_s_n - phi_e_n - ocp_n_p2))
        )
        j0_p = param.p.prim.j0(c_e_p, c_s_surf_p, T)
        a_j_s = pybamm.PrimaryBroadcast(0, "separator")
        j_p = (
            2
            * j0_p
            * pybamm.sinh(param.p.prim.ne / 2 * F_RT * (phi_s_p - phi_e_p - ocp_p))
        )

        # Volumetric
        a_n_p1 = 3 * param.n.prim.epsilon_s_av / param.n.prim.R_typ
        a_n_p2 = 3 * param.n.sec.epsilon_s_av / param.n.sec.R_typ
        a_p = 3 * param.p.prim.epsilon_s_av / param.p.prim.R_typ
        a_j_n_p1 = a_n_p1 * j_n_p1
        a_j_n_p2 = a_n_p2 * j_n_p2
        a_j_n = a_j_n_p1 + a_j_n_p2
        a_j_p = a_p * j_p
        a_j = pybamm.concatenation(a_j_n, a_j_s, a_j_p)

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

        # The div and grad operators will be converted to the appropriate matrix
        # multiplication at the discretisation stage
        N_s_n_p1 = -param.n.prim.D(c_s_n_p1, T) * pybamm.grad(c_s_n_p1)
        N_s_n_p2 = -param.n.sec.D(c_s_n_p2, T) * pybamm.grad(c_s_n_p2)
        N_s_p = -param.p.prim.D(c_s_p, T) * pybamm.grad(c_s_p)
        self.rhs[c_s_n_p1] = -pybamm.div(N_s_n_p1)
        self.rhs[c_s_n_p2] = -pybamm.div(N_s_n_p2)
        self.rhs[c_s_p] = -pybamm.div(N_s_p)
        # Boundary conditions must be provided for equations with spatial derivatives
        self.boundary_conditions[c_s_n_p1] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (-j_n_p1 / param.F / param.n.prim.D(c_s_surf_n_p1, T), "Neumann"),
        }
        self.boundary_conditions[c_s_n_p2] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (-j_n_p2 / param.F / param.n.sec.D(c_s_surf_n_p2, T), "Neumann"),
        }
        self.boundary_conditions[c_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (-j_p / param.F / param.p.prim.D(c_s_surf_p, T), "Neumann"),
        }
        self.initial_conditions[c_s_n_p1] = param.n.prim.c_init
        self.initial_conditions[c_s_n_p2] = param.n.sec.c_init
        self.initial_conditions[c_s_p] = param.p.prim.c_init
        # Events specify points at which a solution should terminate
        tolerance = 0.0000001
        self.events += [
            pybamm.Event(
                "Minimum negative particle surface concentration of phase 1",
                pybamm.min(sto_surf_n_p1) - tolerance,
            ),
            pybamm.Event(
                "Maximum negative particle surface concentration of phase 1",
                (1 - tolerance) - pybamm.max(sto_surf_n_p1),
            ),
            pybamm.Event(
                "Minimum negative particle surface concentration of phase 2",
                pybamm.min(sto_surf_n_p2) - tolerance,
            ),
            pybamm.Event(
                "Maximum negative particle surface concentration of phase 2",
                (1 - tolerance) - pybamm.max(sto_surf_n_p2),
            ),
            pybamm.Event(
                "Minimum positive particle surface concentration",
                pybamm.min(sto_surf_p) - tolerance,
            ),
            pybamm.Event(
                "Maximum positive particle surface concentration",
                (1 - tolerance) - pybamm.max(sto_surf_p),
            ),
        ]
        ######################
        # Current in the solid
        ######################
        sigma_eff_n = param.n.sigma(T) * eps_s_n**param.n.b_s
        i_s_n = -sigma_eff_n * pybamm.grad(phi_s_n)
        sigma_eff_p = param.p.sigma(T) * eps_s_p**param.p.b_s
        i_s_p = -sigma_eff_p * pybamm.grad(phi_s_p)
        # The `algebraic` dictionary contains differential equations, with the key being
        # the main scalar variable of interest in the equation
        # multiply by Lx**2 to improve conditioning
        self.algebraic[phi_s_n] = param.L_x**2 * (pybamm.div(i_s_n) + a_j_n)
        self.algebraic[phi_s_p] = param.L_x**2 * (pybamm.div(i_s_p) + a_j_p)
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
        # We evaluate c_n_init at x=0 and c_p_init at x=1 (this is just an initial
        # guess so actual value is not too important)
        self.initial_conditions[phi_s_n] = pybamm.Scalar(0)
        self.initial_conditions[phi_s_p] = param.ocv_init

        ######################
        # Current in the electrolyte
        ######################
        i_e = (param.kappa_e(c_e, T) * tor) * (
            param.chiRT_over_Fc(c_e, T) * pybamm.grad(c_e) - pybamm.grad(phi_e)
        )
        # multiply by Lx**2 to improve conditioning
        self.algebraic[phi_e] = param.L_x**2 * (pybamm.div(i_e) - a_j)
        self.boundary_conditions[phi_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[phi_e] = -param.n.prim.U_init

        ######################
        # Electrolyte concentration
        ######################
        N_e = -tor * param.D_e(c_e, T) * pybamm.grad(c_e)
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) + (1 - param.t_plus(c_e, T)) * a_j / param.F
        )
        self.boundary_conditions[c_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[c_e] = param.c_e_init

        ######################
        # (Some) variables
        ######################
        voltage = pybamm.boundary_value(phi_s_p, "right")
        c_s_rav_n_p1 = pybamm.r_average(c_s_n_p1)
        c_s_rav_n_p2 = pybamm.r_average(c_s_n_p2)
        c_s_xrav_n_p1 = pybamm.x_average(c_s_rav_n_p1)
        c_s_xrav_n_p2 = pybamm.x_average(c_s_rav_n_p2)
        c_s_rav_p = pybamm.r_average(c_s_p)
        c_s_xrav_p = pybamm.x_average(c_s_rav_p)
        j_n_p1_av = pybamm.x_average(j_n_p1)
        j_n_p2_av = pybamm.x_average(j_n_p2)
        ocp_av_n_p1 = pybamm.x_average(ocp_n_p1)
        ocp_av_n_p2 = pybamm.x_average(ocp_n_p2)
        ocp_av_p = pybamm.x_average(ocp_p)
        a_j_n_p1_av = pybamm.x_average(a_j_n_p1)
        a_j_n_p2_av = pybamm.x_average(a_j_n_p2)
        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        self.variables = {
            "Negative primary particle concentration [mol.m-3]": c_s_n_p1,
            "Negative secondary particle concentration [mol.m-3]": c_s_n_p2,
            "Negative electrode potential [V]": phi_s_n,
            "Electrolyte potential [V]": phi_e,
            "Positive electrode potential [V]": phi_s_p,
            "Current [A]": I,
            "Discharge capacity [A.h]": Q,
            "Time [s]": pybamm.t,
            "Voltage [V]": voltage,
            "Negative electrode primary open-circuit potential [V]": ocp_n_p1,
            "Negative electrode secondary open-circuit potential [V]": ocp_n_p2,
            "X-averaged negative electrode primary open-circuit potential "
            "[V]": ocp_av_n_p1,
            "X-averaged negative electrode secondary open-circuit potential "
            "[V]": ocp_av_n_p2,
            "Positive electrode open-circuit potential [V]": ocp_p,
            "X-averaged positive electrode open-circuit potential [V]": ocp_av_p,
            "R-averaged negative primary particle concentration "
            "[mol.m-3]": c_s_rav_n_p1,
            "R-averaged negative secondary particle concentration "
            "[mol.m-3]": c_s_rav_n_p2,
            "Average negative primary particle concentration "
            "[mol.m-3]": c_s_xrav_n_p1,
            "Average negative secondary particle concentration "
            "[mol.m-3]": c_s_xrav_n_p2,
            "Average positive particle concentration [mol.m-3]": c_s_xrav_p,
            "Negative electrode primary interfacial current density [A.m-2]": j_n_p1,
            "Negative electrode secondary interfacial current density [A.m-2]": j_n_p2,
            "X-averaged negative electrode primary interfacial current density "
            "[A.m-2]": j_n_p1_av,
            "X-averaged negative electrode secondary interfacial current density "
            "[A.m-2]": j_n_p2_av,
            "Negative electrode primary volumetric interfacial current density "
            "[A.m-3]": a_j_n_p1,
            "Negative electrode secondary volumetric interfacial current density "
            "[A.m-3]": a_j_n_p2,
            "X-averaged negative electrode primary volumetric "
            "interfacial current density [A.m-3]": a_j_n_p1_av,
            "X-averaged negative electrode secondary volumetric "
            "interfacial current density [A.m-3]": a_j_n_p2_av,
        }
        self.events += [
            pybamm.Event("Minimum voltage [V]", voltage - param.voltage_low_cut),
            pybamm.Event("Maximum voltage [V]", param.voltage_high_cut - voltage),
        ]
