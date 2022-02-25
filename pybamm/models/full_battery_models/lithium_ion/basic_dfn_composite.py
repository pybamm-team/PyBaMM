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

    def __init__(self, name="Doyle-Fuller-Newman model"):
        super().__init__({}, name)
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
            "Negative electrolyte concentration", domain="negative electrode"
        )
        c_e_s = pybamm.Variable(
            "Separator electrolyte concentration", domain="separator"
        )
        c_e_p = pybamm.Variable(
            "Positive electrolyte concentration", domain="positive electrode"
        )
        # Concatenations combine several variables into a single variable, to simplify
        # implementing equations that hold over several domains
        c_e = pybamm.concatenation(c_e_n, c_e_s, c_e_p)

        # Electrolyte potential
        phi_e_n = pybamm.Variable(
            "Negative electrolyte potential", domain="negative electrode"
        )
        phi_e_s = pybamm.Variable("Separator electrolyte potential", domain="separator")
        phi_e_p = pybamm.Variable(
            "Positive electrolyte potential", domain="positive electrode"
        )
        phi_e = pybamm.concatenation(phi_e_n, phi_e_s, phi_e_p)

        # Electrode potential
        phi_s_n = pybamm.Variable(
            "Negative electrode potential", domain="negative electrode"
        )
        phi_s_p = pybamm.Variable(
            "Positive electrode potential", domain="positive electrode"
        )
        # Particle concentrations are variables on the particle domain, but also vary in
        # the x-direction (electrode domain) and so must be provided with auxiliary
        # domains
        c_s_n_p1 = pybamm.Variable(
            "Negative particle concentration of phase 1",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        c_s_n_p2 = pybamm.Variable(
            "Negative particle concentration of phase 2",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        c_s_p = pybamm.Variable(
            "Positive particle concentration",
            domain="positive particle",
            auxiliary_domains={"secondary": "positive electrode"},
        )

        # Constant temperature
        T = param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = param.current_with_time

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
        eps_s_n = pybamm.Parameter("Negative electrode active material volume fraction")
        eps_s_p = pybamm.Parameter("Positive electrode active material volume fraction")

        # Tortuosity
        tor = pybamm.concatenation(
            eps_n ** param.b_e_n, eps_s ** param.b_e_s, eps_p ** param.b_e_p
        )

        # Interfacial reactions
        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        c_s_surf_n_p1 = pybamm.surf(c_s_n_p1)
        j0_n_p1 = (
            param.j0_n(c_e_n, c_s_surf_n_p1, T, "phase 1")
            / param.C_r_n
            * param.a_p1_a_n
        )
        ocp_n_p1 = param.U_n(c_s_surf_n_p1, T, "phase 1")
        j_n_p1 = (
            2 * j0_n_p1 * pybamm.sinh(param.ne_n / 2 * (phi_s_n - phi_e_n - ocp_n_p1))
        )
        c_s_surf_n_p2 = pybamm.surf(c_s_n_p2)
        j0_n_p2 = (
            param.j0_n(c_e_n, c_s_surf_n_p2, T, "phase 2")
            / param.C_r_n
            * param.a_p2_a_n
        )
        ocp_n_p2 = param.U_n(c_s_surf_n_p2, T, "phase 2")
        j_n_p2 = (
            2 * j0_n_p2 * pybamm.sinh(param.ne_n / 2 * (phi_s_n - phi_e_n - ocp_n_p2))
        )
        j_n = j_n_p1 + j_n_p2
        c_s_surf_p = pybamm.surf(c_s_p)
        j0_p = param.gamma_p * param.j0_p(c_e_p, c_s_surf_p, T) / param.C_r_p
        j_s = pybamm.PrimaryBroadcast(0, "separator")
        ocp_p = param.U_p(c_s_surf_p, T)
        j_p = (
            2
            * j0_p
            * pybamm.sinh(
                param.ne_p / 2 * (phi_s_p - phi_e_p - param.U_p(c_s_surf_p, T))
            )
        )
        j = pybamm.concatenation(j_n, j_s, j_p)

        ######################
        # State of Charge
        ######################
        I = param.dimensional_current_with_time
        # The `rhs` dictionary contains differential equations, with the key being the
        # variable in the d/dt
        self.rhs[Q] = I * param.timescale / 3600
        # Initial conditions must be provided for the ODEs
        self.initial_conditions[Q] = pybamm.Scalar(0)

        ######################
        # Particles
        ######################

        # The div and grad operators will be converted to the appropriate matrix
        # multiplication at the discretisation stage
        N_s_n_p1 = -param.D_n(c_s_n_p1, T, "phase 1") * pybamm.grad(c_s_n_p1)
        N_s_n_p2 = -param.D_n(c_s_n_p2, T, "phase 2") * pybamm.grad(c_s_n_p2)
        N_s_p = -param.D_p(c_s_p, T) * pybamm.grad(c_s_p)
        self.rhs[c_s_n_p1] = -(1 / param.C_n_p1) * pybamm.div(N_s_n_p1)
        self.rhs[c_s_n_p2] = -(1 / param.C_n_p2) * pybamm.div(N_s_n_p2)
        self.rhs[c_s_p] = -(1 / param.C_p) * pybamm.div(N_s_p)
        # Boundary conditions must be provided for equations with spatial derivatives
        self.boundary_conditions[c_s_n_p1] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -param.C_n_p1
                * j_n_p1
                / pybamm.maximum(param.a_R_n_p1, 0.0000001)
                / param.gamma_n_p1
                / param.D_n(c_s_surf_n_p1, T, "phase 1"),
                "Neumann",
            ),
        }
        self.boundary_conditions[c_s_n_p2] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -param.C_n_p2
                * j_n_p2
                / pybamm.maximum(param.a_R_n_p2, 0.0000001)
                / param.gamma_n_p2
                / param.D_n(c_s_surf_n_p2, T, "phase 2"),
                "Neumann",
            ),
        }
        self.boundary_conditions[c_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -param.C_p
                * j_p
                / param.a_R_p
                / param.gamma_p
                / param.D_p(c_s_surf_p, T),
                "Neumann",
            ),
        }
        # c_n_init and c_p_init can in general be functions of x
        # Note the broadcasting, for domains
        x_n = pybamm.PrimaryBroadcast(
            pybamm.standard_spatial_vars.x_n, "negative particle"
        )
        self.initial_conditions[c_s_n_p1] = param.c_n_init_comp(x_n, "phase 1")
        self.initial_conditions[c_s_n_p2] = param.c_n_init_comp(x_n, "phase 2")
        self.initial_conditions[c_s_p] = param.c_p_init
        # Events specify points at which a solution should terminate
        tolerance = 0.0000001
        self.events += [
            pybamm.Event(
                "Minimum negative particle surface concentration of phase 1",
                pybamm.min(c_s_surf_n_p1) - tolerance,
            ),
            pybamm.Event(
                "Maximum negative particle surface concentration of phase 1",
                (1 - tolerance) - pybamm.max(c_s_surf_n_p1),
            ),
            pybamm.Event(
                "Minimum negative particle surface concentration of phase 2",
                pybamm.min(c_s_surf_n_p2) - tolerance,
            ),
            pybamm.Event(
                "Maximum negative particle surface concentration of phase 2",
                (1 - tolerance) - pybamm.max(c_s_surf_n_p2),
            ),
            pybamm.Event(
                "Minimum positive particle surface concentration",
                pybamm.min(c_s_surf_p) - tolerance,
            ),
            pybamm.Event(
                "Maximum positive particle surface concentration",
                (1 - tolerance) - pybamm.max(c_s_surf_p),
            ),
        ]
        ######################
        # Current in the solid
        ######################
        sigma_eff_n = param.sigma_n(T) * eps_s_n ** param.b_s_n
        i_s_n = -sigma_eff_n * pybamm.grad(phi_s_n)
        sigma_eff_p = param.sigma_p(T) * eps_s_p ** param.b_s_p
        i_s_p = -sigma_eff_p * pybamm.grad(phi_s_p)
        # The `algebraic` dictionary contains differential equations, with the key being
        # the main scalar variable of interest in the equation
        self.algebraic[phi_s_n] = pybamm.div(i_s_n) + j_n
        self.algebraic[phi_s_p] = pybamm.div(i_s_p) + j_p
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
        i_e = (param.kappa_e(c_e, T) * tor * param.gamma_e / param.C_e) * (
            param.chi(c_e, T) * pybamm.grad(c_e) / c_e - pybamm.grad(phi_e)
        )
        self.algebraic[phi_e] = pybamm.div(i_e) - j
        self.boundary_conditions[phi_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[phi_e] = -param.U_n_init

        ######################
        # Electrolyte concentration
        ######################
        N_e = -tor * param.D_e(c_e, T) * pybamm.grad(c_e)
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) / param.C_e
            + (1 - param.t_plus(c_e, T)) * j / param.gamma_e
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
        pot_scale = param.potential_scale
        U_ref = param.U_p_ref - param.U_n_ref
        voltage_dim = U_ref + voltage * pot_scale
        ocp_n_p1_dim = param.U_n_ref + param.potential_scale * ocp_n_p1
        ocp_av_n_p1_dim = pybamm.x_average(ocp_n_p1_dim)
        ocp_n_p2_dim = param.U_n_ref + param.potential_scale * ocp_n_p2
        ocp_av_n_p2_dim = pybamm.x_average(ocp_n_p2_dim)
        ocp_p_dim = param.U_p_ref + param.potential_scale * ocp_p
        ocp_av_p_dim = pybamm.x_average(ocp_p_dim)
        c_s_rav_n_p1 = pybamm.r_average(c_s_n_p1)
        c_s_rav_n_p1_dim = c_s_rav_n_p1 * param.c_n_p1_max
        c_s_rav_n_p2 = pybamm.r_average(c_s_n_p2)
        c_s_rav_n_p2_dim = c_s_rav_n_p2 * param.c_n_p2_max
        c_s_xrav_n_p1 = pybamm.x_average(c_s_rav_n_p1)
        c_s_xrav_n_p1_dim = c_s_xrav_n_p1 * param.c_n_p1_max
        c_s_xrav_n_p2 = pybamm.x_average(c_s_rav_n_p2)
        c_s_xrav_n_p2_dim = c_s_xrav_n_p2 * param.c_n_p2_max
        c_s_rav_p = pybamm.r_average(c_s_p)
        c_s_xrav_p = pybamm.x_average(c_s_rav_p)
        c_s_xrav_p_dim = c_s_xrav_p * param.c_p_max
        j_n_p1_dim = j_n_p1 * param.j_scale_n / pybamm.maximum(param.a_p1_a_n, 0.000001)
        j_n_p2_dim = j_n_p2 * param.j_scale_n / pybamm.maximum(param.a_p2_a_n, 0.000001)
        j_n_p1_av_dim = pybamm.x_average(j_n_p1_dim)
        j_n_p2_av_dim = pybamm.x_average(j_n_p2_dim)
        j_n_p1_v_dim = j_n_p1 * param.i_typ / param.L_x
        j_n_p2_v_dim = j_n_p2 * param.i_typ / param.L_x
        j_n_p1_v_av_dim = pybamm.x_average(j_n_p1_v_dim)
        j_n_p2_v_av_dim = pybamm.x_average(j_n_p2_v_dim)
        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        self.variables = {
            "Negative particle concentration of phase 1": c_s_n_p1,
            "Negative particle concentration of phase 2": c_s_n_p2,
            "Positive particle concentration": c_s_p,
            "Negative particle concentration": c_s_p,
            "Negative particle surface concentration of phase 1": c_s_surf_n_p1,
            "Negative particle surface concentration of phase 2": c_s_surf_n_p2,
            "Electrolyte concentration": c_e,
            "Positive particle surface concentration": c_s_surf_p,
            "Negative electrode potential [V]": param.potential_scale * phi_s_n,
            "Electrolyte potential [V]": -param.U_n_ref + param.potential_scale * phi_e,
            "Positive electrode potential [V]": param.U_p_ref
            - param.U_n_ref
            + param.potential_scale * phi_s_p,
            "Negative electrolyte potential": phi_e_n,
            "Separator electrolyte potential": phi_e_s,
            "Positive electrolyte potential": phi_e_p,
            "Negative electrolyte concentration": c_e_n,
            "Separator electrolyte concentration": c_e_s,
            "Positive electrolyte concentration": c_e_p,
            "Positive electrode potential": phi_s_p,
            "Negative electrode potential": phi_s_n,
            "Terminal voltage": voltage,
            "Current [A]": I,
            "Discharge capacity [A.h]": Q,
            "Time [s]": pybamm.t * param.timescale,
            "Terminal voltage [V]": voltage_dim,
            "Negative electrode open circuit potential of phase 1 [V]": ocp_n_p1_dim,
            "Negative electrode open circuit potential of phase 2 [V]": ocp_n_p2_dim,
            "X-averaged negative electrode open circuit potential "
            + "of phase 1 [V]": ocp_av_n_p1_dim,
            "X-averaged negative electrode open circuit potential "
            + "of phase 2 [V]": ocp_av_n_p2_dim,
            "Positive electrode open circuit potential [V]": ocp_p_dim,
            "X-averaged positive electrode open circuit potential [V]": ocp_av_p_dim,
            "R-averaged negative particle concentration of phase 1": c_s_rav_n_p1,
            "R-averaged negative particle concentration of phase 2": c_s_rav_n_p2,
            "R-averaged negative particle concentration "
            + "of phase 1 [mol.m-3]": c_s_rav_n_p1_dim,
            "R-averaged negative particle concentration "
            + "of phase 2 [mol.m-3]": c_s_rav_n_p2_dim,
            "Averaged negative electrode concentration of phase 1": c_s_xrav_n_p1,
            "Averaged negative electrode concentration of phase 2": c_s_xrav_n_p2,
            "Averaged negative electrode concentration "
            + "of phase 1 [mol.m-3]": c_s_xrav_n_p1_dim,
            "Averaged negative electrode concentration "
            + "of phase 2 [mol.m-3]": c_s_xrav_n_p2_dim,
            "Negative electrode concentration of phase 1": c_s_n_p1,
            "Negative electrode concentration of phase 2": c_s_n_p2,
            "Negative electrode concentration of phase 1 [mol.m-3]": c_s_n_p1
            * param.c_n_p1_max,
            "Negative electrode concentration of phase 2 [mol.m-3]": c_s_n_p2
            * param.c_n_p2_max,
            "Averaged positive electrode concentration": c_s_xrav_p,
            "Averaged positive electrode concentration [mol.m-3]": c_s_xrav_p_dim,
            "Negative electrode interfacial current density "
            + "of phase 1 [A.m-2]": j_n_p1_dim,
            "Negative electrode interfacial current density "
            + "of phase 2 [A.m-2]": j_n_p2_dim,
            "X-averaged negative electrode interfacial current density "
            + "of phase 1 [A.m-2]": j_n_p1_av_dim,
            "X-averaged negative electrode interfacial current density "
            + "of phase 2 [A.m-2]": j_n_p2_av_dim,
            "Negative electrode interfacial current density "
            + "of phase 1 per volume [A.m-3]": j_n_p1_v_dim,
            "Negative electrode interfacial current density "
            + "of phase 2 per volume [A.m-3]": j_n_p2_v_dim,
            "X-averaged negative electrode interfacial current density "
            + "of phase 1 per volume [A.m-3]": j_n_p1_v_av_dim,
            "X-averaged negative electrode interfacial current density "
            + "of phase 2 per volume [A.m-3]": j_n_p2_v_av_dim,
        }
        self.events += [
            pybamm.Event("Minimum voltage", voltage - param.voltage_low_cut),
            pybamm.Event("Maximum voltage", voltage - param.voltage_high_cut),
        ]

    def new_empty_copy(self):
        return pybamm.BaseModel.new_empty_copy(self)
