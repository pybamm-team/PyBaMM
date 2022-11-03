#
# Basic Doyle-Fuller-Newman (DFN) Model
#
import pybamm


class BasicDFN(pybamm.lithium_ion.BaseModel):
    """Doyle-Fuller-Newman (DFN) model of a lithium-ion battery, from [2]_.

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
    .. [2] SG Marquis, V Sulzer, R Timms, CP Please and SJ Chapman. “An asymptotic
           derivation of a single particle model with electrolyte”. Journal of The
           Electrochemical Society, 166(15):A3693–A3706, 2019

    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(self, name="Doyle-Fuller-Newman model"):
        super().__init__({"timescale": 1}, name)
        self._length_scales = {
            "negative electrode": pybamm.Scalar(1),
            "separator": pybamm.Scalar(1),
            "positive electrode": pybamm.Scalar(1),
        }
        pybamm.citations.register("Marquis2019")
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
            scale=1 + 0 * param.c_e_typ,
        )
        c_e_s = pybamm.Variable(
            "Separator electrolyte concentration [mol.m-3]",
            domain="separator",
            scale=1 + 0 * param.c_e_typ,
        )
        c_e_p = pybamm.Variable(
            "Positive electrolyte concentration [mol.m-3]",
            domain="positive electrode",
            scale=1 + 0 * param.c_e_typ,
        )
        # Concatenations combine several variables into a single variable, to simplify
        # implementing equations that hold over several domains
        c_e = pybamm.concatenation(c_e_n, c_e_s, c_e_p)

        # Electrolyte potential
        phi_e_n = pybamm.Variable(
            "Negative electrolyte potential [V]", domain="negative electrode"
        )
        phi_e_s = pybamm.Variable(
            "Separator electrolyte potential [V]", domain="separator"
        )
        phi_e_p = pybamm.Variable(
            "Positive electrolyte potential [V]", domain="positive electrode"
        )
        phi_e = pybamm.concatenation(phi_e_n, phi_e_s, phi_e_p)

        # Electrode potential
        phi_s_n = pybamm.Variable(
            "Negative electrode potential [V]", domain="negative electrode"
        )
        phi_s_p = pybamm.Variable(
            "Positive electrode potential [V]", domain="positive electrode"
        )
        # Particle concentrations are variables on the particle domain, but also vary in
        # the x-direction (electrode domain) and so must be provided with auxiliary
        # domains
        c_s_n = pybamm.Variable(
            "Negative particle concentration",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
            scale=1 + 0 * param.n.prim.c_max,
        )
        c_s_p = pybamm.Variable(
            "Positive particle concentration",
            domain="positive particle",
            auxiliary_domains={"secondary": "positive electrode"},
            scale=param.p.prim.c_max,
        )
        c_s_p_nondim = c_s_p
        c_s_p_dim = c_s_p * param.p.prim.c_max
        c_scale = param.p.prim.c_max
        # c_s_p_dim = c_s_p
        # c_s_p_nondim = c_s_p / param.p.prim.c_max

        # Constant temperature
        T = param.T_init_dim

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = param.dimensional_current_density_with_time

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

        # transport_efficiency
        tor = pybamm.concatenation(
            eps_n**param.n.b_e, eps_s**param.s.b_e, eps_p**param.p.b_e
        )
        a_n = param.n.prim.a_typ
        a_p = param.p.prim.a_typ

        # Interfacial reactions
        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        c_s_surf_n = pybamm.surf(c_s_n)
        j0_n = param.n.prim.j0_dimensional(c_e_n, c_s_surf_n, T)
        eta_n = (
            phi_s_n
            - phi_e_n
            - param.n.prim.U_dimensional(c_s_surf_n / param.n.prim.c_max, T)
        )
        Feta_RT_n = param.F * eta_n / (param.R * T)
        j_n = 2 * j0_n * pybamm.sinh(param.n.prim.ne / 2 * Feta_RT_n)
        # j_n = pybamm.PrimaryBroadcast(i_cell / (a_n * param.n.L), "negative electrode")
        c_s_surf_p_dim = pybamm.surf(c_s_p_dim)
        c_s_surf_p_nondim = pybamm.surf(c_s_p_nondim)
        j0_p = param.p.prim.j0_dimensional(c_e_p, c_s_surf_p_dim, T)
        eta_p = phi_s_p - phi_e_p - param.p.prim.U_dimensional(c_s_surf_p_nondim, T)
        Feta_RT_p = param.F * eta_p / (param.R * T)
        j_s = pybamm.PrimaryBroadcast(0, "separator")
        j_p = 2 * j0_p * pybamm.sinh(param.p.prim.ne / 2 * Feta_RT_p)
        # j_p = pybamm.PrimaryBroadcast(-i_cell / (a_p * param.p.L), "positive electrode")
        j = pybamm.concatenation(j_n, j_s, j_p)

        a_j_n = a_n * j_n
        a_j_p = a_p * j_p
        a_j = pybamm.concatenation(a_j_n, j_s, a_j_p)

        ######################
        # State of Charge
        ######################
        I = param.dimensional_current_with_time
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
        N_s_n = -param.n.prim.D_dimensional(c_s_n, T) * pybamm.grad(c_s_n)
        N_s_p = -param.p.prim.D_dimensional(c_s_p_dim, T) * pybamm.grad(c_s_p) * c_scale
        self.rhs[c_s_n] = -pybamm.div(N_s_n)
        self.rhs[c_s_p] = -pybamm.div(N_s_p)
        # Boundary conditions must be provided for equations with spatial derivatives
        self.boundary_conditions[c_s_n] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -j_n / (param.F * param.n.prim.D_dimensional(c_s_surf_n, T)),
                "Neumann",
            ),
        }
        self.boundary_conditions[c_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -j_p / (param.F * param.p.prim.D_dimensional(c_s_surf_p_dim, T)),
                "Neumann",
            ),
        }
        self.initial_conditions[c_s_n] = param.n.prim.c_init_dimensional
        self.initial_conditions[c_s_p] = param.p.prim.c_init_dimensional
        ######################
        # Current in the solid
        ######################
        sigma_eff_n = param.n.sigma_dimensional(T) * eps_s_n**param.n.b_s
        i_s_n = -sigma_eff_n * pybamm.grad(phi_s_n)
        sigma_eff_p = param.p.sigma_dimensional(T) * eps_s_p**param.p.b_s
        i_s_p = -sigma_eff_p * pybamm.grad(phi_s_p)
        # The `algebraic` dictionary contains differential equations, with the key being
        # the main scalar variable of interest in the equation
        self.algebraic[phi_s_n] = pybamm.div(i_s_n) + a_j_n
        self.algebraic[phi_s_p] = pybamm.div(i_s_p) + a_j_p
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
        self.initial_conditions[phi_s_p] = param.ocv_init_dim

        ######################
        # Current in the electrolyte
        ######################
        i_e = (param.kappa_e_dimensional(c_e, T) * tor) * (
            param.chiRT_over_Fc_dimensional(c_e, T) * pybamm.grad(c_e)
            - pybamm.grad(phi_e)
        )
        self.algebraic[phi_e] = pybamm.div(i_e) - a_j
        self.boundary_conditions[phi_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[phi_e] = -param.n.prim.U_init_dim

        ######################
        # Electrolyte concentration
        ######################
        N_e = -tor * param.D_e_dimensional(c_e, T) * pybamm.grad(c_e)
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) + (1 - param.t_plus_dimensional(c_e, T)) * a_j / param.F
        )
        self.boundary_conditions[c_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[c_e] = param.c_e_init_dimensional

        ######################
        # (Some) variables
        ######################
        voltage = pybamm.boundary_value(phi_s_p, "right")
        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        self.variables = {
            "Negative particle surface concentration [mol.m-3]": c_s_surf_n,
            "Negative particle surface concentration": c_s_surf_n / param.n.prim.c_max,
            "Electrolyte concentration [mol.m-3]": c_e,
            "Positive particle surface concentration [mol.m-3]": c_s_surf_p_dim,
            "Positive particle surface concentration": c_s_surf_p_nondim,
            "Current [A]": I,
            "Negative electrode potential [V]": phi_s_n,
            "Electrolyte potential [V]": phi_e,
            "Positive electrode potential [V]": phi_s_p,
            "Terminal voltage [V]": voltage,
            "Time [s]": pybamm.t,
        }
        # Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event(
                "Minimum voltage", voltage - param.voltage_low_cut_dimensional
            ),
            pybamm.Event(
                "Maximum voltage", param.voltage_high_cut_dimensional - voltage
            ),
        ]

    @property
    def default_geometry(self):
        param = self.param
        L_n = param.n.L
        L_s = param.s.L
        L_p = param.p.L
        L_n_L_s = L_n + L_s
        geometry = {
            "negative electrode": {"x_n": {"min": 0, "max": L_n}},
            "separator": {"x_s": {"min": L_n, "max": L_n_L_s}},
            "positive electrode": {"x_p": {"min": L_n_L_s, "max": param.L_x}},
            "negative particle": {"r_n": {"min": 0, "max": param.n.prim.R_typ}},
            "positive particle": {"r_p": {"min": 0, "max": param.p.prim.R_typ}},
            "current collector": {"z": {"position": 1}},
        }
        return geometry


pybamm.set_logging_level("INFO")
model = BasicDFN()
var_pts = {"x_n": 10, "x_s": 10, "x_p": 10, "r_n": 10, "r_p": 10}
# sim = pybamm.Simulation(model, solver=pybamm.CasadiSolver("fast", root_method="lm"))
sim = pybamm.Simulation(
    model, solver=pybamm.IDAKLUSolver(root_method="lm"), var_pts=var_pts
)
sol = sim.solve([0, 3600])
sol = sim.solve([0, 3600])

# model = pybamm.lithium_ion.DFN()
# sim = pybamm.Simulation(
#     model, solver=pybamm.IDAKLUSolver(root_method="lm"), var_pts=var_pts
# )
# # sim = pybamm.Simulation(model, solver=pybamm.CasadiSolver("fast", root_method="lm"))
# sol2 = sim.solve([0, 3600])
# sol2 = sim.solve([0, 3600])

pybamm.dynamic_plot(
    [sol, sol],
    [
        "Negative particle surface concentration",
        "Positive particle surface concentration",
        "Electrolyte concentration [mol.m-3]",
        "Negative electrode potential [V]",
        "Positive electrode potential [V]",
        "Electrolyte potential [V]",
        "Terminal voltage [V]",
    ],
    labels=["dimensional", "dimensionless"],
)
