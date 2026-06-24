import pybamm


class BasicDFN(pybamm.lithium_ion.BaseModel):
    """Doyle-Fuller-Newman (DFN) model of a sodium-ion battery, from
    :footcite:t:`Marquis2019`.

    Parameters
    ----------
    name : str, optional
        The name of the model.

    """

    def __init__(self, name="Doyle-Fuller-Newman model"):
        super().__init__(name=name)
        pybamm.citations.register("Marquis2019")
        param = self.param

        # === Variables ===
        Q = pybamm.Variable("Discharge capacity [A.h]")

        # Spatially-varying variables created with domain
        c_e_n = pybamm.Variable(
            "Negative electrolyte concentration [mol.m-3]",
            domain="negative electrode",
        )
        c_e_s = pybamm.Variable(
            "Separator electrolyte concentration [mol.m-3]",
            domain="separator",
        )
        c_e_p = pybamm.Variable(
            "Positive electrolyte concentration [mol.m-3]",
            domain="positive electrode",
        )
        # Concatenations combine variables over several domains into one
        c_e = pybamm.concatenation(c_e_n, c_e_s, c_e_p)

        # Electrolyte potential
        phi_e_n = pybamm.Variable(
            "Negative electrolyte potential [V]",
            domain="negative electrode",
        )
        phi_e_s = pybamm.Variable(
            "Separator electrolyte potential [V]",
            domain="separator",
        )
        phi_e_p = pybamm.Variable(
            "Positive electrolyte potential [V]",
            domain="positive electrode",
        )
        phi_e = pybamm.concatenation(phi_e_n, phi_e_s, phi_e_p)

        # Electrode potential
        phi_s_n = pybamm.Variable(
            "Negative electrode potential [V]", domain="negative electrode"
        )
        phi_s_p = pybamm.Variable(
            "Positive electrode potential [V]",
            domain="positive electrode",
        )
        # Particle concentrations vary in particle and electrode (x) domains
        c_s_n = pybamm.Variable(
            "Negative particle concentration [mol.m-3]",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        c_s_p = pybamm.Variable(
            "Positive particle concentration [mol.m-3]",
            domain="positive particle",
            auxiliary_domains={"secondary": "positive electrode"},
        )

        # Constant temperature
        T = param.T_init

        # === Other set-up ===

        # Current density (time-dependent)
        i_cell = param.current_density_with_time

        # Porosity - PrimaryBroadcast lifts scalars across a domain for multiplication
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
        a_n = 3 * param.n.prim.epsilon_s_av / param.n.prim.R_typ
        a_p = 3 * param.p.prim.epsilon_s_av / param.p.prim.R_typ

        # Interfacial reactions - surf() returns the right-boundary value
        c_s_surf_n = pybamm.surf(c_s_n)
        sto_surf_n = c_s_surf_n / param.n.prim.c_max
        j0_n = param.n.prim.j0(c_e_n, c_s_surf_n, T)
        eta_n = phi_s_n - phi_e_n - param.n.prim.U(sto_surf_n, T)
        Feta_RT_n = param.F * eta_n / (param.R * T)
        j_n = 2 * j0_n * pybamm.sinh(param.n.prim.ne / 2 * Feta_RT_n)

        c_s_surf_p = pybamm.surf(c_s_p)
        sto_surf_p = c_s_surf_p / param.p.prim.c_max
        j0_p = param.p.prim.j0(c_e_p, c_s_surf_p, T)
        eta_p = phi_s_p - phi_e_p - param.p.prim.U(sto_surf_p, T)
        Feta_RT_p = param.F * eta_p / (param.R * T)
        j_s = pybamm.PrimaryBroadcast(0, "separator")
        j_p = 2 * j0_p * pybamm.sinh(param.p.prim.ne / 2 * Feta_RT_p)

        a_j_n = a_n * j_n
        a_j_p = a_p * j_p
        a_j = pybamm.concatenation(a_j_n, j_s, a_j_p)

        # === State of Charge ===
        I = param.current_with_time
        # rhs: dict of ODEs keyed by the differentiated variable
        self.rhs[Q] = I / 3600
        # Initial conditions for ODEs
        self.initial_conditions[Q] = pybamm.Scalar(0)

        # === Particles ===

        # div/gradients become matrix multiplications at discretisation
        N_s_n = -param.n.prim.D(c_s_n, T) * pybamm.grad(c_s_n)
        N_s_p = -param.p.prim.D(c_s_p, T) * pybamm.grad(c_s_p)
        self.rhs[c_s_n] = -pybamm.div(N_s_n)
        self.rhs[c_s_p] = -pybamm.div(N_s_p)
        # Boundary conditions must be provided for equations with spatial derivatives
        self.boundary_conditions[c_s_n] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -j_n / (param.F * pybamm.surf(param.n.prim.D(c_s_n, T))),
                "Neumann",
            ),
        }
        self.boundary_conditions[c_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -j_p / (param.F * pybamm.surf(param.p.prim.D(c_s_p, T))),
                "Neumann",
            ),
        }
        self.initial_conditions[c_s_n] = param.n.prim.c_init
        self.initial_conditions[c_s_p] = param.p.prim.c_init
        # === Current in the solid ===
        sigma_eff_n = param.n.sigma(sto_surf_n, T) * eps_s_n**param.n.b_s
        i_s_n = -sigma_eff_n * pybamm.grad(phi_s_n)
        sigma_eff_p = param.p.sigma(sto_surf_p, T) * eps_s_p**param.p.b_s
        i_s_p = -sigma_eff_p * pybamm.grad(phi_s_p)
        # algebraic: DAE equations scaled by Lx**2 for conditioning
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
        # Algebraic equations need initial guesses for consistent IC calculation
        self.initial_conditions[phi_s_n] = pybamm.Scalar(0)
        self.initial_conditions[phi_s_p] = param.ocv_init

        # === Current in the electrolyte ===
        i_e = (param.kappa_e(c_e, T) * tor) * (
            param.chiRT_over_Fc(c_e, T) * pybamm.grad(c_e) - pybamm.grad(phi_e)
        )
        # Scale by Lx**2 for conditioning
        self.algebraic[phi_e] = param.L_x**2 * (pybamm.div(i_e) - a_j)
        self.boundary_conditions[phi_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[phi_e] = -param.n.prim.U_init

        # === Electrolyte concentration ===
        N_e = -tor * param.D_e(c_e, T) * pybamm.grad(c_e)
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) + (1 - param.t_plus(c_e, T)) * a_j / param.F
        )
        self.boundary_conditions[c_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[c_e] = param.c_e_init

        # === (Some) variables ===
        voltage = pybamm.boundary_value(phi_s_p, "right")
        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )
        # variables: outputs useful for visualising the solution
        self.variables = {
            "Negative particle concentration [mol.m-3]": c_s_n,
            "Negative particle surface concentration [mol.m-3]": c_s_surf_n,
            "Electrolyte concentration [mol.m-3]": c_e,
            "Negative electrolyte concentration [mol.m-3]": c_e_n,
            "Separator electrolyte concentration [mol.m-3]": c_e_s,
            "Positive electrolyte concentration [mol.m-3]": c_e_p,
            "Positive particle concentration [mol.m-3]": c_s_p,
            "Positive particle surface concentration [mol.m-3]": c_s_surf_p,
            "Current [A]": I,
            "Current variable [A]": I,  # for compatibility with pybamm.Experiment
            "Negative electrode potential [V]": phi_s_n,
            "Electrolyte potential [V]": phi_e,
            "Negative electrolyte potential [V]": phi_e_n,
            "Separator electrolyte potential [V]": phi_e_s,
            "Positive electrolyte potential [V]": phi_e_p,
            "Positive electrode potential [V]": phi_s_p,
            "Voltage [V]": voltage,
            "Voltage expression [V]": voltage,
            "Battery voltage [V]": voltage * num_cells,
            "Time [s]": pybamm.t,
            "Discharge capacity [A.h]": Q,
        }
        # Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event("Minimum voltage [V]", voltage - param.voltage_low_cut),
            pybamm.Event("Maximum voltage [V]", param.voltage_high_cut - voltage),
        ]

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues("Chayambuka2022")
