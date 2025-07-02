import pybamm


class CustomModel(pybamm.lithium_ion.BaseModel):
    """
    A custom Single Particle Model (SPM) for a lithium-ion battery,
    written in a single file without relying on inherited parameter values.
    """

    def __init__(self, options=None, name="Custom Single Particle Model"):
        # Initialize base model with empty options
        super().__init__({}, name)

        # Register citation
        pybamm.citations.register("Marquis2019")

        # Use symbolic parameters only
        param = pybamm.LithiumIonParameters()
        self.param = param
        self._geometry = self.default_geometry
        ######################
        # Variables
        ######################
        Q = pybamm.Variable("Discharge capacity [A.h]")
        c_s_n = pybamm.Variable(
            "X-averaged negative particle concentration [mol.m-3]",
            domain="negative particle",
        )
        c_s_p = pybamm.Variable(
            "X-averaged positive particle concentration [mol.m-3]",
            domain="positive particle",
        )

        T = param.T_init
        I = param.current_with_time
        i_cell = param.current_density_with_time

        a_n = 3 * param.n.prim.epsilon_s_av / param.n.prim.R_typ
        a_p = 3 * param.p.prim.epsilon_s_av / param.p.prim.R_typ
        j_n = i_cell / (param.n.L * a_n)
        j_p = -i_cell / (param.p.L * a_p)

        ######################
        # Differential equations
        ######################
        self.rhs[Q] = I / 3600
        self.initial_conditions[Q] = pybamm.Scalar(0)

        N_s_n = -param.n.prim.D(c_s_n, T) * pybamm.grad(c_s_n)
        N_s_p = -param.p.prim.D(c_s_p, T) * pybamm.grad(c_s_p)
        self.rhs[c_s_n] = -pybamm.div(N_s_n)
        self.rhs[c_s_p] = -pybamm.div(N_s_p)

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

        self.initial_conditions[c_s_n] = pybamm.x_average(param.n.prim.c_init)
        self.initial_conditions[c_s_p] = pybamm.x_average(param.p.prim.c_init)

        ######################
        # Events: Stoichiometry
        ######################
        c_s_surf_n = pybamm.surf(c_s_n)
        c_s_surf_p = pybamm.surf(c_s_p)
        sto_surf_n = c_s_surf_n / param.n.prim.c_max
        sto_surf_p = c_s_surf_p / param.p.prim.c_max
        self.events += [
            pybamm.Event(
                "Minimum negative particle surface stoichiometry",
                pybamm.min(sto_surf_n) - 0.01,
            ),
            pybamm.Event(
                "Maximum negative particle surface stoichiometry",
                (1 - 0.01) - pybamm.max(sto_surf_n),
            ),
            pybamm.Event(
                "Minimum positive particle surface stoichiometry",
                pybamm.min(sto_surf_p) - 0.01,
            ),
            pybamm.Event(
                "Maximum positive particle surface stoichiometry",
                (1 - 0.01) - pybamm.max(sto_surf_p),
            ),
        ]

        ######################
        # Voltage and potentials
        ######################
        RT_F = param.R * T / param.F
        j0_n = param.n.prim.j0(param.c_e_init_av, c_s_surf_n, T)
        j0_p = param.p.prim.j0(param.c_e_init_av, c_s_surf_p, T)
        eta_n = (2 / param.n.prim.ne) * RT_F * pybamm.arcsinh(j_n / (2 * j0_n))
        eta_p = (2 / param.p.prim.ne) * RT_F * pybamm.arcsinh(j_p / (2 * j0_p))
        phi_s_n = 0
        phi_e = -eta_n - param.n.prim.U(sto_surf_n, T)
        phi_s_p = eta_p + phi_e + param.p.prim.U(sto_surf_p, T)
        V = phi_s_p

        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )

        ######################
        # Output variables
        ######################
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        self.variables = {
            "Time [s]": pybamm.t,
            "Discharge capacity [A.h]": Q,
            "X-averaged negative particle concentration [mol.m-3]": c_s_n,
            "Negative particle surface concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                c_s_surf_n, "negative electrode"
            ),
            "Electrolyte concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                param.c_e_init_av, whole_cell
            ),
            "X-averaged positive particle concentration [mol.m-3]": c_s_p,
            "Positive particle surface concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                c_s_surf_p, "positive electrode"
            ),
            "Current [A]": I,
            "Current variable [A]": I,  # for compatibility
            "Negative electrode potential [V]": pybamm.PrimaryBroadcast(
                phi_s_n, "negative electrode"
            ),
            "Electrolyte potential [V]": pybamm.PrimaryBroadcast(phi_e, whole_cell),
            "Positive electrode potential [V]": pybamm.PrimaryBroadcast(
                phi_s_p, "positive electrode"
            ),
            "Voltage [V]": V,
            "Battery voltage [V]": V * num_cells,
        }

        ######################
        # Voltage cutoff events
        ######################
        self.events += [
            pybamm.Event("Minimum voltage [V]", V - param.voltage_low_cut),
            pybamm.Event("Maximum voltage [V]", param.voltage_high_cut - V),
        ]
