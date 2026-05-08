import pybamm

from .multilayer_3d_thermal_spm import MultiLayer3DThermalSPM


class MultiLayer3DThermalDFN(MultiLayer3DThermalSPM):
    """
    Multilayer Doyle-Fuller-Newman (DFN) model with layer-specific 3D
    thermal domains.

    Extends :class:`MultiLayer3DThermalSPM` by replacing the per-layer SPM
    electrochemistry with a full DFN system: spatially-resolved particle
    concentrations, electrolyte concentration and potential PDEs, and
    solid-phase charge conservation. Each layer is a complete 1D
    electrochemical model coupled to its own 3D temperature field.

    Parameters
    ----------
    num_physical_layers : int, optional
        Total number of physical unit cells in the real stack (default 3,
        minimum 2).
    num_subdivisions : int, optional
        Number of thermal/electrochemical zones (default equal to
        ``num_physical_layers``, minimum 2).
    connection : str, optional
        ``"parallel"`` (default) or ``"series"``.
    mesh_h : float or str, optional
        Target mesh size for 3D FEM thermal meshes (default ``0.1``).
    options : dict, optional
        Model options. ``"cell geometry"`` must be ``"pouch"``.
    name : str, optional
        Model name.

    Notes
    -----
    Compared to the SPM variant, the DFN adds per layer:

    * Spatially-resolved particle concentrations ``c_s(r, x)`` with
      auxiliary (secondary) electrode domains.
    * Electrolyte concentration PDE ``c_e(x, t)`` across the full cell.
    * Electrolyte potential algebraic equation ``phi_e(x)`` enforcing
      charge conservation.
    * Solid-phase potential algebraic equations ``phi_s_n(x)``,
      ``phi_s_p(x)`` enforcing solid charge conservation.
    * Butler-Volmer kinetics solved implicitly (not analytically inverted).

    This makes the system a differential-algebraic equation (DAE) rather
    than a pure ODE.

    All thermal coupling, geometry, mesh, stack scaling, and connectivity
    logic is inherited unchanged from :class:`MultiLayer3DThermalSPM`.
    """

    def __init__(
        self,
        num_physical_layers=None,
        num_subdivisions=None,
        connection="parallel",
        mesh_h=0.1,
        options=None,
        name="Multi-Layer 3D Thermal DFN",
        *,
        num_layers=None,
        layers_per_zone=None,
    ):
        super().__init__(
            num_physical_layers=num_physical_layers,
            num_subdivisions=num_subdivisions,
            connection=connection,
            mesh_h=mesh_h,
            options=options,
            name=name,
            num_layers=num_layers,
            layers_per_zone=layers_per_zone,
        )

    def _build_electrochemistry_layer(self, layer_id):
        """Build full DFN equations for a single layer.

        Returns the standard layer dict expected by the parent class
        connectivity and thermal methods.
        """
        # Prefix for layer-specific variable names
        pfx = f"Layer {layer_id}"

        # --- Particle variables (spatially resolved in x) ---
        c_s_n = pybamm.Variable(
            f"{pfx} Negative particle concentration [mol.m-3]",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        c_s_p = pybamm.Variable(
            f"{pfx} Positive particle concentration [mol.m-3]",
            domain="positive particle",
            auxiliary_domains={"secondary": "positive electrode"},
        )

        # Algebraic variable linked to layer's volume-averaged temperature
        T_av = pybamm.Variable(f"{pfx} average temperature [K]")

        # --- Electrolyte variables ---
        c_e_n = pybamm.Variable(
            f"{pfx} Negative electrolyte concentration [mol.m-3]",
            domain="negative electrode",
        )
        c_e_s = pybamm.Variable(
            f"{pfx} Separator electrolyte concentration [mol.m-3]",
            domain="separator",
        )
        c_e_p = pybamm.Variable(
            f"{pfx} Positive electrolyte concentration [mol.m-3]",
            domain="positive electrode",
        )
        c_e = pybamm.concatenation(c_e_n, c_e_s, c_e_p)

        # --- Potential variables ---
        phi_e_n = pybamm.Variable(
            f"{pfx} Negative electrolyte potential [V]",
            domain="negative electrode",
        )
        phi_e_s = pybamm.Variable(
            f"{pfx} Separator electrolyte potential [V]",
            domain="separator",
        )
        phi_e_p = pybamm.Variable(
            f"{pfx} Positive electrolyte potential [V]",
            domain="positive electrode",
        )
        phi_e = pybamm.concatenation(phi_e_n, phi_e_s, phi_e_p)

        phi_s_n = pybamm.Variable(
            f"{pfx} Negative electrode potential [V]",
            domain="negative electrode",
        )
        phi_s_p = pybamm.Variable(
            f"{pfx} Positive electrode potential [V]",
            domain="positive electrode",
        )

        # --- Layer current ---
        n = self.layers_per_zone
        I_app = self.param.current_with_time
        i_cell_app = self.param.current_density_with_time
        if self.connection == "parallel":
            f_i = pybamm.Variable(f"{pfx} current fraction")
            I_layer = f_i * I_app / n
            i_cell = f_i * i_cell_app / n
        else:  # series
            f_i = None
            I_layer = I_app / n
            i_cell = i_cell_app / n

        # Use constant temperature for transport (coupled via T_av)
        T = T_av

        # --- Electrode parameters ---
        eps_n = pybamm.Parameter("Negative electrode porosity")
        eps_s = pybamm.Parameter("Separator porosity")
        eps_p = pybamm.Parameter("Positive electrode porosity")
        eps_s_n_vol = pybamm.Parameter(
            "Negative electrode active material volume fraction"
        )
        eps_s_p_vol = pybamm.Parameter(
            "Positive electrode active material volume fraction"
        )

        # Broadcast porosities
        eps_n_bc = pybamm.PrimaryBroadcast(eps_n, "negative electrode")
        eps_s_bc = pybamm.PrimaryBroadcast(eps_s, "separator")
        eps_p_bc = pybamm.PrimaryBroadcast(eps_p, "positive electrode")
        eps = pybamm.concatenation(eps_n_bc, eps_s_bc, eps_p_bc)

        # Transport efficiency (Bruggeman)
        tor = pybamm.concatenation(
            eps_n_bc**self.param.n.b_e,
            eps_s_bc**self.param.s.b_e,
            eps_p_bc**self.param.p.b_e,
        )

        a_n = 3 * self.param.n.prim.epsilon_s_av / self.param.n.prim.R_typ
        a_p = 3 * self.param.p.prim.epsilon_s_av / self.param.p.prim.R_typ

        # --- Interfacial reactions (Butler-Volmer) ---
        c_s_surf_n = pybamm.surf(c_s_n)
        sto_surf_n = c_s_surf_n / self.param.n.prim.c_max
        j0_n = self.param.n.prim.j0(c_e_n, c_s_surf_n, T)
        eta_n = phi_s_n - phi_e_n - self.param.n.prim.U(sto_surf_n, T)
        Feta_RT_n = self.param.F * eta_n / (self.param.R * T)
        j_n = 2 * j0_n * pybamm.sinh(self.param.n.prim.ne / 2 * Feta_RT_n)

        c_s_surf_p = pybamm.surf(c_s_p)
        sto_surf_p = c_s_surf_p / self.param.p.prim.c_max
        j0_p = self.param.p.prim.j0(c_e_p, c_s_surf_p, T)
        eta_p = phi_s_p - phi_e_p - self.param.p.prim.U(sto_surf_p, T)
        Feta_RT_p = self.param.F * eta_p / (self.param.R * T)
        j_s = pybamm.PrimaryBroadcast(0, "separator")
        j_p = 2 * j0_p * pybamm.sinh(self.param.p.prim.ne / 2 * Feta_RT_p)

        a_j_n = a_n * j_n
        a_j_p = a_p * j_p
        a_j = pybamm.concatenation(a_j_n, j_s, a_j_p)

        # --- Particle diffusion ---
        N_s_n = -self.param.n.prim.D(c_s_n, T) * pybamm.grad(c_s_n)
        N_s_p = -self.param.p.prim.D(c_s_p, T) * pybamm.grad(c_s_p)
        self.rhs[c_s_n] = -pybamm.div(N_s_n)
        self.rhs[c_s_p] = -pybamm.div(N_s_p)

        self.boundary_conditions[c_s_n] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -j_n / (self.param.F * pybamm.surf(self.param.n.prim.D(c_s_n, T))),
                "Neumann",
            ),
        }
        self.boundary_conditions[c_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -j_p / (self.param.F * pybamm.surf(self.param.p.prim.D(c_s_p, T))),
                "Neumann",
            ),
        }
        self.initial_conditions[c_s_n] = self.param.n.prim.c_init
        self.initial_conditions[c_s_p] = self.param.p.prim.c_init

        # --- Solid-phase charge conservation (algebraic) ---
        sigma_eff_n = self.param.n.sigma(T) * eps_s_n_vol**self.param.n.b_s
        i_s_n = -sigma_eff_n * pybamm.grad(phi_s_n)
        sigma_eff_p = self.param.p.sigma(T) * eps_s_p_vol**self.param.p.b_s
        i_s_p = -sigma_eff_p * pybamm.grad(phi_s_p)

        # multiply by L_x^2 to improve conditioning
        L_x = self.param.L_x
        self.algebraic[phi_s_n] = L_x**2 * (pybamm.div(i_s_n) + a_j_n)
        self.algebraic[phi_s_p] = L_x**2 * (pybamm.div(i_s_p) + a_j_p)

        self.boundary_conditions[phi_s_n] = {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.boundary_conditions[phi_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                i_cell / pybamm.boundary_value(-sigma_eff_p, "right"),
                "Neumann",
            ),
        }
        self.initial_conditions[phi_s_n] = pybamm.Scalar(0)
        self.initial_conditions[phi_s_p] = self.param.ocv_init

        # --- Electrolyte charge conservation (algebraic) ---
        i_e = (self.param.kappa_e(c_e, T) * tor) * (
            self.param.chiRT_over_Fc(c_e, T) * pybamm.grad(c_e) - pybamm.grad(phi_e)
        )
        self.algebraic[phi_e] = L_x**2 * (pybamm.div(i_e) - a_j)
        self.boundary_conditions[phi_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[phi_e] = -self.param.n.prim.U_init

        # --- Electrolyte concentration (PDE) ---
        N_e = -tor * self.param.D_e(c_e, T) * pybamm.grad(c_e)
        t_plus = self.param.t_plus(c_e, T)
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) + (1 - t_plus) * a_j / self.param.F
        )
        self.boundary_conditions[c_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[c_e] = self.param.c_e_init

        # --- Voltage ---
        V_layer = pybamm.boundary_value(phi_s_p, "right")

        # --- Heat generation ---
        L_n = self.param.n.L
        L_p = self.param.p.L
        # Irreversible: overpotential * reaction current
        Q_irr_n = pybamm.x_average(a_j_n * eta_n)
        Q_irr_p = pybamm.x_average(a_j_p * eta_p)
        # Reversible: entropic heating
        dUdT_n = self.param.n.prim.dUdT(sto_surf_n)
        dUdT_p = self.param.p.prim.dUdT(sto_surf_p)
        Q_rev_n = pybamm.x_average(a_n * j_n * T * dUdT_n)
        Q_rev_p = pybamm.x_average(a_p * j_p * T * dUdT_p)
        # Ohmic in solid
        Q_ohm_s_n = pybamm.x_average(
            sigma_eff_n * pybamm.inner(pybamm.grad(phi_s_n), pybamm.grad(phi_s_n))
        )
        Q_ohm_s_p = pybamm.x_average(
            sigma_eff_p * pybamm.inner(pybamm.grad(phi_s_p), pybamm.grad(phi_s_p))
        )
        # Ohmic in electrolyte (simplified x-averaged)
        kappa_eff_avg = self.param.kappa_e(self.param.c_e_init_av, T) * (
            eps_n**self.param.n.b_e
        )
        Q_ohm_e = (
            i_cell**2
            * (
                L_n / (3 * kappa_eff_avg)
                + self.param.s.L
                / (
                    self.param.kappa_e(self.param.c_e_init_av, T)
                    * eps_s**self.param.s.b_e
                )
                + L_p
                / (
                    3
                    * self.param.kappa_e(self.param.c_e_init_av, T)
                    * eps_p**self.param.p.b_e
                )
            )
            / L_x
        )

        Q_vol = (
            (Q_irr_n + Q_rev_n) * L_n / L_x
            + (Q_irr_p + Q_rev_p) * L_p / L_x
            + (Q_ohm_s_n * L_n + Q_ohm_s_p * L_p) / L_x
            + Q_ohm_e
        )

        # --- Stoichiometry events ---
        self.events += [
            pybamm.Event(
                f"{pfx} minimum negative particle surface stoichiometry",
                pybamm.min(sto_surf_n) - 0.01,
            ),
            pybamm.Event(
                f"{pfx} maximum negative particle surface stoichiometry",
                (1 - 0.01) - pybamm.max(sto_surf_n),
            ),
            pybamm.Event(
                f"{pfx} minimum positive particle surface stoichiometry",
                pybamm.min(sto_surf_p) - 0.01,
            ),
            pybamm.Event(
                f"{pfx} maximum positive particle surface stoichiometry",
                (1 - 0.01) - pybamm.max(sto_surf_p),
            ),
        ]

        return {
            "c_s_n": c_s_n,
            "c_s_p": c_s_p,
            "c_s_surf_n": c_s_surf_n,
            "c_s_surf_p": c_s_surf_p,
            "T_av": T_av,
            "voltage": V_layer,
            "current": I_layer,
            "current_fraction": f_i,
            "phi_s_n": phi_s_n,
            "phi_s_p": phi_s_p,
            "phi_e": phi_e,
            "Q_vol": Q_vol,
            "c_e": c_e,
        }

    def _register_variables(self):
        """Extend parent registration with DFN-specific variables."""
        super()._register_variables()
        for i, layer in enumerate(self.layers):
            self.variables[f"Layer {i} electrolyte concentration [mol.m-3]"] = layer[
                "c_e"
            ]
            self.variables[f"Layer {i} electrolyte potential [V]"] = layer["phi_e"]
