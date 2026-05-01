import pybamm

from .base_lithium_ion_model import BaseModel


class MultiLayer3DThermalSPM(BaseModel):
    """
    Multilayer Single Particle Model (SPM) with layer-specific 3D thermal
    domains.

    Each electrochemical layer has its own 3D temperature field ``T_i(x, y, z)``
    on a dedicated mesh ``"cell layer i"``, enabling thermal stratification
    through the stack thickness. Adjacent thermal layers are coupled at their
    ``x`` interfaces via perfect-contact Dirichlet boundary conditions, and the
    outer faces see convective cooling. Each layer's electrochemistry is
    two-way coupled to its own volume-averaged temperature.

    Supported cell geometry is ``"pouch"``. Cylindrical geometry is not
    supported by this model because layers do not stack in the radial
    direction.

    Parameters
    ----------
    num_layers : int, optional
        Number of electrochemical layers in the stack (default 3, minimum 2).
    connection : str, optional
        Electrical connection between layers: ``"parallel"`` (default) or
        ``"series"``.
    thermal_contact_resistance : float, optional
        Thermal contact resistance ``R_th`` [K.m^2/W] between adjacent
        layers, used in the Neumann flux interface coupling. Smaller values
        approximate perfect thermal contact (default ``1e-4``).
    mesh_h : float or str, optional
        Target characteristic length for the per-layer 3D FEM mesh, passed
        through to ``ScikitFemGenerator3D(h=...)``. Smaller values produce
        finer meshes and more visualisation points per layer at the cost of
        a slower solve. Default ``0.1``.
    options : dict, optional
        Model options dictionary. ``"cell geometry"`` must be ``"pouch"``.
    name : str, optional
        Model name.
    """

    def __init__(
        self,
        num_layers=3,
        connection="parallel",
        thermal_contact_resistance=1e-4,
        mesh_h=0.1,
        options=None,
        name="Multi-Layer 3D Thermal SPM",
    ):
        if num_layers is None or int(num_layers) < 2:
            raise ValueError("num_layers must be an integer >= 2")
        if connection not in ("parallel", "series"):
            raise ValueError(
                f"connection must be 'parallel' or 'series', got '{connection}'"
            )
        if thermal_contact_resistance <= 0:
            raise ValueError("thermal_contact_resistance must be positive")
        if float(mesh_h) <= 0:
            raise ValueError("mesh_h must be positive")

        # This model requires a pouch geometry; inject it into options so
        # downstream calls (geometry, parameters) stay consistent.
        options = dict(options) if options else {}
        options.setdefault("cell geometry", "pouch")

        super().__init__(options, name)
        pybamm.citations.register("Marquis2019")

        self.num_layers = int(num_layers)
        self.connection = connection
        self.thermal_contact_resistance = float(thermal_contact_resistance)
        # ScikitFemGenerator3D accepts either a float or a string for h; we
        # normalise to a string for stable downstream hashing/equality.
        self.mesh_h = str(mesh_h)

        geom_type = self.options.get("cell geometry", "pouch")
        if geom_type != "pouch":
            raise NotImplementedError(
                "MultiLayer3DThermalSPM currently only supports "
                "cell geometry 'pouch'."
            )

        # Build the model in one go, matching the pattern used by
        # Basic3DThermalSPM.
        self._layer_spatial_vars = {}  # layer_id -> [x_i, y_i, z_i]
        self._create_thermal_variables()
        self.layers = [self._build_spm_layer(i) for i in range(self.num_layers)]

        if self.connection == "parallel":
            self._connect_parallel()
        else:
            self._connect_series()

        for i in range(self.num_layers):
            self._build_layer_thermal(i)

        self._set_thermal_interfaces()
        self._set_external_cooling_bcs()
        self._register_variables()

        V_term = self.variables["Voltage [V]"]
        # For series connections the terminal voltage is the sum of layer
        # voltages, so cut-offs scale with num_layers.
        v_scale = self.num_layers if self.connection == "series" else 1
        self.events += [
            pybamm.Event(
                "Minimum voltage [V]",
                V_term - v_scale * self.param.voltage_low_cut,
            ),
            pybamm.Event(
                "Maximum voltage [V]",
                v_scale * self.param.voltage_high_cut - V_term,
            ),
        ]

    # ------------------------------------------------------------------ #
    # Variable construction
    # ------------------------------------------------------------------ #
    def _layer_domain(self, layer_id):
        return f"cell layer {layer_id}"

    def _create_thermal_variables(self):
        self.thermal_variables = []
        for i in range(self.num_layers):
            T_i = pybamm.Variable(
                f"Layer {i} temperature [K]", domain=self._layer_domain(i)
            )
            self.thermal_variables.append(T_i)

    # ------------------------------------------------------------------ #
    # Per-layer SPM
    # ------------------------------------------------------------------ #
    def _build_spm_layer(self, layer_id):
        """Build the SPM equations for a single layer."""
        c_s_n = pybamm.Variable(
            f"Layer {layer_id} X-averaged negative particle concentration "
            "[mol.m-3]",
            domain="negative particle",
        )
        c_s_p = pybamm.Variable(
            f"Layer {layer_id} X-averaged positive particle concentration "
            "[mol.m-3]",
            domain="positive particle",
        )

        # Algebraic variable linked to layer's volume-averaged temperature
        T_av = pybamm.Variable(f"Layer {layer_id} average temperature [K]")

        # Layer current
        I_app = self.param.current_with_time
        i_cell_app = self.param.current_density_with_time
        if self.connection == "parallel":
            f_i = pybamm.Variable(f"Layer {layer_id} current fraction")
            I_layer = f_i * I_app
            i_cell = f_i * i_cell_app
        else:  # series
            f_i = None
            I_layer = I_app
            i_cell = i_cell_app

        # Interfacial reactions
        a_n = 3 * self.param.n.prim.epsilon_s_av / self.param.n.prim.R_typ
        a_p = 3 * self.param.p.prim.epsilon_s_av / self.param.p.prim.R_typ
        j_n = i_cell / (self.param.n.L * a_n)
        j_p = -i_cell / (self.param.p.L * a_p)

        # Particle diffusion
        N_s_n = -self.param.n.prim.D(c_s_n, T_av) * pybamm.grad(c_s_n)
        N_s_p = -self.param.p.prim.D(c_s_p, T_av) * pybamm.grad(c_s_p)
        self.rhs[c_s_n] = -pybamm.div(N_s_n)
        self.rhs[c_s_p] = -pybamm.div(N_s_p)

        self.boundary_conditions[c_s_n] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -j_n
                / (self.param.F * pybamm.surf(self.param.n.prim.D(c_s_n, T_av))),
                "Neumann",
            ),
        }
        self.boundary_conditions[c_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -j_p
                / (self.param.F * pybamm.surf(self.param.p.prim.D(c_s_p, T_av))),
                "Neumann",
            ),
        }

        self.initial_conditions[c_s_n] = pybamm.x_average(self.param.n.prim.c_init)
        self.initial_conditions[c_s_p] = pybamm.x_average(self.param.p.prim.c_init)

        c_s_surf_n = pybamm.surf(c_s_n)
        c_s_surf_p = pybamm.surf(c_s_p)
        sto_surf_n = c_s_surf_n / self.param.n.prim.c_max
        sto_surf_p = c_s_surf_p / self.param.p.prim.c_max

        # Potentials and overpotentials (temperature dependent)
        RT_F = self.param.R * T_av / self.param.F
        j0_n = self.param.n.prim.j0(self.param.c_e_init_av, c_s_surf_n, T_av)
        j0_p = self.param.p.prim.j0(self.param.c_e_init_av, c_s_surf_p, T_av)
        eta_n = (2 / self.param.n.prim.ne) * RT_F * pybamm.arcsinh(
            j_n / (2 * j0_n)
        )
        eta_p = (2 / self.param.p.prim.ne) * RT_F * pybamm.arcsinh(
            j_p / (2 * j0_p)
        )
        phi_s_n = pybamm.Scalar(0)
        phi_e = -eta_n - self.param.n.prim.U(sto_surf_n, T_av)
        phi_s_p = eta_p + phi_e + self.param.p.prim.U(sto_surf_p, T_av)
        V_layer = phi_s_p

        # Heat generation (volumetric average over one-layer thickness)
        dUdT_n = self.param.n.prim.dUdT(sto_surf_n)
        dUdT_p = self.param.p.prim.dUdT(sto_surf_p)
        Q_rev_n = a_n * j_n * T_av * dUdT_n
        Q_rev_p = a_p * j_p * T_av * dUdT_p
        Q_irr_n = a_n * j_n * eta_n
        Q_irr_p = a_p * j_p * eta_p
        Q_total_n = Q_rev_n + Q_irr_n
        Q_total_p = Q_rev_p + Q_irr_p

        L_n = self.param.n.L
        L_p = self.param.p.L
        L_x = self.param.L_x
        Q_vol = (Q_total_n * L_n + Q_total_p * L_p) / L_x

        # Stoichiometry events (per layer)
        self.events += [
            pybamm.Event(
                f"Layer {layer_id} minimum negative particle surface stoichiometry",
                pybamm.min(sto_surf_n) - 0.01,
            ),
            pybamm.Event(
                f"Layer {layer_id} maximum negative particle surface stoichiometry",
                (1 - 0.01) - pybamm.max(sto_surf_n),
            ),
            pybamm.Event(
                f"Layer {layer_id} minimum positive particle surface stoichiometry",
                pybamm.min(sto_surf_p) - 0.01,
            ),
            pybamm.Event(
                f"Layer {layer_id} maximum positive particle surface stoichiometry",
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
        }

    # ------------------------------------------------------------------ #
    # Electrical connections
    # ------------------------------------------------------------------ #
    def _connect_parallel(self):
        V_ref = self.layers[0]["voltage"]
        fractions = [layer["current_fraction"] for layer in self.layers]

        # Voltage-equality algebraic constraints for layers 1..n-1
        for i in range(1, self.num_layers):
            self.algebraic[fractions[i]] = self.layers[i]["voltage"] - V_ref

        # Current-fraction sum constraint on layer 0 fraction
        self.algebraic[fractions[0]] = sum(fractions) - 1

        for i in range(self.num_layers):
            self.initial_conditions[fractions[i]] = pybamm.Scalar(
                1.0 / self.num_layers
            )

        self._terminal_voltage = V_ref

    def _connect_series(self):
        V_total = sum(layer["voltage"] for layer in self.layers)
        self._terminal_voltage = V_total

    # ------------------------------------------------------------------ #
    # Per-layer 3D heat equation
    # ------------------------------------------------------------------ #
    def _build_layer_thermal(self, layer_id):
        domain = self._layer_domain(layer_id)
        T_i = self.thermal_variables[layer_id]
        Q_vol = self.layers[layer_id]["Q_vol"]

        x_i = pybamm.SpatialVariable("x", domain=domain)
        y_i = pybamm.SpatialVariable("y", domain=domain)
        z_i = pybamm.SpatialVariable("z", domain=domain)
        self._layer_spatial_vars[layer_id] = (x_i, y_i, z_i)
        integration_vars = [x_i, y_i, z_i]

        volume = pybamm.Integral(
            pybamm.PrimaryBroadcast(1.0, domain), integration_vars
        )
        T_av_integral = pybamm.Integral(T_i, integration_vars) / volume

        # Algebraic link: layer's electrochemical T_av == spatial average of T_i
        T_av_var = self.layers[layer_id]["T_av"]
        self.algebraic[T_av_var] = T_av_var - T_av_integral
        self.initial_conditions[T_av_var] = self.param.T_init

        # Broadcast heat source to the layer's 3D mesh
        Q_source = pybamm.PrimaryBroadcast(Q_vol, domain)
        Q_source = pybamm.source(Q_source, T_i)

        rho_c_p_eff = self.param.rho_c_p_eff(T_i)
        lambda_eff = self.param.lambda_eff(T_i)

        term1 = lambda_eff * pybamm.laplacian(T_i)
        term2 = pybamm.inner(pybamm.grad(lambda_eff), pybamm.grad(T_i))

        self.rhs[T_i] = (term1 + term2 + Q_source) / rho_c_p_eff
        self.initial_conditions[T_i] = pybamm.PrimaryBroadcast(
            self.param.T_init, domain
        )

    # ------------------------------------------------------------------ #
    # Boundary conditions
    # ------------------------------------------------------------------ #
    def _set_thermal_interfaces(self):
        """Thermal coupling between adjacent layers via contact resistance.

        Each internal interface uses matching Neumann fluxes on both sides,
        with the flux driven by the temperature difference across the
        interface and the thermal contact resistance ``R_th`` (K.m^2/W).

        Using a small ``R_th`` (default ``1e-4``) approximates perfect
        thermal contact while keeping the FEM problem well posed — a
        pure Dirichlet-both-sides coupling on independent meshes tends to
        over-constrain the system.
        """
        R_th = self.thermal_contact_resistance
        for i in range(self.num_layers - 1):
            T_left = self.thermal_variables[i]
            T_right = self.thermal_variables[i + 1]

            T_left_bv = pybamm.boundary_value(T_left, "x_max")
            T_right_bv = pybamm.boundary_value(T_right, "x_min")

            lambda_left_bv = pybamm.boundary_value(
                self.param.lambda_eff(T_left), "x_max"
            )
            lambda_right_bv = pybamm.boundary_value(
                self.param.lambda_eff(T_right), "x_min"
            )

            # Heat flux from left to right across the interface [W/m^2].
            q_interface = (T_left_bv - T_right_bv) / R_th

            if T_left not in self.boundary_conditions:
                self.boundary_conditions[T_left] = {}
            if T_right not in self.boundary_conditions:
                self.boundary_conditions[T_right] = {}

            # Neumann BC written in the same form as the convective cooling
            # BCs (q / lambda). Heat leaves layer i at x_max.
            self.boundary_conditions[T_left]["x_max"] = (
                -q_interface / lambda_left_bv,
                "Neumann",
            )
            # Heat enters layer i+1 at x_min (opposite sign).
            self.boundary_conditions[T_right]["x_min"] = (
                q_interface / lambda_right_bv,
                "Neumann",
            )

    def _set_external_cooling_bcs(self):
        """Convective cooling on externally exposed faces only."""
        face_h_params = {
            "x_min": self.param.h_edge_x_min,
            "x_max": self.param.h_edge_x_max,
            "y_min": self.param.h_edge_y_min,
            "y_max": self.param.h_edge_y_max,
            "z_min": self.param.h_edge_z_min,
            "z_max": self.param.h_edge_z_max,
        }

        for i in range(self.num_layers):
            T_i = self.thermal_variables[i]
            _, y_i, z_i = self._layer_spatial_vars[i]
            T_amb = self.param.T_amb(y_i, z_i, pybamm.t)

            if T_i not in self.boundary_conditions:
                self.boundary_conditions[T_i] = {}

            # Which x-faces are external for this layer?
            external_faces = ["y_min", "y_max", "z_min", "z_max"]
            if i == 0:
                external_faces.append("x_min")
            if i == self.num_layers - 1:
                external_faces.append("x_max")

            for face in external_faces:
                # Do not overwrite an interface BC already set
                if face in self.boundary_conditions[T_i]:
                    continue
                h_coeff = face_h_params[face]
                T_boundary = pybamm.boundary_value(T_i, face)
                lambda_eff_boundary = pybamm.boundary_value(
                    self.param.lambda_eff(T_i), face
                )
                q_face = -h_coeff * (T_boundary - T_amb)
                self.boundary_conditions[T_i][face] = (
                    q_face / lambda_eff_boundary,
                    "Neumann",
                )

    # ------------------------------------------------------------------ #
    # Output variables
    # ------------------------------------------------------------------ #
    def _register_variables(self):
        I = self.param.current_with_time
        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )
        V = self._terminal_voltage

        self.variables = {
            "Time [s]": pybamm.t,
            "Current [A]": I,
            "Current variable [A]": I,
            "Voltage [V]": V,
            "Terminal voltage [V]": V,
            "Battery voltage [V]": V * num_cells,
        }

        # Per-layer output
        for i, layer in enumerate(self.layers):
            T_i = self.thermal_variables[i]
            self.variables[
                f"Layer {i} X-averaged negative particle concentration [mol.m-3]"
            ] = layer["c_s_n"]
            self.variables[
                f"Layer {i} X-averaged positive particle concentration [mol.m-3]"
            ] = layer["c_s_p"]
            self.variables[f"Layer {i} temperature [K]"] = T_i
            self.variables[f"Layer {i} average temperature [K]"] = layer["T_av"]
            self.variables[f"Layer {i} heat generation [W.m-3]"] = layer["Q_vol"]
            self.variables[f"Layer {i} voltage [V]"] = layer["voltage"]
            self.variables[f"Layer {i} current [A]"] = layer["current"]
            if layer["current_fraction"] is not None:
                self.variables[f"Layer {i} current fraction"] = layer[
                    "current_fraction"
                ]

        # Global thermal diagnostics on scalar per-layer averages
        T_avs = [layer["T_av"] for layer in self.layers]
        T_stack_av = sum(T_avs) / self.num_layers
        T_max = T_avs[0]
        T_min = T_avs[0]
        for T_av in T_avs[1:]:
            T_max = pybamm.maximum(T_max, T_av)
            T_min = pybamm.minimum(T_min, T_av)
        self.variables["Stack-averaged temperature [K]"] = T_stack_av
        self.variables["Maximum layer-averaged temperature [K]"] = T_max
        self.variables["Minimum layer-averaged temperature [K]"] = T_min
        self.variables["Temperature spread [K]"] = T_max - T_min

    # ------------------------------------------------------------------ #
    # Geometry / mesh / spatial method overrides
    # ------------------------------------------------------------------ #
    @property
    def default_geometry(self):
        geometry = pybamm.battery_geometry(options=self.options)

        geo = pybamm.GeometricParameters(self.options)
        L_x = geo.L_x
        L_y = geo.L_y
        L_z = geo.L_z

        for i in range(self.num_layers):
            geometry[self._layer_domain(i)] = {
                "x": {"min": i * L_x, "max": (i + 1) * L_x},
                "y": {"min": 0, "max": L_y},
                "z": {"min": 0, "max": L_z},
            }

        return geometry

    @property
    def default_submesh_types(self):
        submeshes = super().default_submesh_types
        for i in range(self.num_layers):
            submeshes[self._layer_domain(i)] = pybamm.ScikitFemGenerator3D(
                geom_type="pouch", h=self.mesh_h
            )
        return submeshes

    @property
    def default_spatial_methods(self):
        methods = super().default_spatial_methods
        for i in range(self.num_layers):
            methods[self._layer_domain(i)] = pybamm.ScikitFiniteElement3D()
        return methods

    @property
    def default_var_pts(self):
        var_pts = super().default_var_pts
        # 3D mesh generator uses geometric bounds directly; no var_pts required,
        # but keep placeholders so users can override per-axis resolution.
        var_pts.setdefault("x", None)
        var_pts.setdefault("y", None)
        var_pts.setdefault("z", None)
        return var_pts
