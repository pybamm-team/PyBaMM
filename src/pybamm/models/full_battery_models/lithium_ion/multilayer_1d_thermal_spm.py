#
# Multi-layer 1D thermal SPM model.
#
# Each layer has a 1D temperature field T(x) spanning the electrode sandwich
# (negative electrode | separator | positive electrode) with full finite-volume
# resolution, plus scalar current-collector nodes at the boundaries. Adjacent
# layers are coupled via thermal contact resistance at the CC interface.
#
import pybamm

from .base_lithium_ion_model import BaseModel


class MultiLayer1DThermalSPM(BaseModel):
    """Multi-layer pouch cell with 1D through-thickness thermal resolution.

    Each layer (zone) solves:
      - SPM electrochemistry (particle diffusion + voltage)
      - 1D heat equation through the electrode sandwich:
        ``rho*cp * dT/dt = div(lambda * grad(T)) + Q``
      - Scalar current-collector temperature nodes coupled to adjacent layers
        via thermal contact resistance.

    The through-stack temperature profile T(x) is the concatenation of all
    layers' 1D profiles, showing intra-cell gradients within each layer and
    inter-layer jumps from contact resistance.

    Parameters
    ----------
    num_physical_layers : int
        Total number of physical unit cells in the stack.
    num_subdivisions : int, optional
        Number of computational zones (thermal layers). Each zone lumps
        ``num_physical_layers / num_subdivisions`` adjacent cells.
        Defaults to ``num_physical_layers`` (no coarsening).
    connection : str
        Electrical connection between layers: ``"parallel"`` or ``"series"``.
    """

    CONTACT_RESISTANCE_PARAM = "Inter-layer thermal contact resistance [K.m2.W-1]"
    DEFAULT_CONTACT_RESISTANCE = 1e-4

    def __init__(
        self,
        num_physical_layers=None,
        num_subdivisions=None,
        connection="parallel",
        options=None,
        name="Multi-Layer 1D Thermal SPM",
    ):
        # Resolve layer/zone counts
        num_physical_layers = (
            3 if num_physical_layers is None else int(num_physical_layers)
        )
        num_subdivisions = (
            num_physical_layers if num_subdivisions is None else int(num_subdivisions)
        )

        if num_physical_layers < 2:
            raise ValueError("num_physical_layers must be an integer >= 2")
        if num_subdivisions < 2:
            raise ValueError("num_subdivisions must be an integer >= 2")
        if num_physical_layers % num_subdivisions != 0:
            raise ValueError(
                "num_physical_layers must be divisible by num_subdivisions "
                f"(got {num_physical_layers} and {num_subdivisions})"
            )
        if connection not in ("parallel", "series"):
            raise ValueError(
                f"connection must be 'parallel' or 'series', got '{connection}'"
            )

        options = dict(options) if options else {}
        options.setdefault("cell geometry", "pouch")

        super().__init__(options, name)
        pybamm.citations.register("Marquis2019")

        self.num_layers = num_subdivisions
        self.num_subdivisions = num_subdivisions
        self.num_physical_layers = num_physical_layers
        self.layers_per_zone = num_physical_layers // num_subdivisions
        self.connection = connection

        # Build the model
        self.layers = [
            self._build_electrochemistry_layer(i) for i in range(self.num_layers)
        ]

        if self.connection == "parallel":
            self._connect_parallel()
        else:
            self._connect_series()

        self._build_thermal()
        self._register_variables()

        V_term = self.variables["Voltage [V]"]
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
    # Per-layer SPM electrochemistry (reused from 3D multilayer pattern)
    # ------------------------------------------------------------------ #
    def _build_electrochemistry_layer(self, layer_id):
        """Build the SPM equations for a single layer."""
        c_s_n = pybamm.Variable(
            f"Layer {layer_id} X-averaged negative particle concentration [mol.m-3]",
            domain="negative particle",
        )
        c_s_p = pybamm.Variable(
            f"Layer {layer_id} X-averaged positive particle concentration [mol.m-3]",
            domain="positive particle",
        )

        # Algebraic variable for layer's x-averaged temperature
        T_av = pybamm.Variable(f"Layer {layer_id} average temperature [K]")

        # Layer current
        n = self.layers_per_zone
        I_app = self.param.current_with_time
        i_cell_app = self.param.current_density_with_time
        if self.connection == "parallel":
            f_i = pybamm.Variable(f"Layer {layer_id} current fraction")
            I_layer = f_i * I_app / n
            i_cell = f_i * i_cell_app / n
        else:
            f_i = None
            I_layer = I_app / n
            i_cell = i_cell_app / n

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
                -j_n / (self.param.F * pybamm.surf(self.param.n.prim.D(c_s_n, T_av))),
                "Neumann",
            ),
        }
        self.boundary_conditions[c_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -j_p / (self.param.F * pybamm.surf(self.param.p.prim.D(c_s_p, T_av))),
                "Neumann",
            ),
        }

        self.initial_conditions[c_s_n] = pybamm.x_average(self.param.n.prim.c_init)
        self.initial_conditions[c_s_p] = pybamm.x_average(self.param.p.prim.c_init)

        c_s_surf_n = pybamm.surf(c_s_n)
        c_s_surf_p = pybamm.surf(c_s_p)
        sto_surf_n = c_s_surf_n / self.param.n.prim.c_max
        sto_surf_p = c_s_surf_p / self.param.p.prim.c_max

        # Potentials and overpotentials
        RT_F = self.param.R * T_av / self.param.F
        j0_n = self.param.n.prim.j0(self.param.c_e_init_av, c_s_surf_n, T_av)
        j0_p = self.param.p.prim.j0(self.param.c_e_init_av, c_s_surf_p, T_av)
        eta_n = (2 / self.param.n.prim.ne) * RT_F * pybamm.arcsinh(j_n / (2 * j0_n))
        eta_p = (2 / self.param.p.prim.ne) * RT_F * pybamm.arcsinh(j_p / (2 * j0_p))
        phi_s_n = pybamm.Scalar(0)
        phi_e = -eta_n - self.param.n.prim.U(sto_surf_n, T_av)
        phi_s_p = eta_p + phi_e + self.param.p.prim.U(sto_surf_p, T_av)
        V_layer = phi_s_p

        # Heat generation (volumetric average)
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

        # Stoichiometry events
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

        for i in range(1, self.num_layers):
            self.algebraic[fractions[i]] = self.layers[i]["voltage"] - V_ref

        self.algebraic[fractions[0]] = sum(fractions) - 1

        for i in range(self.num_layers):
            self.initial_conditions[fractions[i]] = pybamm.Scalar(1.0 / self.num_layers)

        self._terminal_voltage = V_ref

    def _connect_series(self):
        V_total = sum(layer["voltage"] for layer in self.layers)
        self._terminal_voltage = V_total

    # ------------------------------------------------------------------ #
    # 1D thermal model
    # ------------------------------------------------------------------ #
    def _build_thermal(self):
        """Build the 1D thermal PDEs for all layers.

        Each layer gets:
          - T_i(x): temperature on ["negative electrode", "separator",
            "positive electrode"] solved with div(λ·grad(T)) + Q

        The current collectors carry no separate state: they are taken as
        infinitely conductive, so each layer's CC temperature is just the
        boundary value of T_i at that face.

        Coupling:
          - T_av_i (used by SPM) == x_average(T_i) (algebraic constraint)
          - Adjacent layers: conduction across the inter-layer thermal
            contact resistance (:attr:`CONTACT_RESISTANCE_PARAM`)
          - External: convective cooling on the two outermost faces
        """
        T_init = self.param.T_init
        T_amb = self.param.T_amb(pybamm.Scalar(0), pybamm.Scalar(0), pybamm.t)

        # Cooling coefficients for the left/right x-faces of the stack
        h_left = pybamm.Parameter("Left face heat transfer coefficient [W.m-2.K-1]")
        h_right = pybamm.Parameter("Right face heat transfer coefficient [W.m-2.K-1]")

        # Store thermal variables for registration
        self._thermal_cell_temps = []  # 1D concatenated T_i per layer

        for i in range(self.num_layers):
            pfx = f"Layer {i}"

            # --- 1D temperature variables on electrode domains ---
            T_n_i = pybamm.Variable(
                f"{pfx} negative electrode temperature [K]",
                domain="negative electrode",
            )
            T_s_i = pybamm.Variable(
                f"{pfx} separator temperature [K]",
                domain="separator",
            )
            T_p_i = pybamm.Variable(
                f"{pfx} positive electrode temperature [K]",
                domain="positive electrode",
            )
            T_i = pybamm.concatenation(T_n_i, T_s_i, T_p_i)

            # --- Thermal conductivity and heat capacity (spatially resolved) ---
            lambda_n = self.param.n.lambda_(T_n_i)
            lambda_s = self.param.s.lambda_(T_s_i)
            lambda_p = self.param.p.lambda_(T_p_i)
            lambda_ = pybamm.concatenation(lambda_n, lambda_s, lambda_p)

            rho_c_p_n = self.param.n.rho_c_p(T_n_i)
            rho_c_p_s = self.param.s.rho_c_p(T_s_i)
            rho_c_p_p = self.param.p.rho_c_p(T_p_i)
            rho_c_p = pybamm.concatenation(rho_c_p_n, rho_c_p_s, rho_c_p_p)

            # --- Heat source ---
            # SPM Q_vol is thickness-weighted average: (Q_n*L_n + Q_p*L_p)/L_x
            # Broadcast uniformly across the entire sandwich
            Q_vol = self.layers[i]["Q_vol"]
            Q = pybamm.concatenation(
                pybamm.PrimaryBroadcast(Q_vol, "negative electrode"),
                pybamm.PrimaryBroadcast(Q_vol, "separator"),
                pybamm.PrimaryBroadcast(Q_vol, "positive electrode"),
            )

            # --- 1D heat equation ---
            self.rhs[T_i] = (pybamm.div(lambda_ * pybamm.grad(T_i)) + Q) / rho_c_p

            # Initial conditions for T_i
            self.initial_conditions[T_i] = T_init

            self._thermal_cell_temps.append(T_i)

            # Algebraic constraint: T_av == x_average(T_i)
            T_av_i = self.layers[i]["T_av"]
            self.algebraic[T_av_i] = T_av_i - pybamm.x_average(T_i)
            self.initial_conditions[T_av_i] = T_init

        # --- Boundary conditions ---
        # External faces: convective cooling. Internal faces: conduction into
        # the adjacent layer across the inter-layer thermal contact resistance
        # R_th, written as a matched pair of Neumann conditions so the flux
        # leaving layer i at its right face is the flux entering layer i+1 at
        # its left face. This mirrors the coupling in MultiLayer3DThermalSPM.
        #
        # PyBaMM's finite-volume "Neumann" value is dT/dx itself, not the
        # outward normal derivative, so a +x-directed flux q = -lambda*dT/dx
        # gives dT/dx = -q / lambda on BOTH sides of an interface.
        R_th = pybamm.Parameter(self.CONTACT_RESISTANCE_PARAM)

        # lambda at a layer's outer faces: negative electrode on the left,
        # positive electrode on the right. Evaluated at T_init, as the
        # convective conditions below are.
        lambda_left_face = self.param.n.lambda_(T_init)
        lambda_right_face = self.param.p.lambda_(T_init)

        # Heat flux from layer i into layer i+1 [W.m-2], positive in +x.
        q_interface = [
            (
                pybamm.boundary_value(self._thermal_cell_temps[i], "right")
                - pybamm.boundary_value(self._thermal_cell_temps[i + 1], "left")
            )
            / R_th
            for i in range(self.num_layers - 1)
        ]

        for i in range(self.num_layers):
            T_i = self._thermal_cell_temps[i]

            # Left BC
            if i == 0:
                # External cooling on the left face of the stack. The outward
                # normal is -x, so lambda * dT/dx|_left = h * (T - T_amb).
                T_left_bv = pybamm.boundary_value(T_i, "left")
                left_bc = (
                    h_left * (T_left_bv - T_amb) / lambda_left_face,
                    "Neumann",
                )
            else:
                # Heat arriving from layer i-1 across the contact resistance.
                left_bc = (-q_interface[i - 1] / lambda_left_face, "Neumann")

            # Right BC
            if i == self.num_layers - 1:
                # External cooling on the right face of the stack. The outward
                # normal is +x, so -lambda * dT/dx|_right = h * (T - T_amb).
                T_right_bv = pybamm.boundary_value(T_i, "right")
                right_bc = (
                    -h_right * (T_right_bv - T_amb) / lambda_right_face,
                    "Neumann",
                )
            else:
                # Heat leaving toward layer i+1 across the contact resistance.
                right_bc = (-q_interface[i] / lambda_right_face, "Neumann")

            self.boundary_conditions[T_i] = {
                "left": left_bc,
                "right": right_bc,
            }

        # CC temperatures are just the boundary values (for output only)
        self._thermal_cc_n_temps = [
            pybamm.boundary_value(T_i, "left") for T_i in self._thermal_cell_temps
        ]
        self._thermal_cc_p_temps = [
            pybamm.boundary_value(T_i, "right") for T_i in self._thermal_cell_temps
        ]

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

        for i, layer in enumerate(self.layers):
            T_i = self._thermal_cell_temps[i]
            T_cn_i = self._thermal_cc_n_temps[i]
            T_cp_i = self._thermal_cc_p_temps[i]

            self.variables[
                f"Layer {i} X-averaged negative particle concentration [mol.m-3]"
            ] = layer["c_s_n"]
            self.variables[
                f"Layer {i} X-averaged positive particle concentration [mol.m-3]"
            ] = layer["c_s_p"]
            self.variables[f"Layer {i} cell temperature [K]"] = T_i
            self.variables[f"Layer {i} negative CC temperature [K]"] = T_cn_i
            self.variables[f"Layer {i} positive CC temperature [K]"] = T_cp_i
            self.variables[f"Layer {i} average temperature [K]"] = layer["T_av"]
            self.variables[f"Layer {i} heat generation [W.m-3]"] = layer["Q_vol"]
            self.variables[f"Layer {i} voltage [V]"] = layer["voltage"]
            self.variables[f"Layer {i} current [A]"] = layer["current"]
            self.variables[f"Layer {i} per-unit-cell current [A]"] = layer["current"]
            if layer["current_fraction"] is not None:
                self.variables[f"Layer {i} current fraction"] = layer[
                    "current_fraction"
                ]

        # Global thermal diagnostics
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
    # Stack scaling helper
    # ------------------------------------------------------------------ #
    def apply_stack_scaling(self, parameter_values, verbose=True):
        """Scale parameters for the multilayer stack.

        Multiplies ``"Nominal cell capacity [A.h]"`` by
        ``num_physical_layers`` so a C-rate maps to the correct total
        applied current for the entire stack.
        """
        Q_single = parameter_values["Nominal cell capacity [A.h]"]
        Q_stack = Q_single * self.num_physical_layers
        parameter_values["Nominal cell capacity [A.h]"] = Q_stack
        if self.CONTACT_RESISTANCE_PARAM not in parameter_values.keys():
            parameter_values.update(
                {self.CONTACT_RESISTANCE_PARAM: self.DEFAULT_CONTACT_RESISTANCE}
            )
        if verbose:
            print(
                f"Physical stack: {self.num_layers} zones x "
                f"{self.layers_per_zone} layers/zone = "
                f"{self.num_physical_layers} unit cells"
            )
            print(
                f"Nominal cell capacity scaled: {Q_single:.3f} Ah -> {Q_stack:.3f} Ah"
            )
        return parameter_values

    @property
    def default_parameter_values(self):
        pv = super().default_parameter_values
        pv.update({self.CONTACT_RESISTANCE_PARAM: self.DEFAULT_CONTACT_RESISTANCE})
        return pv
