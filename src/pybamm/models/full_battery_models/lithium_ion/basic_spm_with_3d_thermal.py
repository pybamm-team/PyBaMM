import pybamm

from .base_lithium_ion_model import BaseModel


class Basic3DThermalSPM(BaseModel):
    """Single Particle Model (SPM) model of a lithium-ion battery, from
    :footcite:t:`Marquis2019`.

    This class differs from the :class:`pybamm.lithium_ion.SPM` model class in that it
    shows the whole model in a single class. This comes at the cost of flexibility in
    combining different physical effects, and in general the main SPM class should be
    used instead.

    The model is based on the SPM, but with a separate cell domain for the temperature
    variable. This allows for a more detailed treatment of the thermal effects in the
    cell, as the temperature can vary independently of the other variables in the
    model.

    Parameters
    ----------
    name : str, optional
        The name of the model.
    """

    def __init__(self, options=None, name="SPM with Separate Cell Domain"):
        super().__init__(options, name)
        pybamm.citations.register("Marquis2019")
        # `param` is a class containing all the relevant parameters and functions for
        # this model. These are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = pybamm.Variable("Discharge capacity [A.h]")
        # Variables that vary spatially are created with a domain
        c_s_n = pybamm.Variable(
            "X-averaged negative particle concentration [mol.m-3]",
            domain="negative particle",
        )
        c_s_p = pybamm.Variable(
            "X-averaged positive particle concentration [mol.m-3]",
            domain="positive particle",
        )

        T = pybamm.Variable("Cell temperature [K]", domain="cell")

        if self.options.get("cell geometry") == "pouch":
            self.x_cell = pybamm.SpatialVariable("x", domain="cell")
            self.y_cell = pybamm.SpatialVariable("y", domain="cell")
            self.z_cell = pybamm.SpatialVariable("z", domain="cell")
            integration_vars = [self.x_cell, self.y_cell, self.z_cell]
        elif self.options.get("cell geometry") == "cylindrical":
            self.r_cell = pybamm.SpatialVariable("r_macro", domain="cell")
            self.z_cell = pybamm.SpatialVariable("z", domain="cell")
            integration_vars = [self.r_cell, self.z_cell]
        else:
            raise ValueError(
                f"Geometry type '{self.options.get('cell geometry')}' is not supported. "
                "Supported types are 'pouch' and 'cylindrical'."
            )  # pragma: no cover

        volume = pybamm.Integral(pybamm.PrimaryBroadcast(1.0, "cell"), integration_vars)
        T_av = pybamm.Integral(T, integration_vars) / volume

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = self.param.current_density_with_time
        a_n = 3 * self.param.n.prim.epsilon_s_av / self.param.n.prim.R_typ
        a_p = 3 * self.param.p.prim.epsilon_s_av / self.param.p.prim.R_typ
        j_n = i_cell / (self.param.n.L * a_n)
        j_p = -i_cell / (self.param.p.L * a_p)

        ######################
        # State of Charge
        ######################
        I = self.param.current_with_time
        # The `rhs` dictionary contains differential equations, with the key being the
        # variable in the d/dt
        self.rhs[Q] = I / 3600
        # Initial conditions must be provided for the ODEs
        self.initial_conditions[Q] = pybamm.Scalar(0)

        ######################
        # Particles
        ######################

        N_s_n = -self.param.n.prim.D(c_s_n, T_av) * pybamm.grad(c_s_n)
        N_s_p = -self.param.p.prim.D(c_s_p, T_av) * pybamm.grad(c_s_p)
        self.rhs[c_s_n] = -pybamm.div(N_s_n)
        self.rhs[c_s_p] = -pybamm.div(N_s_p)

        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        c_s_surf_n = pybamm.surf(c_s_n)
        c_s_surf_p = pybamm.surf(c_s_p)

        # Boundary conditions must be provided for equations with spatial derivatives
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

        # c_n_init and c_p_init are functions of r and x, but for the SPM we
        # take the x-averaged value since there is no x-dependence in the particles
        self.initial_conditions[c_s_n] = pybamm.x_average(self.param.n.prim.c_init)
        self.initial_conditions[c_s_p] = pybamm.x_average(self.param.p.prim.c_init)
        # Events specify points at which a solution should terminate
        sto_surf_n = c_s_surf_n / self.param.n.prim.c_max
        sto_surf_p = c_s_surf_p / self.param.p.prim.c_max

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

        # Note that the SPM does not have any algebraic equations, so the `algebraic`
        # dictionary remains empty

        ######################
        # (Some) variables
        ######################
        # Interfacial reactions
        RT_F = self.param.R * T_av / self.param.F
        j0_n = self.param.n.prim.j0(self.param.c_e_init_av, c_s_surf_n, T_av)
        j0_p = self.param.p.prim.j0(self.param.c_e_init_av, c_s_surf_p, T_av)
        eta_n = (2 / self.param.n.prim.ne) * RT_F * pybamm.arcsinh(j_n / (2 * j0_n))
        eta_p = (2 / self.param.p.prim.ne) * RT_F * pybamm.arcsinh(j_p / (2 * j0_p))
        phi_s_n = 0
        phi_e = -eta_n - self.param.n.prim.U(sto_surf_n, T_av)
        phi_s_p = eta_p + phi_e + self.param.p.prim.U(sto_surf_p, T_av)
        V = phi_s_p
        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )

        # Compute heat source

        # Reversible (entropic) heating
        dUdT_n = self.param.n.prim.dUdT(sto_surf_n)
        dUdT_p = self.param.p.prim.dUdT(sto_surf_p)
        Q_rev_n = a_n * j_n * T_av * dUdT_n
        Q_rev_p = a_p * j_p * T_av * dUdT_p

        # Irreversible (Joule) heating
        Q_irr_n = a_n * j_n * eta_n
        Q_irr_p = a_p * j_p * eta_p

        # Total heating in each electrode domain
        Q_total_n = Q_rev_n + Q_irr_n
        Q_total_p = Q_rev_p + Q_irr_p

        # Average the 1D heat source and broadcast to the 3D 'cell' domain
        L_n = self.param.n.L
        L_p = self.param.p.L
        L_x = self.param.L_x  # Total cell thickness

        # Calculate the true volume-weighted average heat source in W/m^3
        Q_vol_avg = (Q_total_n * L_n + Q_total_p * L_p) / L_x

        # Broadcast this uniform volumetric heat source to the entire 3D cell domain
        Q_source = pybamm.PrimaryBroadcast(Q_vol_avg, "cell")

        # Wrap in source term to get the correct mass matrix
        Q_source = pybamm.source(Q_source, T)

        # Define the 3D heat equation
        # Effective parameters are functions of temperature
        rho_c_p_eff = self.param.rho_c_p_eff(T)
        lambda_eff = self.param.lambda_eff(T)

        # The heat equation is d(rho*cp*T)/dt = div(lambda*grad(T)) + Q
        term1 = lambda_eff * pybamm.laplacian(T)
        term2 = pybamm.inner(pybamm.grad(lambda_eff), pybamm.grad(T))

        self.rhs[T] = (term1 + term2 + Q_source) / rho_c_p_eff

        # Cooling taken care of by boundary conditions

        self.initial_conditions[T] = pybamm.PrimaryBroadcast(self.param.T_init, "cell")
        self.set_thermal_bcs(T)

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        # Primary broadcasts are used to broadcast scalar quantities across a domain
        # into a vector of the right shape, for multiplying with other vectors
        self.variables = {
            "Time [s]": pybamm.t,
            "Discharge capacity [A.h]": Q,
            "X-averaged negative particle concentration [mol.m-3]": c_s_n,
            "Negative particle surface "
            "concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                c_s_surf_n, "negative electrode"
            ),
            "Electrolyte concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                self.param.c_e_init_av, whole_cell
            ),
            "X-averaged positive particle concentration [mol.m-3]": c_s_p,
            "Positive particle surface "
            "concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                c_s_surf_p, "positive electrode"
            ),
            "Current [A]": I,
            "Current variable [A]": I,  # for compatibility with pybamm.Experiment
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

        # Add new thermal variables
        self.variables.update(
            {
                "Cell temperature [K]": T,
                "Volume-averaged cell temperature [K]": T_av,
            }
        )

        # Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event("Minimum voltage [V]", V - self.param.voltage_low_cut),
            pybamm.Event("Maximum voltage [V]", self.param.voltage_high_cut - V),
        ]

    def set_thermal_bcs(self, T):
        geometry_type = self.options.get("cell geometry", "pouch")
        if geometry_type == "pouch":
            # Reuse the spatial variables created in __init__
            T_amb = self.param.T_amb(self.y_cell, self.z_cell, pybamm.t)
            face_params = {
                "x_min": self.param.h_edge_x_min,
                "x_max": self.param.h_edge_x_max,
                "y_min": self.param.h_edge_y_min,
                "y_max": self.param.h_edge_y_max,
                "z_min": self.param.h_edge_z_min,
                "z_max": self.param.h_edge_z_max,
            }
        elif geometry_type == "cylindrical":
            # Reuse the spatial variables created in __init__
            T_amb = self.param.T_amb(self.r_cell, self.z_cell, pybamm.t)
            face_params = {
                "r_min": self.param.h_edge_radial_min,
                "r_max": self.param.h_edge_radial_max,
                "z_min": self.param.h_edge_z_min,
                "z_max": self.param.h_edge_z_max,
            }
        else:
            raise ValueError(
                f"Geometry type '{geometry_type}' is not supported. "
                "Supported types are 'pouch' and 'cylindrical'."
            )  # pragma: no cover

        self.boundary_conditions[T] = {}

        for face, h_coeff in face_params.items():
            # Evaluate T and lambda_eff at the boundary
            T_boundary = pybamm.boundary_value(T, face)
            lambda_eff_boundary = pybamm.boundary_value(self.param.lambda_eff(T), face)
            # T_amb is already evaluated at the boundary coordinates
            q_face = -h_coeff * (T_boundary - T_amb)
            self.boundary_conditions[T][face] = (
                q_face / lambda_eff_boundary,
                "Neumann",
            )
