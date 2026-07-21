import pybamm

from .base_lithium_ion_model import BaseModel


class BasicDFN(BaseModel):
    """Doyle-Fuller-Newman (DFN) model of a lithium-ion battery, from
    :footcite:t:`Marquis2019`.

    This class differs from the :class:`pybamm.lithium_ion.DFN` model class in that it
    shows the whole model in a single class. This comes at the cost of flexibility in
    comparing different physical effects, and in general the main DFN class should be
    used instead.

    Parameters
    ----------
    name : str, optional
        The name of the model.

    """

    def __init__(self, name="Doyle-Fuller-Newman model"):
        super().__init__(name=name)
        pybamm.citations.register("Marquis2019")
        # `param` contains symbolic parameters set by `ParameterValues` at processing
        # Time-only variables created without domain
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
        # Concatenations combine several variables into a single variable, to simplify
        # implementing equations that hold over several domains
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
        # Particle concentrations vary in particle domain + x (electrode) auxiliary domains
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
        T = self.param.T_init

        ###################### Other set-up ######################

        # Current density (time-dependent)
        i_cell = self.param.current_density_with_time

        # Porosity: PrimaryBroadcasts broadcast scalars to vectors for multiplication
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
            eps_n**self.param.n.b_e, eps_s**self.param.s.b_e, eps_p**self.param.p.b_e
        )
        a_n = 3 * self.param.n.prim.epsilon_s_av / self.param.n.prim.R_typ
        a_p = 3 * self.param.p.prim.epsilon_s_av / self.param.p.prim.R_typ

        # Interfacial reactions: `pybamm.surf()` gets surface value (boundary on right side)
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

        ###################### State of Charge ######################
        I = self.param.current_with_time
        # `rhs` dict: differential equations keyed by d/dt variable
        self.rhs[Q] = I / 3600
        # Initial conditions for ODEs
        self.initial_conditions[Q] = pybamm.Scalar(0)

        ###################### Particles ######################

        # div/grad operators convert to matrix multiplication at discretisation
        N_s_n = -self.param.n.prim.D(c_s_n, T) * pybamm.grad(c_s_n)
        N_s_p = -self.param.p.prim.D(c_s_p, T) * pybamm.grad(c_s_p)
        self.rhs[c_s_n] = -pybamm.div(N_s_n)
        self.rhs[c_s_p] = -pybamm.div(N_s_p)
        # Boundary conditions must be provided for equations with spatial derivatives
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
        ###################### Current in the solid ######################
        sigma_eff_n = self.param.n.sigma(sto_surf_n, T) * eps_s_n**self.param.n.b_s
        i_s_n = -sigma_eff_n * pybamm.grad(phi_s_n)
        sigma_eff_p = self.param.p.sigma(sto_surf_p, T) * eps_s_p**self.param.p.b_s
        i_s_p = -sigma_eff_p * pybamm.grad(phi_s_p)
        # algebraic dict: differential equations keyed by main scalar variable;
        # multiplied by Lx**2 to improve conditioning.
        self.algebraic[phi_s_n] = self.param.L_x**2 * (pybamm.div(i_s_n) + a_j_n)
        self.algebraic[phi_s_p] = self.param.L_x**2 * (pybamm.div(i_s_p) + a_j_p)
        self.boundary_conditions[phi_s_n] = {
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.boundary_conditions[phi_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (i_cell / pybamm.boundary_value(-sigma_eff_p, "right"), "Neumann"),
        }
        # Initial conditions for algebraic equations serve as root-finding guesses
        self.initial_conditions[phi_s_n] = pybamm.Scalar(0)
        self.initial_conditions[phi_s_p] = self.param.ocv_init

        ###################### Current in the electrolyte ######################
        i_e = (self.param.kappa_e(c_e, T) * tor) * (
            self.param.chiRT_over_Fc(c_e, T) * pybamm.grad(c_e) - pybamm.grad(phi_e)
        )
        # multiply by Lx**2 to improve conditioning
        self.algebraic[phi_e] = self.param.L_x**2 * (pybamm.div(i_e) - a_j)
        self.boundary_conditions[phi_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[phi_e] = -self.param.n.prim.U_init

        ###################### Electrolyte concentration ######################
        N_e = -tor * self.param.D_e(c_e, T) * pybamm.grad(c_e)
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) + (1 - self.param.t_plus(c_e, T)) * a_j / self.param.F
        )
        self.boundary_conditions[c_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[c_e] = self.param.c_e_init

        ######################
        # (Some) variables
        voltage = pybamm.boundary_value(phi_s_p, "right")
        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )
        # `variables` dict: all variables useful for visualising the solution
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
            pybamm.Event("Minimum voltage [V]", voltage - self.param.voltage_low_cut),
            pybamm.Event("Maximum voltage [V]", self.param.voltage_high_cut - voltage),
        ]
