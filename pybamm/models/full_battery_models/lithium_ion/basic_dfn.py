#
# Basic Doyle-Fuller-Newman (DFN) Model
#
import pybamm
from .base_lithium_ion_model import BaseModel


class BasicDFN(BaseModel):
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
        super().__init__({}, name)
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
        c_s_n = pybamm.Variable(
            "Negative particle concentration",
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

        # transport_efficiency
        tor = pybamm.concatenation(
            eps_n**param.n.b_e, eps_s**param.s.b_e, eps_p**param.p.b_e
        )

        # Interfacial reactions
        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        c_s_surf_n = pybamm.surf(c_s_n)
        j0_n = param.n.prim.j0(c_e_n, c_s_surf_n, T)
        j_n = (
            2
            * j0_n
            * pybamm.sinh(
                param.n.prim.ne
                / 2
                * (phi_s_n - phi_e_n - param.n.prim.U(c_s_surf_n, T))
            )
        )
        c_s_surf_p = pybamm.surf(c_s_p)
        j0_p = param.p.prim.j0(c_e_p, c_s_surf_p, T)
        j_s = pybamm.PrimaryBroadcast(0, "separator")
        j_p = (
            2
            * j0_p
            * pybamm.sinh(
                param.p.prim.ne
                / 2
                * (phi_s_p - phi_e_p - param.p.prim.U(c_s_surf_p, T))
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
        N_s_n = -param.n.prim.D(c_s_n, T) * pybamm.grad(c_s_n)
        N_s_p = -param.p.prim.D(c_s_p, T) * pybamm.grad(c_s_p)
        self.rhs[c_s_n] = -(1 / param.n.prim.C_diff) * pybamm.div(N_s_n)
        self.rhs[c_s_p] = -(1 / param.p.prim.C_diff) * pybamm.div(N_s_p)
        # Boundary conditions must be provided for equations with spatial derivatives
        self.boundary_conditions[c_s_n] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -param.n.prim.C_diff
                * j_n
                / param.n.prim.a_R
                / param.n.prim.gamma
                / param.n.prim.D(c_s_surf_n, T),
                "Neumann",
            ),
        }
        self.boundary_conditions[c_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -param.p.prim.C_diff
                * j_p
                / param.p.prim.a_R
                / param.p.prim.gamma
                / param.p.prim.D(c_s_surf_p, T),
                "Neumann",
            ),
        }
        self.initial_conditions[c_s_n] = param.n.prim.c_init
        self.initial_conditions[c_s_p] = param.p.prim.c_init
        ######################
        # Current in the solid
        ######################
        sigma_eff_n = param.n.sigma(T) * eps_s_n**param.n.b_s
        i_s_n = -sigma_eff_n * pybamm.grad(phi_s_n)
        sigma_eff_p = param.p.sigma(T) * eps_s_p**param.p.b_s
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
        # We evaluate c_n_init at r=0, x=0 and c_p_init at r=0, x=1
        # (this is just an initial guess so actual value is not too important)
        self.initial_conditions[phi_s_n] = pybamm.Scalar(0)
        self.initial_conditions[phi_s_p] = param.ocv_init

        ######################
        # Current in the electrolyte
        ######################
        i_e = (param.kappa_e(c_e, T) * tor * param.gamma_e / param.C_e) * (
            param.chiRT_over_Fc(c_e, T) * pybamm.grad(c_e) - pybamm.grad(phi_e)
        )
        self.algebraic[phi_e] = pybamm.div(i_e) - j
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
            -pybamm.div(N_e) / param.C_e
            + (1 - param.t_plus(c_e, T)) * j / param.gamma_e
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
        pot_scale = param.potential_scale
        U_ref = param.ocv_ref
        voltage_dim = U_ref + voltage * pot_scale
        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        self.variables = {
            "Negative particle surface concentration": c_s_surf_n,
            "Electrolyte concentration": c_e,
            "Positive particle surface concentration": c_s_surf_p,
            "Current [A]": I,
            "Negative electrode potential": phi_s_n,
            "Electrolyte potential": phi_e,
            "Positive electrode potential": phi_s_p,
            "Terminal voltage": voltage,
            "Terminal voltage [V]": voltage_dim,
            "Time [s]": pybamm.t * self.param.timescale,
        }
        # Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event("Minimum voltage", voltage - param.voltage_low_cut),
            pybamm.Event("Maximum voltage", param.voltage_high_cut - voltage),
        ]
