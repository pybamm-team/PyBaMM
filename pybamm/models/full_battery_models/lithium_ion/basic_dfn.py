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
           derivation of a single particle model with electrolyte”. In: arXiv preprint
           arXiv:1905.12553 (2019).


    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(self, name="Doyle-Fuller-Newman model"):
        super().__init__({}, name)
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
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        # Electrolyte potential
        phi_e_n = pybamm.Variable(
            "Negative electrolyte potential", domain="negative electrode"
        )
        phi_e_s = pybamm.Variable("Separator electrolyte potential", domain="separator")
        phi_e_p = pybamm.Variable(
            "Positive electrolyte potential", domain="positive electrode"
        )
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

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
        eps = pybamm.Concatenation(eps_n, eps_s, eps_p)

        # Tortuosity
        tor = pybamm.Concatenation(
            eps_n ** param.b_e_n, eps_s ** param.b_e_s, eps_p ** param.b_e_p
        )

        # Interfacial reactions
        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        c_s_surf_n = pybamm.surf(c_s_n)
        j0_n = param.j0_n(c_e_n, c_s_surf_n, T) / param.C_r_n
        j_n = (
            2
            * j0_n
            * pybamm.sinh(
                param.ne_n / 2 * (phi_s_n - phi_e_n - param.U_n(c_s_surf_n, T))
            )
        )
        c_s_surf_p = pybamm.surf(c_s_p)
        j0_p = param.gamma_p * param.j0_p(c_e_p, c_s_surf_p, T) / param.C_r_p
        j_s = pybamm.PrimaryBroadcast(0, "separator")
        j_p = (
            2
            * j0_p
            * pybamm.sinh(
                param.ne_p / 2 * (phi_s_p - phi_e_p - param.U_p(c_s_surf_p, T))
            )
        )
        j = pybamm.Concatenation(j_n, j_s, j_p)

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
        N_s_n = -param.D_n(c_s_n, T) * pybamm.grad(c_s_n)
        N_s_p = -param.D_p(c_s_p, T) * pybamm.grad(c_s_p)
        self.rhs[c_s_n] = -(1 / param.C_n) * pybamm.div(N_s_n)
        self.rhs[c_s_p] = -(1 / param.C_p) * pybamm.div(N_s_p)
        # Boundary conditions must be provided for equations with spatial derivatives
        self.boundary_conditions[c_s_n] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -param.C_n * j_n / param.a_n / param.D_n(c_s_surf_n, T),
                "Neumann",
            ),
        }
        self.boundary_conditions[c_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -param.C_p * j_p / param.a_p / param.gamma_p / param.D_p(c_s_surf_p, T),
                "Neumann",
            ),
        }
        # c_n_init and c_p_init can in general be functions of x
        # Note the broadcasting, for domains
        x_n = pybamm.PrimaryBroadcast(
            pybamm.standard_spatial_vars.x_n, "negative particle"
        )
        self.initial_conditions[c_s_n] = param.c_n_init(x_n)
        x_p = pybamm.PrimaryBroadcast(
            pybamm.standard_spatial_vars.x_p, "positive particle"
        )
        self.initial_conditions[c_s_p] = param.c_p_init(x_p)
        # Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event(
                "Minimum negative particle surface concentration",
                pybamm.min(c_s_surf_n) - 0.01,
            ),
            pybamm.Event(
                "Maximum negative particle surface concentration",
                (1 - 0.01) - pybamm.max(c_s_surf_n),
            ),
            pybamm.Event(
                "Minimum positive particle surface concentration",
                pybamm.min(c_s_surf_p) - 0.01,
            ),
            pybamm.Event(
                "Maximum positive particle surface concentration",
                (1 - 0.01) - pybamm.max(c_s_surf_p),
            ),
        ]
        ######################
        # Current in the solid
        ######################
        i_s_n = -param.sigma_n * (1 - eps_n) ** param.b_s_n * pybamm.grad(phi_s_n)
        sigma_eff_p = param.sigma_p * (1 - eps_p) ** param.b_s_p
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
        self.initial_conditions[phi_s_p] = param.U_p(
            param.c_p_init(1), param.T_init
        ) - param.U_n(param.c_n_init(0), param.T_init)

        ######################
        # Current in the electrolyte
        ######################
        i_e = (param.kappa_e(c_e, T) * tor * param.gamma_e / param.C_e) * (
            param.chi(c_e) * pybamm.grad(c_e) / c_e - pybamm.grad(phi_e)
        )
        self.algebraic[phi_e] = pybamm.div(i_e) - j
        self.boundary_conditions[phi_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[phi_e] = -param.U_n(param.c_n_init(0), param.T_init)

        ######################
        # Electrolyte concentration
        ######################
        N_e = -tor * param.D_e(c_e, T) * pybamm.grad(c_e)
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) / param.C_e + (1 - param.t_plus(c_e)) * j / param.gamma_e
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
        }
        self.events += [
            pybamm.Event("Minimum voltage", voltage - param.voltage_low_cut),
            pybamm.Event("Maximum voltage", voltage - param.voltage_high_cut),
        ]
