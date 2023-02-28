#
# Basic Single Particle Model (SPM)
#
import pybamm
from .base_lithium_ion_model import BaseModel


class BasicSPM(BaseModel):
    """Single Particle Model (SPM) model of a lithium-ion battery, from [2]_.

    This class differs from the :class:`pybamm.lithium_ion.SPM` model class in that it
    shows the whole model in a single class. This comes at the cost of flexibility in
    combining different physical effects, and in general the main SPM class should be
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

    def __init__(self, name="Single Particle Model"):
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
        c_s_n = pybamm.Variable(
            "X-averaged negative particle concentration [mol.m-3]",
            domain="negative particle",
        )
        c_s_p = pybamm.Variable(
            "X-averaged positive particle concentration [mol.m-3]",
            domain="positive particle",
        )

        # Constant temperature
        T = param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = param.current_density_with_time
        a_n = 3 * param.n.prim.epsilon_s_av / param.n.prim.R_typ
        a_p = 3 * param.p.prim.epsilon_s_av / param.p.prim.R_typ
        j_n = i_cell / (param.n.L * a_n)
        j_p = -i_cell / (param.p.L * a_p)

        ######################
        # State of Charge
        ######################
        I = param.current_with_time
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
        N_s_n = -param.n.prim.D(c_s_n, T) * pybamm.grad(c_s_n)
        N_s_p = -param.p.prim.D(c_s_p, T) * pybamm.grad(c_s_p)
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
                -j_n / param.F / param.n.prim.D(c_s_surf_n, T),
                "Neumann",
            ),
        }
        self.boundary_conditions[c_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -j_p / param.F / param.p.prim.D(c_s_surf_p, T),
                "Neumann",
            ),
        }
        # c_n_init and c_p_init are functions of r and x, but for the SPM we
        # take the x-averaged value since there is no x-dependence in the particles
        self.initial_conditions[c_s_n] = pybamm.x_average(param.n.prim.c_init)
        self.initial_conditions[c_s_p] = pybamm.x_average(param.p.prim.c_init)
        # Events specify points at which a solution should terminate
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

        # Note that the SPM does not have any algebraic equations, so the `algebraic`
        # dictionary remains empty

        ######################
        # (Some) variables
        ######################
        # Interfacial reactions
        RT_F = param.R * T / param.F
        j0_n = param.n.prim.j0(param.c_e_init_av, c_s_surf_n, T)
        j0_p = param.p.prim.j0(param.c_e_init_av, c_s_surf_p, T)
        eta_n = (2 / param.n.prim.ne) * RT_F * pybamm.arcsinh(j_n / (2 * j0_n))
        eta_p = (2 / param.p.prim.ne) * RT_F * pybamm.arcsinh(j_p / (2 * j0_p))
        phi_s_n = 0
        phi_e = -eta_n - param.n.prim.U(sto_surf_n, T)
        phi_s_p = eta_p + phi_e + param.p.prim.U(sto_surf_p, T)
        V = phi_s_p

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        # Primary broadcasts are used to broadcast scalar quantities across a domain
        # into a vector of the right shape, for multiplying with other vectors
        self.variables = {
            "Discharge capacity [A.h]": Q,
            "Negative particle surface "
            "concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                c_s_surf_n, "negative electrode"
            ),
            "Electrolyte concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                param.c_e_init_av, whole_cell
            ),
            "Positive particle surface "
            "concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                c_s_surf_p, "positive electrode"
            ),
            "Current [A]": I,
            "Negative electrode potential [V]": pybamm.PrimaryBroadcast(
                phi_s_n, "negative electrode"
            ),
            "Electrolyte potential [V]": pybamm.PrimaryBroadcast(phi_e, whole_cell),
            "Positive electrode potential [V]": pybamm.PrimaryBroadcast(
                phi_s_p, "positive electrode"
            ),
            "Voltage [V]": V,
        }
        self.events += [
            pybamm.Event("Minimum voltage [V]", V - param.voltage_low_cut),
            pybamm.Event("Maximum voltage [V]", param.voltage_high_cut - V),
        ]
