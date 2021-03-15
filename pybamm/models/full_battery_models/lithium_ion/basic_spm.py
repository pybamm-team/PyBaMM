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
            "X-averaged negative particle concentration", domain="negative particle"
        )
        c_s_p = pybamm.Variable(
            "X-averaged positive particle concentration", domain="positive particle"
        )

        # Constant temperature
        T = param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = param.current_with_time
        j_n = i_cell / param.l_n
        j_p = -i_cell / param.l_p

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
        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        c_s_surf_n = pybamm.surf(c_s_n)
        c_s_surf_p = pybamm.surf(c_s_p)
        # Boundary conditions must be provided for equations with spatial derivatives
        self.boundary_conditions[c_s_n] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -param.C_n * j_n / param.a_R_n / param.D_n(c_s_surf_n, T),
                "Neumann",
            ),
        }
        self.boundary_conditions[c_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -param.C_p
                * j_p
                / param.a_R_p
                / param.gamma_p
                / param.D_p(c_s_surf_p, T),
                "Neumann",
            ),
        }
        # c_n_init and c_p_init are functions, but for the SPM we evaluate them at x=0
        # and x=1 since there is no x-dependence in the particles
        self.initial_conditions[c_s_n] = param.c_n_init(0)
        self.initial_conditions[c_s_p] = param.c_p_init(1)
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

        # Note that the SPM does not have any algebraic equations, so the `algebraic`
        # dictionary remains empty

        ######################
        # (Some) variables
        ######################
        # Interfacial reactions
        j0_n = param.j0_n(1, c_s_surf_n, T) / param.C_r_n
        j0_p = param.gamma_p * param.j0_p(1, c_s_surf_p, T) / param.C_r_p
        eta_n = (2 / param.ne_n) * pybamm.arcsinh(j_n / (2 * j0_n))
        eta_p = (2 / param.ne_p) * pybamm.arcsinh(j_p / (2 * j0_p))
        phi_s_n = 0
        phi_e = -eta_n - param.U_n(c_s_surf_n, T)
        phi_s_p = eta_p + phi_e + param.U_p(c_s_surf_p, T)
        V = phi_s_p

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        # Primary broadcasts are used to broadcast scalar quantities across a domain
        # into a vector of the right shape, for multiplying with other vectors
        self.variables = {
            "Negative particle surface concentration": pybamm.PrimaryBroadcast(
                c_s_surf_n, "negative electrode"
            ),
            "Electrolyte concentration": pybamm.PrimaryBroadcast(1, whole_cell),
            "Positive particle surface concentration": pybamm.PrimaryBroadcast(
                c_s_surf_p, "positive electrode"
            ),
            "Current [A]": I,
            "Negative electrode potential": pybamm.PrimaryBroadcast(
                phi_s_n, "negative electrode"
            ),
            "Electrolyte potential": pybamm.PrimaryBroadcast(phi_e, whole_cell),
            "Positive electrode potential": pybamm.PrimaryBroadcast(
                phi_s_p, "positive electrode"
            ),
            "Terminal voltage": V,
        }
        self.events += [
            pybamm.Event("Minimum voltage", V - param.voltage_low_cut),
            pybamm.Event("Maximum voltage", V - param.voltage_high_cut),
        ]

    def new_empty_copy(self):
        return pybamm.BaseModel.new_empty_copy(self)
