#
# Basic Reservoir Model
#
import pybamm
from .base_lithium_ion_model import BaseModel


class BasicReservoir(BaseModel):
    """Reservoir model of a lithium-ion battery, from
    :footcite:t:`Marquis2019`.

    Parameters
    ----------
    name : str, optional
        The name of the model.
    """

    def __init__(self, name="Reservoir Model"):
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
        sto_n = pybamm.Variable(
            "Average negative particle stoichiometry",
            domain="current collector",
            bounds=(0, 1),
        )
        sto_p = pybamm.Variable(
            "Average positive particle stoichiometry",
            domain="current collector",
            bounds=(0, 1),
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

        self.rhs[sto_n] = -i_cell / (
            param.n.L * param.n.prim.epsilon_s_av * param.n.prim.c_max * param.F
        )
        self.rhs[sto_p] = i_cell / (
            param.p.L * param.p.prim.epsilon_s_av * param.p.prim.c_max * param.F
        )
        # c_n_init and c_p_init are functions of r and x, but for the reservoir model
        # we take the x-averaged and r-averaged value since there are no x-dependence
        # nor r-dependencein the particles
        self.initial_conditions[sto_n] = (
            pybamm.x_average(pybamm.r_average(param.n.prim.c_init)) / param.n.prim.c_max
        )
        self.initial_conditions[sto_p] = (
            pybamm.x_average(pybamm.r_average(param.p.prim.c_init)) / param.p.prim.c_max
        )

        self.events += [
            pybamm.Event(
                "Minimum negative particle surface stoichiometry",
                sto_n - 0.01,
            ),
            pybamm.Event(
                "Maximum negative particle surface stoichiometry",
                (1 - 0.01) - sto_n,
            ),
            pybamm.Event(
                "Minimum positive particle surface stoichiometry",
                sto_p - 0.01,
            ),
            pybamm.Event(
                "Maximum positive particle surface stoichiometry",
                (1 - 0.01) - sto_p,
            ),
        ]

        # Note that the reservoir model does not have any algebraic equations, so the
        # `algebraic` dictionary remains empty

        ######################
        # (Some) variables
        ######################
        # Interfacial reactions
        RT_F = param.R * T / param.F
        j0_n = param.n.prim.j0(param.c_e_init_av, sto_n * param.n.prim.c_max, T)
        j0_p = param.p.prim.j0(param.c_e_init_av, sto_p * param.p.prim.c_max, T)
        eta_n = (2 / param.n.prim.ne) * RT_F * pybamm.arcsinh(j_n / (2 * j0_n))
        eta_p = (2 / param.p.prim.ne) * RT_F * pybamm.arcsinh(j_p / (2 * j0_p))
        phi_s_n = 0
        phi_e = -eta_n - param.n.prim.U(sto_n, T)
        phi_s_p = eta_p + phi_e + param.p.prim.U(sto_p, T)
        V = phi_s_p
        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )
        c_s_n = sto_n * param.n.prim.c_max
        c_s_p = sto_p * param.p.prim.c_max

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        # Primary broadcasts are used to broadcast scalar quantities across a domain
        # into a vector of the right shape, for multiplying with other vectors
        self.variables = {
            "Time [s]": pybamm.t,
            "Discharge capacity [A.h]": Q,
            "X-averaged negative particle concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                c_s_n, "negative particle"
            ),
            "Negative particle surface "
            "concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                c_s_n, "negative electrode"
            ),
            "Electrolyte concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                param.c_e_init_av, whole_cell
            ),
            "X-averaged positive particle concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                c_s_p, "positive particle"
            ),
            "Positive particle surface "
            "concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                c_s_p, "positive electrode"
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
        # Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event("Minimum voltage [V]", V - param.voltage_low_cut),
            pybamm.Event("Maximum voltage [V]", param.voltage_high_cut - V),
        ]
