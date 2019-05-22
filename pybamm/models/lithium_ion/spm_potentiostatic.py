#
# Single Particle Model potentiostatic (SPM)
#
import pybamm

import numpy as np


class SPM_Potentiostatic(pybamm.LithiumIonBaseModel):
    """Single Particle Model (SPM) of a lithium-ion battery.
    **Extends:** :class:`pybamm.LithiumIonBaseModel`
    """

    def __init__(self):
        super().__init__()
        self.name = "Single Particle Model (potentiostatic)"

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lithium_ion
        v_cell = pybamm.Variable("Applied voltage")

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_s_n = pybamm.standard_variables.c_s_n
        c_s_p = pybamm.standard_variables.c_s_p
        i_cell = pybamm.Variable("Cell current density")

        "-----------------------------------------------------------------------------"
        "Equations"

        # solid fluxes
        N_n = -pybamm.grad(c_s_n)
        N_p = -pybamm.grad(c_s_p)

        # surface concentrations
        c_s_n_surf = pybamm.surf(c_s_n)
        c_s_p_surf = pybamm.surf(c_s_p)

        # open circuit potential
        ocp_n = param.U_n(c_s_n_surf)
        ocp_p = param.U_p(c_s_p_surf)

        # interfacial current
        j_n = i_cell / pybamm.geometric_parameters.l_n
        j_p = -i_cell / pybamm.geometric_parameters.l_p

        # exchange current density
        j0_n = (1 / param.C_r_n) * (c_s_n_surf ** (1 / 2) * (1 - c_s_n_surf) ** (1 / 2))
        j0_p = (param.gamma_p / param.C_r_p) * (
            c_s_p_surf ** (1 / 2) * (1 - c_s_p_surf) ** (1 / 2)
        )

        # reaction overpotentials
        eta_n = (2 / param.ne_n) * pybamm.Function(np.arcsinh, j_n / (2 * j0_n))
        eta_p = (2 / param.ne_p) * pybamm.Function(np.arcsinh, j_p / (2 * j0_p))

        # model equations
        self.rhs = {
            c_s_n: -(1 / param.C_n) * pybamm.div(N_n),
            c_s_p: -(param.gamma_p / param.C_p) * pybamm.div(N_p),
        }
        self.algebraic = {i_cell: v_cell - (ocp_p - ocp_n) - (eta_p - eta_n)}
        self.initial_conditions = {
            c_s_n: param.c_n_init,
            c_s_p: param.c_p_init,
            i_cell: 0,
        }
        rbc_n = -param.C_n * j_n / param.a_n
        rbc_p = -param.C_p * j_p / param.a_p / param.gamma_p
        self.boundary_conditions = {
            c_s_n: {"left": (0, "Neumann"), "right": (rbc_n, "Neumann")},
            c_s_p: {"left": (0, "Neumann"), "right": (rbc_p, "Neumann")},
        }
        self.variables = {
            "Negative particle concentration": c_s_n,
            "Negative particle surface concentration": c_s_n_surf,
            "Positive particle concentration": c_s_p,
            "Positive particle surface concentration": c_s_p_surf,
            "Cell current density": i_cell,
        }

        "-----------------------------------------------------------------------------"
        "Defaults and Solver Conditions"
        # default geometry
        self.default_geometry = pybamm.Geometry("1D macro", "1D micro")
        # default solver to DAE
        self.default_solver = pybamm.ScikitsDaeSolver()
