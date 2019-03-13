#
# Li-ion single particle model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import autograd.numpy as np


class SPM(pybamm.BaseModel):
    """Single Particle Model for li-ion.

    **Extends**: :class:`pybamm.BaseModel`

    """

    def __init__(self):
        super().__init__()

        # Variables
        c_n = pybamm.Variable(
            "Negative particle concentration", domain="negative particle"
        )
        c_p = pybamm.Variable(
            "Positive particle concentration", domain="positive particle"
        )

        # Parameters
        sp = pybamm.standard_parameters
        spli = pybamm.standard_parameters_lithium_ion
        # Current function
        i_cell = sp.current_with_time

        # PDE RHS
        N_n = -spli.C_n * spli.D_n(c_n) * pybamm.grad(c_n)
        dc_n_dt = -pybamm.div(N_n)
        N_p = -spli.C_p * spli.D_p(c_p) * pybamm.grad(c_p)
        dc_p_dt = -pybamm.div(N_p)
        self.rhs = {c_n: dc_n_dt, c_p: dc_p_dt}

        # Boundary conditions
        self.boundary_conditions = {
            N_n: {"left": 0, "right": i_cell / sp.l_n / spli.beta_n},
            N_p: {
                "left": 0,
                "right": -i_cell / sp.l_p / spli.beta_p / spli.gamma_hat_p,
            },
        }

        # Initial conditions
        self.initial_conditions = {c_n: spli.c_n_init, c_p: spli.c_p_init}

        # Variables
        c_n_surf = pybamm.surf(c_n)
        c_p_surf = pybamm.surf(c_p)
        j0_n = sp.m_n * c_n_surf ** 0.5 * (1 - c_n_surf) ** 0.5
        j0_p = sp.m_p * spli.gamma_hat_p * c_p_surf ** 0.5 * (1 - c_p_surf) ** 0.5
        # linearise BV for now
        V = (
            spli.U_p(c_p_surf)
            - spli.U_n(c_n_surf)
            - 2 * (i_cell / (j0_p * sp.l_p))
            - 2 * (i_cell / (j0_n * sp.l_n))
        )

        self.variables = {
            "cn": c_n,
            "cp": c_p,
            "cn_surf": c_n_surf,
            "cp_surf": c_p_surf,
            "V": V,
        }

        # Cut-off if either concentration goes negative
        self.events = [pybamm.Function(np.min, c_n), pybamm.Function(np.min, c_p)]
