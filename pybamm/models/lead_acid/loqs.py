#
# Lead-acid LOQS model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import os


class LOQS(pybamm.BaseModel):
    """Leading-Order Quasi-Static model for lead-acid.

    Attributes
    ----------

    rhs: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the rhs
    initial_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the initial conditions
    boundary_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the boundary conditions
    variables: dict
        A dictionary that maps strings to expressions that represent
        the useful variables

    """

    def __init__(self):
        super().__init__()

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        # Variables
        c_e = pybamm.Variable("c", domain=[])
        eps_n = pybamm.Variable("eps_n", domain=[])
        eps_s = pybamm.Variable("eps_s", domain=[])
        eps_p = pybamm.Variable("eps_p", domain=[])

        # Current function
        i_cell = pybamm.standard_parameters.current_with_time
        # Parameters and functions
        l_n = pybamm.standard_parameters.l_n
        l_s = pybamm.standard_parameters.l_s
        l_p = pybamm.standard_parameters.l_p
        s_n = pybamm.standard_parameters.s_n
        s_p = pybamm.standard_parameters.s_p
        beta_surf_n = pybamm.standard_parameters_lead_acid.beta_surf_n
        beta_surf_p = pybamm.standard_parameters_lead_acid.beta_surf_p
        m_n = pybamm.standard_parameters.m_n
        m_p = pybamm.standard_parameters.m_p
        U_Pb = pybamm.standard_parameters.U_n_ref
        U_PbO2 = pybamm.standard_parameters.U_p_ref
        # Initial conditions
        c_e_init = pybamm.standard_parameters_lead_acid.c_e_init
        eps_n_init = pybamm.standard_parameters_lead_acid.eps_n_init
        eps_s_init = pybamm.standard_parameters_lead_acid.eps_s_init
        eps_p_init = pybamm.standard_parameters_lead_acid.eps_p_init

        # ODEs
        j_n = i_cell / l_n
        j_p = -i_cell / l_p
        deps_n_dt = -beta_surf_n * j_n
        deps_p_dt = -beta_surf_p * j_p
        dc_e_dt = (
            1
            / (l_n * eps_n + l_s * eps_s + l_p * eps_p)
            * ((s_n - s_p) * i_cell - c_e * (l_n * deps_n_dt + l_p * deps_p_dt))
        )
        self.rhs = {
            c_e: dc_e_dt,
            eps_n: deps_n_dt,
            eps_s: pybamm.Scalar(0),
            eps_p: deps_p_dt,
        }
        # Initial conditions
        self.initial_conditions = {
            c_e: c_e_init,
            eps_n: eps_n_init,
            eps_s: eps_s_init,
            eps_p: eps_p_init,
        }
        # ODE model -> no boundary conditions
        self.boundary_conditions = {}

        # Variables
        Phi = -U_Pb - j_n / (2 * m_n * c_e)
        V = Phi + U_PbO2 - j_p / (2 * m_p * c_e)
        # Phis_n = pybamm.Scalar(0)
        # Phis_p = V
        # Concatenate variables
        # eps = pybamm.Concatenation(eps_n, eps_s, eps_p)
        # Phis = pybamm.Concatenation(Phis_n, pybamm.Scalar(0), Phis_p)
        # self.variables = {"c": c, "eps": eps, "Phi": Phi, "Phis": Phis, "V": V}
        self.variables = {
            "c": pybamm.Broadcast(c_e, whole_cell),
            "Phi": pybamm.Broadcast(Phi, whole_cell),
            "V": V,
            "int(epsilon_times_c)dx": (l_n * eps_n + l_s * eps_s + l_p * eps_p) * c_e,
        }

        # Overwrite default parameter values
        self.default_parameter_values = pybamm.ParameterValues(
            "input/parameters/lead-acid/default.csv",
            {
                "Typical current density": 1,
                "Current function": os.path.join(
                    os.getcwd(),
                    "pybamm",
                    "parameters",
                    "standard_current_functions",
                    "constant_current.py",
                ),
            },
        )
