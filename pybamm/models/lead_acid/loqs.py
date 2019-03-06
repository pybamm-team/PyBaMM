#
# Lead-acid LOQS model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import numpy as np
import os


class LOQS(pybamm.BaseModel):
    """Leading-Order Quasi-Static model for lead-acid.

    **Extends**: :class:`pybamm.BaseModel`

    """

    def __init__(self):
        super().__init__()

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        # Variables
        c_e = pybamm.Variable("Concentration", domain=[])
        eps_n = pybamm.Variable("Negative electrode porosity", domain=[])
        eps_s = pybamm.Variable("Separator porosity", domain=[])
        eps_p = pybamm.Variable("Positive electrode porosity", domain=[])

        # Parameters
        sp = pybamm.standard_parameters
        spla = pybamm.standard_parameters_lead_acid
        # Current function
        i_cell = sp.current_with_time

        # ODEs
        j_n = i_cell / sp.l_n
        j_p = -i_cell / sp.l_p
        deps_n_dt = -spla.beta_surf_n * j_n
        deps_p_dt = -spla.beta_surf_p * j_p
        dc_e_dt = (
            1
            / (sp.l_n * eps_n + sp.l_s * eps_s + sp.l_p * eps_p)
            * (
                (sp.s_n - sp.s_p) * i_cell
                - c_e * (sp.l_n * deps_n_dt + sp.l_p * deps_p_dt)
            )
        )
        self.rhs = {c_e: dc_e_dt, eps_n: deps_n_dt, eps_s: 0, eps_p: deps_p_dt}
        # Initial conditions
        self.initial_conditions = {
            c_e: spla.c_e_init,
            eps_n: spla.eps_n_init,
            eps_s: spla.eps_s_init,
            eps_p: spla.eps_p_init,
        }
        # ODE model -> no boundary conditions
        self.boundary_conditions = {}

        # Variables
        j0_n = pybamm.interface.exchange_current_density(c_e, ["negative electrode"])
        j0_p = pybamm.interface.exchange_current_density(c_e, ["positive electrode"])
        Phi = -spla.U_n(c_e) - pybamm.Function(np.arcsinh, j_n / (2 * j0_n * sp.l_n))
        V = Phi + spla.U_p(c_e) - pybamm.Function(np.arcsinh, j_p / (2 * j0_p * sp.l_p))
        # Phis_n = pybamm.Scalar(0)
        # Phis_p = V
        # Concatenate variables
        # eps = pybamm.Concatenation(eps_n, eps_s, eps_p)
        # Phis = pybamm.Concatenation(Phis_n, pybamm.Scalar(0), Phis_p)
        # self.variables = {"c": c, "eps": eps, "Phi": Phi, "Phis": Phis, "V": V}
        self.variables = {
            "Concentration": pybamm.Broadcast(c_e, whole_cell),
            "Porosity": pybamm.Concatenation(
                pybamm.Broadcast(eps_n, ["negative electrode"]),
                pybamm.Broadcast(eps_s, ["separator"]),
                pybamm.Broadcast(eps_p, ["positive electrode"]),
            ),
            "Negative electrode overpotential": pybamm.Broadcast(
                Phi, ["negative electrode"]
            ),
            "Positive electrode overpotential": pybamm.Broadcast(
                V, ["positive electrode"]
            ),
            "Electrolyte potential": pybamm.Broadcast(Phi, whole_cell),
            "Voltage": V,
        }

        # Overwrite default parameter values
        input_path = os.path.join(os.getcwd(), "input", "parameters", "lead-acid")
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
                "Electrolyte diffusivity": os.path.join(
                    input_path, "electrolyte_diffusivity_Gu1997.py"
                ),
                "Electrolyte conductivity": os.path.join(
                    input_path, "electrolyte_conductivity_Gu1997.py"
                ),
                "Darken thermodynamic factor": os.path.join(
                    input_path, "darken_thermodynamic_factor_Chapman1968.py"
                ),
                "Negative electrode OCV": os.path.join(
                    input_path, "lead_electrode_ocv_Bode1977.py"
                ),
                "Positive electrode OCV": os.path.join(
                    input_path, "lead_dioxide_electrode_ocv_Bode1977.py"
                ),
            },
        )
