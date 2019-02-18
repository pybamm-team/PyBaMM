#
# Li-ion single particle model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import numpy as np

class SPM(pybamm.BaseModel):
    """Single Particle Model for li-ion.

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

        # Overwrite default geometry
        # NOTE: Probably not the best way to do this
        self.default_geometry = pybamm.Geometry1DMacro()
        self.default_parameter_values.process_geometry(self.default_geometry)
        submesh_pts = {
            "negative particle": {"r": 10},
            "positive particle": {"r": 10},
        }
        submehs_types = {
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
        }
        self.mesh = pybamm.Mesh(self.default_geometry, submesh_types, submesh_pts)

        # Variables
        cn = pybamm.Variable("cn", domain="negative particle")
        cp = pybamm.Variable("cp", domain="positive particle")

        # Parameters
        ln = pybamm.standard_parameters.ln
        lp = pybamm.standard_parameters.lp
        gamma_n = pybamm.standard_parameters.gamma_n
        gamma_p = pybamm.standard_parameters.gamma_p
        beta_n = pybamm.standard_parameters.beta_n
        beta_p = pybamm.standard_parameters.beta_p
        C_hat_p = pybamm.standard_parameters.C_hat_p
        m_n = pybamm.standard_parameters.m_n
        m_p = pybamm.standard_parameters.m_p
        D_n = pybamm.standard_parameters.D_n
        D_p = pybamm.standard_parameters.D_p
        U_n = pybamm.standard_parameters.U_n
        U_p = pybamm.standard_parameters.U_p
        Lambda = pybamm.standard_parameters.Lambda

        # Initial conditions
        cn_init = pybamm.standard_parameters.cn0
        cp_init = pybamm.standard_parameters.cp0

        # PDE RHS
        Nn = - gamma_n * D_n(cn) * pybamm.grad(cn)
        dcndt = - pybamm.div(Nn)
        Np = - gamma_p * D_p(cp) * pybamm.grad(cp)
        dcpdt = -pybamm.div(Np)
        self.rhs = {cn: dcndt, cp: dcpdt}

        # Boundary conditions
        # Note: this is for constant current discharge only
        self.boundary_conditions = {
            Nn: {"left": pybamm.Scalar(0),
                 "right": pybamm.Scalar(1) / ln / beta_n},
            Np: {"left": pybamm.Scalar(0),
                 "right": pybamm.Scalar(1) / lp / beta_p / C_hat_p},
        }

        # Initial conditions
        self.initial_conditions = {
            cn: cn_init,
            cp: cp_init,
        }

        # Variables
        # Question: how to get surface values of c?
        #  cn_surf =
        #  cp_surf =
        #  gn = m_n * cn_surf ** 0.5 * (1 - cn_surf) ** 0.5
        #  gp = m_p * C_hat_p * cp_surf ** 0.5 * (1 - cp_surf) ** 0.5
        # linearise BV for now
        #  V = (U_p(cp_surf) - U_n(cn_surf)
        #     - (2 / Lambda) * np.arcsinh(1 / (gp * lp))
        #     - (2 / Lambda) * np.arcsinh(1 / (gn * ln)))
        #V = (U_p(cp_surf) - U_n(cn_surf)
        #     - (2 / Lambda) * (1 / (gp * lp))
        #     - (2 / Lambda) * (1 / (gn * ln)))

        # Just returns ref value of OCV
        V = U_p(1) - U_n(0)
        self.variables = {
            "cn": cn,
            "cp": cp,
            "V" : V,
        }
