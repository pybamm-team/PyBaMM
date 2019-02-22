#
# Single Particle Model (SPM)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class SPM(pybamm.BaseModel):
    """Single Particle Model (SPM) of a lithium-ion battery.

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

        "Model Variables"
        # Particle concentration
        c_n = pybamm.Variable("c_n", ["negative particle"])
        c_p = pybamm.Variable("c_p", ["positive particle"])

        "Model Parameters and functions"
        m_n = pybamm.standard_parameters.m_n
        m_p = pybamm.standard_parameters.m_p
        U_n = pybamm.standard_parameters.U_n
        U_p = pybamm.standard_parameters.U_p
        Lambda = pybamm.standard_parameters.Lambda
        C_hat_p = pybamm.standard_parameters.C_hat_p
        ln = pybamm.standard_parameters.ln
        lp = pybamm.standard_parameters.lp

        "Interface Conditions"
        G_n = pybamm.interface.homogeneous_reaction(["negative electrode"])
        G_p = pybamm.interface.homogeneous_reaction(["positive electrode"])

        "Model Equations"
        self.update(
            pybamm.particle.Standard(c_n, G_n), pybamm.particle.Standard(c_p, G_p)
        )

        "Additional Conditions"
        # phi is only determined to a constant so set phi_n = 0 on left boundary
        additional_bcs = {}
        self._boundary_conditions.update(additional_bcs)

        "Additional Model Variables"
        # TODO: add voltage and overpotentials to this
        cn_surf = pybamm.surf(c_n)
        cp_surf = pybamm.surf(c_p)
        gn = m_n * cn_surf ** 0.5 * (1 - cn_surf) ** 0.5
        gp = m_p * C_hat_p * cp_surf ** 0.5 * (1 - cp_surf) ** 0.5
        # linearise BV for now
        ocp = U_p(cp_surf) - U_n(cn_surf)
        reaction_overpotential = -(2 / Lambda) * (1 / (gp * lp)) - (2 / Lambda) * (
            1 / (gn * ln)
        )
        voltage = ocp + reaction_overpotential

        # "opc": ocp,
        # "reaction overpotential": reaction_overpotential,
        additional_variables = {
            "cn_surf": cn_surf,
            "cp_surf": cp_surf,
            "voltage": voltage,
        }
        self._variables.update(additional_variables)

        #
        # ------------------------------------------------------
        #
        "Defaults"
        # NOTE: Is this the best way/place to do this?
        self.default_geometry = pybamm.Geometry1DMicro()
        self.default_parameter_values.process_geometry(self.default_geometry)
        submesh_pts = {"negative particle": {"r": 10}, "positive particle": {"r": 10}}
        submesh_types = {
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
        }
        self.default_spatial_methods = {
            "negative particle": pybamm.FiniteVolume,
            "positive particle": pybamm.FiniteVolume,
        }
        self.mesh = pybamm.Mesh(self.default_geometry, submesh_types, submesh_pts)
        self.default_discretisation = pybamm.Discretisation(
            self.mesh, self.default_spatial_methods
        )
        self.default_solver = pybamm.ScipySolver(method="BDF")

        # --------------------------------------------------------
