#
# Shared methods and classes for testing
#
import pybamm

import numpy as np


class TestDefaults1DMacro:
    def __init__(self):
        self.param = pybamm.ParameterValues(
            base_parameters={"Ln": 0.3, "Ls": 0.3, "Lp": 0.3}
        )

        self.geometry = pybamm.Geometry1DMacro()
        self.param.process_geometry(self.geometry)

        self.submesh_pts = {
            "negative electrode": {"x": 40},
            "separator": {"x": 25},
            "positive electrode": {"x": 35},
        }
        self.submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
        }

        self.mesh_type = pybamm.Mesh

    def set_equal_pts(self, npts):
        n = 3 * round(npts / 3)
        self.submesh_pts = {
            "negative electrode": {"x": n},
            "separator": {"x": n},
            "positive electrode": {"x": n},
        }


class DiscretisationForTesting(pybamm.BaseDiscretisation):
    """Identity operators, no boundary conditions."""

    def __init__(self, mesh_type, submesh_pts, submesh_types):
        super().__init__(mesh_type, submesh_pts, submesh_types)

    def set_mesh(self, geometry):
        self._mesh = self.mesh_type(geometry, self.submesh_types, self.submesh_pts)

    def gradient(self, symbol, y_slices, boundary_conditions):
        discretised_symbol = self.process_symbol(symbol, y_slices, boundary_conditions)
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain].npts
        gradient_matrix = pybamm.Matrix(np.eye(n))
        return gradient_matrix * discretised_symbol

    def divergence(self, symbol, y_slices, boundary_conditions):
        discretised_symbol = self.process_symbol(symbol, y_slices, boundary_conditions)
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain].npts
        divergence_matrix = pybamm.Matrix(np.eye(n))
        return divergence_matrix * discretised_symbol
