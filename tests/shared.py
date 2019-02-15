#
# Shared methods and classes for testing
#
import pybamm

import numpy as np


class TestDefaults1DMacro:
    def __init__(self, npts=0):
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

        if npts != 0:
            n = 3 * round(npts / 3)
            self.submesh_pts = {
                "negative electrode": {"x": n},
                "separator": {"x": n},
                "positive electrode": {"x": n},
            }

        self.mesh = pybamm.Mesh(self.geometry, self.submesh_types, self.submesh_pts)

        self.spatial_methods = {
            "negative electrode": SpatialMethodForTesting,
            "separator": SpatialMethodForTesting,
            "positive electrode": SpatialMethodForTesting,
        }


class TestDefaults1DParticle:
    def __init__(self, n):
        self.geometry = {
            "negative particle": {
                "r": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}
            }
        }
        self.param = pybamm.ParameterValues(base_parameters={})
        self.param.process_geometry(self.geometry)
        self.submesh_pts = {"negative particle": {"r": n}}
        self.submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}

        self.mesh = pybamm.Mesh(self.geometry, self.submesh_types, self.submesh_pts)

        self.spatial_methods = {"negative particle": pybamm.FiniteVolume}


class SpatialMethodForTesting(pybamm.SpatialMethod):
    """Identity operators, no boundary conditions."""

    def __init__(self, mesh):
        self._mesh = mesh
        super().__init__(mesh)

    def spatial_variable(self, symbol):
        # for finite volume we use the cell centres
        symbol_mesh = self._mesh.combine_submeshes(*symbol.domain)
        return pybamm.Vector(symbol_mesh.nodes)

    def broadcast(self, symbol, domain):
        # for finite volume we send variables to cells and so use number_of_cells
        number_of_cells = {dom: submesh.npts for dom, submesh in self._mesh.items()}
        broadcasted_symbol = pybamm.NumpyBroadcast(symbol, domain, number_of_cells)

        # if the broadcasted symbol evaluates to a constant value, replace the
        # symbol-Vector multiplication with a single array
        if broadcasted_symbol.is_constant():
            broadcasted_symbol = pybamm.Array(
                broadcasted_symbol.evaluate(), domain=broadcasted_symbol.domain
            )

        return broadcasted_symbol

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain].npts
        gradient_matrix = pybamm.Matrix(np.eye(n))
        return gradient_matrix * discretised_symbol

    def divergence(self, symbol, discretised_symbol, boundary_conditions):
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain].npts
        divergence_matrix = pybamm.Matrix(np.eye(n))
        return divergence_matrix * discretised_symbol

    def compute_diffusivity(self, symbol):
        return symbol
