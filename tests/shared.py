#
# Shared methods and classes for testing
#
import pybamm

import numpy as np


class SpatialMethodForTesting(pybamm.SpatialMethod):
    """Identity operators, no boundary conditions."""

    def __init__(self, mesh):
        for dom in mesh.keys():
            mesh[dom].npts_for_broadcast = mesh[dom].npts
        super().__init__(mesh)

    def spatial_variable(self, symbol):
        # for finite volume we use the cell centres
        symbol_mesh = self.mesh.combine_submeshes(*symbol.domain)
        return pybamm.Vector(symbol_mesh.nodes)

    def get_num_of_vars(self, domain):
        return self.mesh[domain].npts

    def broadcast(self, symbol, domain):
        # for finite volume we send variables to cells and so use number_of_cells
        broadcasted_symbol = pybamm.NumpyBroadcast(symbol, domain, self.mesh)

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
        return gradient_matrix @ discretised_symbol

    def divergence(self, symbol, discretised_symbol, boundary_conditions):
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain].npts
        divergence_matrix = pybamm.Matrix(np.eye(n))
        return divergence_matrix @ discretised_symbol

    def compute_diffusivity(self, symbol):
        return symbol


def get_mesh_for_testing(npts=None):
    param = pybamm.ParameterValues(base_parameters={"Ln": 0.3, "Ls": 0.3, "Lp": 0.3})

    geometry = pybamm.Geometry("1D macro", "1D micro")
    param.process_geometry(geometry)

    submesh_types = {
        "negative electrode": pybamm.Uniform1DSubMesh,
        "separator": pybamm.Uniform1DSubMesh,
        "positive electrode": pybamm.Uniform1DSubMesh,
        "negative particle": pybamm.Uniform1DSubMesh,
        "positive particle": pybamm.Uniform1DSubMesh,
    }

    if npts is None:
        submesh_pts = {
            "negative electrode": {"x": 40},
            "separator": {"x": 25},
            "positive electrode": {"x": 35},
            "negative particle": {"r": 10},
            "positive particle": {"r": 10},
        }
    else:
        n = 3 * round(npts / 3)
        submesh_pts = {
            "negative electrode": {"x": n},
            "separator": {"x": n},
            "positive electrode": {"x": n},
            "negative particle": {"r": npts},
            "positive particle": {"r": npts},
        }
    return pybamm.Mesh(geometry, submesh_types, submesh_pts)


def get_discretisation_for_testing(npts=None):
    mesh = get_mesh_for_testing(npts)
    spatial_methods = {
        "macroscale": SpatialMethodForTesting,
        "negative particle": SpatialMethodForTesting,
        "positive particle": SpatialMethodForTesting,
    }

    return pybamm.Discretisation(mesh, spatial_methods)
