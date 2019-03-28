#
# Shared methods and classes for testing
#
import pybamm

import numpy as np


class SpatialMethodForTesting(pybamm.SpatialMethod):
    """Identity operators, no boundary conditions."""

    def __init__(self, mesh):
        for dom in mesh.keys():
            for i in range(len(mesh[dom])):
                mesh[dom][i].npts_for_broadcast = mesh[dom][i].npts
        super().__init__(mesh)

    def spatial_variable(self, symbol):
        # for finite volume we use the cell centres
        symbol_mesh = self.mesh.combine_submeshes(*symbol.domain)
        return pybamm.Vector(symbol_mesh[0].nodes)

    def broadcast(self, symbol, domain):
        return pybamm.NumpyBroadcast(symbol, domain, self.mesh)

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain][0].npts
        gradient_matrix = pybamm.Matrix(np.eye(n))
        return gradient_matrix @ discretised_symbol

    def divergence(self, symbol, discretised_symbol, boundary_conditions):
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain][0].npts
        divergence_matrix = pybamm.Matrix(np.eye(n))
        return divergence_matrix @ discretised_symbol

    def compute_diffusivity(
        self, symbol, extrapolate_left=None, extrapolate_right=None
    ):
        return symbol

    def mass_matrix(self, symbol, boundary_conditions):
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain][0].npts
        mass_matrix = pybamm.Matrix(np.eye(n))
        return mass_matrix


def get_mesh_for_testing(npts=None):
    param = pybamm.ParameterValues(
        base_parameters={
            "Negative electrode width": 0.3,
            "Separator width": 0.3,
            "Positive electrode width": 0.3,
        }
    )

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


def get_p2d_mesh_for_testing(npts=None, mpts=None):
    param = pybamm.ParameterValues(
        base_parameters={
            "Negative electrode width": 0.3,
            "Separator width": 0.2,
            "Positive electrode width": 0.3,
        }
    )

    geometry = pybamm.Geometry("1D macro", "1+1D micro")
    param.process_geometry(geometry)

    # provide mesh properties
    submesh_types = {
        "negative electrode": pybamm.Uniform1DSubMesh,
        "separator": pybamm.Uniform1DSubMesh,
        "positive electrode": pybamm.Uniform1DSubMesh,
        "negative particle": pybamm.Uniform1DSubMesh,
        "positive particle": pybamm.Uniform1DSubMesh,
    }

    if mpts is None:
        submesh_pts = {
            "negative electrode": {"x": 40},
            "separator": {"x": 25},
            "positive electrode": {"x": 35},
            "negative particle": {"r": 10, "x": 40},
            "positive particle": {"r": 10, "x": 35},
        }
    else:
        n = 3 * round(npts / 3)
        submesh_pts = {
            "negative electrode": {"x": n},
            "separator": {"x": n},
            "positive electrode": {"x": n},
            "negative particle": {"r": mpts, "x": n},
            "positive particle": {"r": mpts, "x": n},
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
