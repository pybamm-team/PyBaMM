#
# Shared methods and classes for testing
#
import pybamm

from scipy.sparse import eye


class SpatialMethodForTesting(pybamm.SpatialMethod):
    """Identity operators, no boundary conditions."""

    def __init__(self, mesh):
        super().__init__(mesh)

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain][0].npts
        gradient_matrix = pybamm.Matrix(eye(n))
        return gradient_matrix @ discretised_symbol

    def divergence(self, symbol, discretised_symbol, boundary_conditions):
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain][0].npts
        divergence_matrix = pybamm.Matrix(eye(n))
        return divergence_matrix @ discretised_symbol

    def mass_matrix(self, symbol, boundary_conditions):
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain][0].npts
        mass_matrix = pybamm.Matrix(eye(n))
        return mass_matrix


def get_mesh_for_testing(npts=None):
    param = pybamm.ParameterValues(
        base_parameters={
            "Negative electrode width [m]": 0.3,
            "Separator width [m]": 0.3,
            "Positive electrode width [m]": 0.3,
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

    var = pybamm.standard_spatial_vars

    if npts is None:
        var_pts = {var.x_n: 40, var.x_s: 25, var.x_p: 35, var.r_n: 10, var.r_p: 10}
    else:
        var_pts = {
            var.x_n: npts,
            var.x_s: npts,
            var.x_p: npts,
            var.r_n: npts,
            var.r_p: npts,
        }
    return pybamm.Mesh(geometry, submesh_types, var_pts)


def get_p2d_mesh_for_testing(npts=None, mpts=None):
    param = pybamm.ParameterValues(
        base_parameters={
            "Negative electrode width [m]": 0.3,
            "Separator width [m]": 0.2,
            "Positive electrode width [m]": 0.3,
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

    var = pybamm.standard_spatial_vars
    if mpts is None:
        var_pts = {var.x_n: 40, var.x_s: 25, var.x_p: 35, var.r_n: 10, var.r_p: 10}
    else:
        var_pts = {
            var.x_n: npts,
            var.x_s: npts,
            var.x_p: npts,
            var.r_n: mpts,
            var.r_p: mpts,
        }

    return pybamm.Mesh(geometry, submesh_types, var_pts)


def get_discretisation_for_testing(npts=None):
    mesh = get_mesh_for_testing(npts)
    spatial_methods = {
        "macroscale": SpatialMethodForTesting,
        "negative particle": SpatialMethodForTesting,
        "positive particle": SpatialMethodForTesting,
    }

    return pybamm.Discretisation(mesh, spatial_methods)


def get_p2d_discretisation_for_testing(npts=None, mpts=None):
    mesh = get_p2d_mesh_for_testing(npts, mpts)
    spatial_methods = {
        "macroscale": SpatialMethodForTesting,
        "negative particle": SpatialMethodForTesting,
        "positive particle": SpatialMethodForTesting,
    }

    return pybamm.Discretisation(mesh, spatial_methods)
