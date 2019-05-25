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


def get_mesh_for_testing(xpts=None, rpts=10, zpts=15, geometry=None):
    param = pybamm.ParameterValues(
        base_parameters={
            "Negative electrode width [m]": 0.3,
            "Separator width [m]": 0.3,
            "Positive electrode width [m]": 0.3,
            "Electrode height [m]": 0.5,
        }
    )

    if geometry is None:
        geometry = pybamm.Geometry("1D macro", "1D micro")
    param.process_geometry(geometry)

    submesh_types = {
        "negative electrode": pybamm.Uniform1DSubMesh,
        "separator": pybamm.Uniform1DSubMesh,
        "positive electrode": pybamm.Uniform1DSubMesh,
        "negative particle": pybamm.Uniform1DSubMesh,
        "positive particle": pybamm.Uniform1DSubMesh,
        "current collector": pybamm.Uniform1DSubMesh,
    }

    if xpts is None:
        xn_pts, xs_pts, xp_pts = 40, 25, 35
    else:
        xn_pts, xs_pts, xp_pts = xpts, xpts, xpts
    var = pybamm.standard_spatial_vars
    var_pts = {
        var.x_n: xn_pts,
        var.x_s: xs_pts,
        var.x_p: xp_pts,
        var.r_n: rpts,
        var.r_p: rpts,
        var.z: zpts,
    }

    return pybamm.Mesh(geometry, submesh_types, var_pts)


def get_p2d_mesh_for_testing(xpts=None, rpts=10):
    geometry = pybamm.Geometry("1D macro", "1+1D micro")
    return get_mesh_for_testing(xpts=xpts, rpts=rpts, geometry=geometry)


def get_1p1d_mesh_for_testing(xpts=None, zpts=15):
    geometry = pybamm.Geometry("1+1D macro")
    return get_mesh_for_testing(xpts=xpts, zpts=zpts, geometry=geometry)


def get_discretisation_for_testing(xpts=None, mesh=None):
    if mesh is None:
        mesh = get_mesh_for_testing(xpts)
    spatial_methods = {
        "macroscale": SpatialMethodForTesting,
        "negative particle": SpatialMethodForTesting,
        "positive particle": SpatialMethodForTesting,
        "current collector": SpatialMethodForTesting,
    }

    return pybamm.Discretisation(mesh, spatial_methods)


def get_p2d_discretisation_for_testing(xpts=None, rpts=10):
    return get_discretisation_for_testing(mesh=get_p2d_mesh_for_testing(xpts, rpts))


def get_1p1d_discretisation_for_testing(xpts=None, zpts=15):
    return get_discretisation_for_testing(mesh=get_1p1d_mesh_for_testing(xpts, zpts))
