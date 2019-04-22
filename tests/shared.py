#
# Shared methods and classes for testing
#
import pybamm

from scipy.sparse import eye, coo_matrix


class SpatialMethodForTesting(pybamm.SpatialMethod):
    """Identity operators, no boundary conditions."""

    def __init__(self, mesh):
        for dom in mesh.keys():
            for i in range(len(mesh[dom])):
                mesh[dom][i].npts_for_broadcast = mesh[dom][i].npts
        super().__init__(mesh)

    def spatial_variable(self, symbol):
        # use the cell centres
        symbol_mesh = self.mesh.combine_submeshes(*symbol.domain)
        return pybamm.Vector(symbol_mesh[0].nodes)

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

    def boundary_value(self, symbol, discretised_symbol, side):
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain][0].npts
        left_matrix = coo_matrix(([1], ([0], [0])), shape=(n, n))
        right_matrix = coo_matrix(([1], ([n - 1], [n - 1])), shape=(n, n))
        if side == "left":
            bv_matrix = pybamm.Matrix(left_matrix)
        elif side == "right":
            bv_matrix = pybamm.Matrix(right_matrix)
        return bv_matrix @ discretised_symbol


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

    var = pybamm.standard_spatial_vars

    if npts is None:
        var_pts = {var.x_n: 40, var.x_s: 25, var.x_p: 35, var.r_n: 10, var.r_p: 10}
    else:
        n = 3 * round(npts / 3)
        var_pts = {var.x_n: n, var.x_s: n, var.x_p: n, var.r_n: npts, var.r_p: npts}
    return pybamm.Mesh(geometry, submesh_types, var_pts)


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

    var = pybamm.standard_spatial_vars
    if mpts is None:
        var_pts = {var.x_n: 40, var.x_s: 25, var.x_p: 35, var.r_n: 10, var.r_p: 10}
    else:
        n = 3 * round(npts / 3)
        var_pts = {var.x_n: n, var.x_s: n, var.x_p: n, var.r_n: mpts, var.r_p: mpts}

    return pybamm.Mesh(geometry, submesh_types, var_pts)


def get_discretisation_for_testing(npts=None):
    mesh = get_mesh_for_testing(npts)
    spatial_methods = {
        "macroscale": SpatialMethodForTesting,
        "negative particle": SpatialMethodForTesting,
        "positive particle": SpatialMethodForTesting,
    }

    return pybamm.Discretisation(mesh, spatial_methods)
