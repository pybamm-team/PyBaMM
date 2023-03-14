#
# Shared methods and classes for testing
#
import pybamm
from scipy.sparse import eye


class SpatialMethodForTesting(pybamm.SpatialMethod):
    """Identity operators, no boundary conditions."""

    def __init__(self, options=None):
        super().__init__(options)

    def build(self, mesh):
        super().build(mesh)

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain].npts
        gradient_matrix = pybamm.Matrix(eye(n))
        return gradient_matrix @ discretised_symbol

    def divergence(self, symbol, discretised_symbol, boundary_conditions):
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain].npts
        divergence_matrix = pybamm.Matrix(eye(n))
        return divergence_matrix @ discretised_symbol

    def internal_neumann_condition(
        self, left_symbol_disc, right_symbol_disc, left_mesh, right_mesh
    ):
        return pybamm.Scalar(0)

    def mass_matrix(self, symbol, boundary_conditions):
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain].npts
        mass_matrix = pybamm.Matrix(eye(n))
        return mass_matrix


def get_mesh_for_testing(
    xpts=None,
    rpts=10,
    Rpts=10,
    ypts=15,
    zpts=15,
    rcellpts=15,
    geometry=None,
    cc_submesh=None,
):
    param = pybamm.ParameterValues(
        values={
            "Electrode width [m]": 0.4,
            "Electrode height [m]": 0.5,
            "Negative tab width [m]": 0.1,
            "Negative tab centre y-coordinate [m]": 0.1,
            "Negative tab centre z-coordinate [m]": 0.0,
            "Positive tab width [m]": 0.1,
            "Positive tab centre y-coordinate [m]": 0.3,
            "Positive tab centre z-coordinate [m]": 0.5,
            "Negative electrode thickness [m]": 1 / 3,
            "Separator thickness [m]": 1 / 3,
            "Positive electrode thickness [m]": 1 / 3,
            "Negative particle radius [m]": 0.5,
            "Positive particle radius [m]": 0.5,
            "Inner cell radius [m]": 0.2,
            "Outer cell radius [m]": 1.0,
            "Negative minimum particle radius [m]": 0.0,
            "Negative maximum particle radius [m]": 1.0,
            "Positive minimum particle radius [m]": 0.0,
            "Positive maximum particle radius [m]": 1.0,
        }
    )

    if geometry is None:
        geometry = pybamm.battery_geometry(options={"particle size": "distribution"})
    param.process_geometry(geometry)

    submesh_types = {
        "negative electrode": pybamm.Uniform1DSubMesh,
        "separator": pybamm.Uniform1DSubMesh,
        "positive electrode": pybamm.Uniform1DSubMesh,
        "negative particle": pybamm.Uniform1DSubMesh,
        "positive particle": pybamm.Uniform1DSubMesh,
        "negative particle size": pybamm.Uniform1DSubMesh,
        "positive particle size": pybamm.Uniform1DSubMesh,
        "current collector": pybamm.SubMesh0D,
    }
    if cc_submesh:
        submesh_types["current collector"] = cc_submesh

    if xpts is None:
        xn_pts, xs_pts, xp_pts = 40, 25, 35
    else:
        xn_pts, xs_pts, xp_pts = xpts, xpts, xpts
    var_pts = {
        "x_n": xn_pts,
        "x_s": xs_pts,
        "x_p": xp_pts,
        "r_n": rpts,
        "r_p": rpts,
        "y": ypts,
        "z": zpts,
        "r_macro": rcellpts,
        "R_n": Rpts,
        "R_p": Rpts,
    }
    return pybamm.Mesh(geometry, submesh_types, var_pts)


def get_p2d_mesh_for_testing(xpts=None, rpts=10):
    geometry = pybamm.battery_geometry()
    return get_mesh_for_testing(xpts=xpts, rpts=rpts, geometry=geometry)


def get_size_distribution_mesh_for_testing(
    xpts=None,
    rpts=10,
    Rpts=10,
    zpts=15,
    cc_submesh=pybamm.Uniform1DSubMesh,
):
    options = {"particle size": "distribution", "dimensionality": 1}
    geometry = pybamm.battery_geometry(options=options)
    return get_mesh_for_testing(
        xpts=xpts,
        rpts=rpts,
        Rpts=Rpts,
        zpts=zpts,
        geometry=geometry,
        cc_submesh=cc_submesh,
    )


def get_1p1d_mesh_for_testing(
    xpts=None,
    rpts=10,
    zpts=15,
    cc_submesh=pybamm.Uniform1DSubMesh,
):
    geometry = pybamm.battery_geometry(options={"dimensionality": 1})
    return get_mesh_for_testing(
        xpts=xpts, rpts=rpts, zpts=zpts, geometry=geometry, cc_submesh=cc_submesh
    )


def get_2p1d_mesh_for_testing(
    xpts=None,
    rpts=10,
    ypts=15,
    zpts=15,
    include_particles=True,
    cc_submesh=pybamm.MeshGenerator(pybamm.ScikitUniform2DSubMesh),
):
    geometry = pybamm.battery_geometry(
        include_particles=include_particles, options={"dimensionality": 2}
    )
    return get_mesh_for_testing(
        xpts=xpts,
        rpts=rpts,
        ypts=ypts,
        zpts=zpts,
        geometry=geometry,
        cc_submesh=cc_submesh,
    )


def get_unit_2p1D_mesh_for_testing(ypts=15, zpts=15, include_particles=True):
    param = pybamm.ParameterValues(
        values={
            "Electrode width [m]": 1,
            "Electrode height [m]": 1,
            "Negative tab width [m]": 1,
            "Negative tab centre y-coordinate [m]": 0.5,
            "Negative tab centre z-coordinate [m]": 0,
            "Positive tab width [m]": 1,
            "Positive tab centre y-coordinate [m]": 0.5,
            "Positive tab centre z-coordinate [m]": 1,
            "Negative electrode thickness [m]": 0.3,
            "Separator thickness [m]": 0.3,
            "Positive electrode thickness [m]": 0.3,
        }
    )

    geometry = pybamm.battery_geometry(
        include_particles=include_particles, options={"dimensionality": 2}
    )
    param.process_geometry(geometry)

    var_pts = {"x_n": 3, "x_s": 3, "x_p": 3, "y": ypts, "z": zpts}

    submesh_types = {
        "negative electrode": pybamm.Uniform1DSubMesh,
        "separator": pybamm.Uniform1DSubMesh,
        "positive electrode": pybamm.Uniform1DSubMesh,
        "current collector": pybamm.MeshGenerator(pybamm.ScikitUniform2DSubMesh),
    }

    return pybamm.Mesh(geometry, submesh_types, var_pts)


def get_cylindrical_mesh_for_testing(
    xpts=10, rpts=10, rcellpts=15, include_particles=False
):
    geometry = pybamm.battery_geometry(
        include_particles=include_particles,
        options={"dimensionality": 1},
        form_factor="cylindrical",
    )
    return get_mesh_for_testing(
        xpts=xpts,
        rpts=rpts,
        rcellpts=rcellpts,
        geometry=geometry,
        cc_submesh=pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
    )


def get_discretisation_for_testing(
    xpts=None, rpts=10, mesh=None, cc_method=SpatialMethodForTesting
):
    if mesh is None:
        mesh = get_mesh_for_testing(xpts=xpts, rpts=rpts)
    spatial_methods = {
        "macroscale": SpatialMethodForTesting(),
        "negative particle": SpatialMethodForTesting(),
        "positive particle": SpatialMethodForTesting(),
        "negative particle size": SpatialMethodForTesting(),
        "positive particle size": SpatialMethodForTesting(),
        "current collector": cc_method(),
    }
    return pybamm.Discretisation(mesh, spatial_methods)


def get_p2d_discretisation_for_testing(xpts=None, rpts=10):
    return get_discretisation_for_testing(mesh=get_p2d_mesh_for_testing(xpts, rpts))


def get_size_distribution_disc_for_testing(xpts=None, rpts=10, Rpts=10, zpts=15):
    return get_discretisation_for_testing(
        mesh=get_size_distribution_mesh_for_testing(xpts, rpts, Rpts, zpts)
    )


def get_1p1d_discretisation_for_testing(xpts=None, rpts=10, zpts=15):
    return get_discretisation_for_testing(
        mesh=get_1p1d_mesh_for_testing(xpts, rpts, zpts),
        cc_method=pybamm.FiniteVolume,
    )


def get_2p1d_discretisation_for_testing(
    xpts=None, rpts=10, ypts=15, zpts=15, include_particles=True
):
    return get_discretisation_for_testing(
        mesh=get_2p1d_mesh_for_testing(xpts, rpts, ypts, zpts, include_particles),
        cc_method=pybamm.ScikitFiniteElement,
    )


def get_cylindrical_discretisation_for_testing(
    xpts=10, rpts=10, rcellpts=15, include_particles=False
):
    return get_discretisation_for_testing(
        mesh=get_cylindrical_mesh_for_testing(xpts, rpts, rcellpts, include_particles),
        cc_method=pybamm.FiniteVolume,
    )
