#
# Shared methods and classes for testing
#
import importlib.metadata as importlib_metadata
import re
import socket

from scipy.sparse import eye

import pybamm


class DummyDiscretisationClass:
    boundary_conditions = None


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


def get_mesh_for_testing_2d(
    xpts=None,
    rpts=10,
    Rpts=10,
    ypts=15,
    zpts=15,
):
    param = pybamm.ParameterValues(
        values={
            "Electrode height [m]": 1,
            "Negative electrode thickness [m]": 0.3333333333333333,
            "Separator thickness [m]": 0.3333333333333333,
            "Positive electrode thickness [m]": 0.3333333333333334,
            "Negative particle radius [m]": 0.5,
            "Positive particle radius [m]": 0.5,
        }
    )

    x = pybamm.SpatialVariable(
        "x", ["negative electrode", "separator", "positive electrode"], direction="lr"
    )
    z = pybamm.SpatialVariable(
        "z", ["negative electrode", "separator", "positive electrode"], direction="tb"
    )
    r_n = pybamm.SpatialVariable("r_n", ["negative particle"])
    r_p = pybamm.SpatialVariable("r_p", ["positive particle"])

    geometry = {
        "negative electrode": {
            x: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Parameter("Negative electrode thickness [m]"),
            },
            z: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Parameter("Electrode height [m]"),
            },
        },
        "separator": {
            x: {
                "min": pybamm.Parameter("Negative electrode thickness [m]"),
                "max": pybamm.Parameter("Separator thickness [m]")
                + pybamm.Parameter("Negative electrode thickness [m]"),
            },
            z: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Parameter("Electrode height [m]"),
            },
        },
        "positive electrode": {
            x: {
                "min": pybamm.Parameter("Separator thickness [m]")
                + pybamm.Parameter("Negative electrode thickness [m]"),
                "max": pybamm.Parameter("Positive electrode thickness [m]")
                + pybamm.Parameter("Separator thickness [m]")
                + pybamm.Parameter("Negative electrode thickness [m]"),
            },
            z: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Parameter("Electrode height [m]"),
            },
        },
        "negative particle": {
            r_n: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Scalar(1),
            },
        },
        "positive particle": {
            r_p: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Scalar(1),
            },
        },
    }
    param.process_geometry(geometry)

    submesh_types = {
        "negative electrode": pybamm.Uniform2DSubMesh,
        "separator": pybamm.Uniform2DSubMesh,
        "positive electrode": pybamm.Uniform2DSubMesh,
        "negative particle": pybamm.Uniform1DSubMesh,
        "positive particle": pybamm.Uniform1DSubMesh,
    }

    if xpts is None:
        xn_pts = 40
    else:
        xn_pts = xpts
    var_pts = {
        x: xn_pts,
        z: zpts,
        r_n: rpts,
        r_p: rpts,
    }
    return pybamm.Mesh(geometry, submesh_types, var_pts)


def get_unit_3d_mesh_for_testing(geom_type="pouch", **geom_params):
    if geom_type == "pouch":
        x = pybamm.SpatialVariable("x", ["current collector"])
        y = pybamm.SpatialVariable("y", ["current collector"])
        z = pybamm.SpatialVariable("z", ["current collector"])
        x_max = geom_params.get("x_max", 1.0)
        y_max = geom_params.get("y_max", 2.0)
        z_max = geom_params.get("z_max", 3.0)
        geometry = {
            "current collector": {
                x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(x_max)},
                y: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(y_max)},
                z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(z_max)},
            }
        }
        var_pts = {x: 5, y: 5, z: 5}
    elif geom_type == "cylinder":
        r = pybamm.SpatialVariable(
            "r", ["current collector"], coord_sys="cylindrical polar"
        )
        z = pybamm.SpatialVariable(
            "z", ["current collector"], coord_sys="cylindrical polar"
        )
        radius = geom_params.get("radius", 1.0)
        height = geom_params.get("height", 1.0)
        r_inner = geom_params.get("r_inner", 0.0)
        geometry = {
            "current collector": {
                r: {"min": pybamm.Scalar(r_inner), "max": pybamm.Scalar(radius)},
                z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(height)},
            }
        }
        var_pts = {r: 5, z: 5}
    else:
        raise ValueError(f"geom_type '{geom_type}' not recognised")

    generator_params = {"h": 0.2}
    generator_params.update(geom_params)
    generator = pybamm.ScikitFemGenerator3D(geom_type, **generator_params)
    submesh_types = {"current collector": generator}
    return pybamm.Mesh(geometry, submesh_types, var_pts)


def get_3d_mesh_for_testing(
    xpts=5, ypts=5, zpts=5, geom_type="pouch", include_particles=False, **geom_params
):
    param = pybamm.ParameterValues(
        {
            "Electrode width [m]": 1.0,
            "Electrode height [m]": 1.0,
            "Electrode depth [m]": 1.0,
            "Negative electrode thickness [m]": 1 / 3,
            "Separator thickness [m]": 1 / 3,
            "Positive electrode thickness [m]": 1 / 3,
        }
    )

    x = pybamm.SpatialVariable(
        "x", ["negative electrode", "separator", "positive electrode"]
    )
    y = pybamm.SpatialVariable(
        "y", ["negative electrode", "separator", "positive electrode"]
    )
    z = pybamm.SpatialVariable(
        "z", ["negative electrode", "separator", "positive electrode"]
    )

    geometry = {
        "negative electrode": {
            x: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Parameter("Negative electrode thickness [m]"),
            },
            y: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Parameter("Electrode width [m]"),
            },
            z: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Parameter("Electrode height [m]"),
            },
        },
        "separator": {
            x: {
                "min": pybamm.Parameter("Negative electrode thickness [m]"),
                "max": pybamm.Parameter("Separator thickness [m]")
                + pybamm.Parameter("Negative electrode thickness [m]"),
            },
            y: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Parameter("Electrode width [m]"),
            },
            z: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Parameter("Electrode height [m]"),
            },
        },
        "positive electrode": {
            x: {
                "min": pybamm.Parameter("Separator thickness [m]")
                + pybamm.Parameter("Negative electrode thickness [m]"),
                "max": pybamm.Parameter("Positive electrode thickness [m]")
                + pybamm.Parameter("Separator thickness [m]")
                + pybamm.Parameter("Negative electrode thickness [m]"),
            },
            y: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Parameter("Electrode width [m]"),
            },
            z: {
                "min": pybamm.Scalar(0),
                "max": pybamm.Parameter("Electrode height [m]"),
            },
        },
    }
    param.process_geometry(geometry)

    # Create generator with parameters
    generator_params = {"h": 0.2}
    generator_params.update(geom_params)

    generator = pybamm.ScikitFemGenerator3D(geom_type, **generator_params)

    submesh_types = {
        "negative electrode": generator,
        "separator": generator,
        "positive electrode": generator,
    }

    var_pts = {x: xpts, y: ypts, z: zpts}

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
    cc_submesh=None,
):
    if cc_submesh is None:
        cc_submesh = pybamm.MeshGenerator(pybamm.ScikitUniform2DSubMesh)
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


def function_test(arg):
    return arg + arg


def multi_var_function_test(arg1, arg2):
    return arg1 + arg2


def multi_var_function_cube_test(arg1, arg2):
    return arg1 + arg2**3


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


def get_base_model_with_battery_geometry(**kwargs):
    model = pybamm.BaseModel()
    model._geometry = pybamm.battery_geometry(**kwargs)
    return model


def get_required_distribution_deps(package_name):
    pattern = re.compile(r"(?!.*extra\b)^([^<>=;\[]+)\b.*$")
    if json_deps := importlib_metadata.metadata(package_name).json.get("requires_dist"):
        return {m.group(1) for dep_name in json_deps if (m := pattern.match(dep_name))}
    return set()


def get_optional_distribution_deps(package_name):
    pattern = re.compile(rf"(?!.*{package_name}\b|.*docs\b|.*dev\b)^([^<>=;\[]+)\b.*$")
    if json_deps := importlib_metadata.metadata(package_name).json.get("requires_dist"):
        return {
            m.group(1)
            for dep_name in json_deps
            if (m := pattern.match(dep_name)) and "extra" in m.group(0)
        }
    return set()


def get_present_optional_import_deps(package_name, optional_distribution_deps=None):
    if optional_distribution_deps is None:
        optional_distribution_deps = get_optional_distribution_deps(package_name)

    present_optional_import_deps = set()
    for (
        import_pkg,
        distribution_pkgs,
    ) in importlib_metadata.packages_distributions().items():
        if any(dep in optional_distribution_deps for dep in distribution_pkgs):
            present_optional_import_deps.add(import_pkg)
    return present_optional_import_deps


def no_internet_connection():
    try:
        host = socket.gethostbyname("www.github.com")
        conn = socket.create_connection((host, 80), 2)
        conn.close()
        return False
    except (socket.gaierror, TimeoutError):
        return True


def assert_domain_equal(a, b):
    """Check that two domains are equal, ignoring empty domains"""
    a_dict = {k: v for k, v in a.items() if v != []}
    b_dict = {k: v for k, v in b.items() if v != []}
    assert a_dict == b_dict


def get_mesh_for_testing_symbolic():
    submesh_types = {"domain": pybamm.SymbolicUniform1DSubMesh}
    geometry = {
        "domain": {"x": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(2)}},
    }
    var_pts = {"x": 15}
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
    return mesh


def get_mesh_for_testing_symbolic_concatenation():
    submesh_types = {
        "domain 1": pybamm.SymbolicUniform1DSubMesh,
        "domain 2": pybamm.SymbolicUniform1DSubMesh,
    }
    geometry = {
        "domain 1": {"x": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(2)}},
        "domain 2": {"x": {"min": pybamm.Scalar(2), "max": pybamm.Scalar(4)}},
    }
    var_pts = {"x": 15}
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
    return mesh


def get_spherical_mesh_for_testing_symbolic():
    submesh_types = {"spherical domain": pybamm.SymbolicUniform1DSubMesh}
    geometry = {
        "spherical domain": {"r_n": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(2)}},
    }
    var_pts = {"r_n": 15}
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
    return mesh


def get_cylindrical_mesh_for_testing_symbolic():
    submesh_types = {"cylindrical domain": pybamm.SymbolicUniform1DSubMesh}
    cylindrical_r = pybamm.SpatialVariable(
        "r", ["cylindrical domain"], coord_sys="cylindrical polar"
    )
    geometry = {
        "cylindrical domain": {
            cylindrical_r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(2)}
        },
    }
    var_pts = {cylindrical_r: 15}
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
    return mesh
