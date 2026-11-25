import json

import numpy as np

import pybamm
from pybamm.expression_tree.operations.serialise import Serialise
from pybamm.meshes.unstructured_submesh import _hex_grid


def _box(x_edges):
    """Tagged hex box on [x_edges] x [0, 1] x [0, 1]."""
    submesh = pybamm.UnstructuredSubMesh(
        *_hex_grid(x_edges, np.linspace(0, 1, 3), np.linspace(0, 1, 4))
    )
    submesh.detect_box_boundaries()
    top = submesh.boundary_faces.pop("top")
    tab = submesh.face_centroids[top, 1] < 0.5
    submesh.boundary_faces["top"] = top[~tab]
    submesh.boundary_faces["tab_top"] = top[tab]
    return submesh


def _model():
    """Two prebuilt 3D domains and one generated 1D domain, all with custom
    spatial variables, carrying their discretisation as model defaults."""
    x_a = pybamm.SpatialVariable("x_a", domain="negative electrode", direction="lr")
    y_a = pybamm.SpatialVariable("y_a", domain="negative electrode", direction="fb")
    z_a = pybamm.SpatialVariable("z_a", domain="negative electrode", direction="tb")
    x_b = pybamm.SpatialVariable("x_b", domain="separator", direction="lr")
    y_b = pybamm.SpatialVariable("y_b", domain="separator", direction="fb")
    z_b = pybamm.SpatialVariable("z_b", domain="separator", direction="tb")
    r = pybamm.SpatialVariable(
        "r_custom", domain="negative particle", coord_sys="spherical polar"
    )

    model = pybamm.BaseModel()
    c_a = pybamm.Variable("c_a", domain="negative electrode")
    c_b = pybamm.Variable("c_b", domain="separator")
    c = pybamm.concatenation(c_a, c_b)
    source = pybamm.concatenation(
        pybamm.FunctionParameter("Source", {"x": x_a}),
        pybamm.PrimaryBroadcast(0, "separator"),
    )
    diffusivity = pybamm.Parameter("Diffusivity")
    model.rhs[c] = pybamm.div(diffusivity * pybamm.grad(c)) + source
    zero = (pybamm.Scalar(0), "Neumann")
    model.boundary_conditions[c] = {
        "left": zero,
        "right": zero,
        "front": zero,
        "back": zero,
        "bottom": zero,
        "top": zero,
        "tab_top": (pybamm.Scalar(1), "Dirichlet"),
    }
    model.initial_conditions[c] = pybamm.Scalar(2)

    q = pybamm.Variable("q", domain="negative particle")
    model.rhs[q] = pybamm.div(pybamm.grad(q))
    model.boundary_conditions[q] = {
        "left": zero,
        "right": (pybamm.Scalar(-1), "Neumann"),
    }
    model.initial_conditions[q] = pybamm.Scalar(1)
    model.variables = {
        "Mean concentration": pybamm.x_average(c_a) + pybamm.x_average(c_b),
        "Maximum concentration": pybamm.Max(c),
        "Surface q": pybamm.surf(q),
    }

    model._default_geometry = {
        "negative electrode": {
            x_a: {"min": 0, "max": 1},
            y_a: {"min": 0, "max": 1},
            z_a: {"min": 0, "max": 1},
        },
        "separator": {
            x_b: {"min": 1, "max": 1.5},
            y_b: {"min": 0, "max": 1},
            z_b: {"min": 0, "max": 1},
        },
        "negative particle": {r: {"min": 0, "max": 1}},
    }
    model._default_submesh_types = {
        "negative electrode": _box(np.linspace(0, 1, 4)),
        "separator": _box(np.linspace(1, 1.5, 3)),
        "negative particle": pybamm.Uniform1DSubMesh,
    }
    model._default_spatial_methods = {
        "negative electrode": pybamm.FiniteVolumeUnstructured(),
        "separator": pybamm.FiniteVolumeUnstructured(),
        "negative particle": pybamm.FiniteVolume(),
    }
    model._default_var_pts = {x_a: 3, y_a: 2, z_a: 3, x_b: 2, y_b: 2, z_b: 3, r: 7}
    return model


def _solve(model):
    parameter_values = pybamm.ParameterValues(
        {"Diffusivity": 0.1, "Source": lambda x: 1 + x}
    )
    simulation = pybamm.Simulation(model, parameter_values=parameter_values)
    return simulation.solve([0, 3])


class TestUnstructuredModelRoundTrip:
    def test_loaded_model_solves_like_original(self):
        expected = _solve(_model())

        payload = json.loads(json.dumps(Serialise.serialise_custom_model(_model())))
        loaded = Serialise.load_custom_model(payload)
        solution = _solve(loaded)

        t = np.linspace(0, 3, 7)
        for name in ("Mean concentration", "Maximum concentration", "Surface q"):
            np.testing.assert_allclose(
                solution[name](t), expected[name](t), rtol=1e-10, atol=1e-12
            )
        # A solve that ignored the tab or the source would not move from 2
        assert not np.allclose(expected["Mean concentration"](t), 4)
