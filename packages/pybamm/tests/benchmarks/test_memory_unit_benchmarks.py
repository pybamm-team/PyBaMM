import numpy as np
import pytest

import pybamm

pytestmark = pytest.mark.memory_bench


def _create_expression():
    R = pybamm.Parameter("Particle radius [m]")
    D = pybamm.Parameter("Diffusion coefficient [m2.s-1]")
    j = pybamm.Parameter("Interfacial current density [A.m-2]")
    F = pybamm.Parameter("Faraday constant [C.mol-1]")
    c0 = pybamm.Parameter("Initial concentration [mol.m-3]")
    model = pybamm.BaseModel()

    c = pybamm.Variable("Concentration [mol.m-3]", domain="negative particle")
    N = -D * pybamm.grad(c)
    dcdt = -pybamm.div(N)
    model.rhs = {c: dcdt}

    lbc = pybamm.Scalar(0)
    rbc = -j / F / D
    model.boundary_conditions = {
        c: {"left": (lbc, "Neumann"), "right": (rbc, "Neumann")}
    }

    model.initial_conditions = {c: c0}
    model.variables = {
        "Concentration [mol.m-3]": c,
        "Surface concentration [mol.m-3]": pybamm.surf(c),
        "Flux [mol.m-2.s-1]": N,
    }
    return R, model


def _parameterise(R, model):
    param = pybamm.ParameterValues(
        {
            "Particle radius [m]": 10e-6,
            "Diffusion coefficient [m2.s-1]": 3.9e-14,
            "Interfacial current density [A.m-2]": 1.4,
            "Faraday constant [C.mol-1]": 96485,
            "Initial concentration [mol.m-3]": 2.5e4,
        }
    )

    r = pybamm.SpatialVariable(
        "r", domain=["negative particle"], coord_sys="spherical polar"
    )

    geometry = {"negative particle": {r: {"min": pybamm.Scalar(0), "max": R}}}
    param.process_model(model)
    param.process_geometry(geometry)
    return r, geometry


def _discretise(r, geometry, model):
    submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
    var_pts = {r: 20}
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

    spatial_methods = {"negative particle": pybamm.FiniteVolume()}
    disc = pybamm.Discretisation(mesh, spatial_methods)
    disc.process_model(model)


@pytest.mark.limit_memory("2 KB")
def test_create_expression_memory():
    _create_expression()


@pytest.mark.limit_memory("4 KB")
def test_parameterise_memory():
    R, model = _create_expression()
    _parameterise(R, model)


@pytest.mark.limit_memory("50 KB")
def test_discretise_memory():
    R, model = _create_expression()
    r, geometry = _parameterise(R, model)
    _discretise(r, geometry, model)


@pytest.mark.limit_memory("2.5 MB")
def test_solve_memory():
    R, model = _create_expression()
    r, geometry = _parameterise(R, model)
    _discretise(r, geometry, model)

    solver = pybamm.IDAKLUSolver()
    t_eval = np.linspace(0, 3600, 600)
    solver.solve(model, t_eval=t_eval)
