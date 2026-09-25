import numpy as np
import pytest

import pybamm

pytestmark = pytest.mark.time_bench


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


def test_create_expression(benchmark):
    benchmark(_create_expression)


class TestTreeUtilityBenchmarks:
    @pytest.mark.parametrize("structure", ["tree", "dfn"])
    @pytest.mark.parametrize("matched", [False, True])
    @pytest.mark.parametrize("shared_cache", [False, True])
    def test_replace_symbols(self, benchmark, structure, matched, shared_cache):
        if structure == "tree":
            leaves = [pybamm.Parameter(f"p{index}") for index in range(4096)]
            target = leaves[0]
            while len(leaves) > 1:
                leaves = [
                    pybamm.Addition(leaves[index], leaves[index + 1])
                    for index in range(0, len(leaves), 2)
                ]
            roots = leaves
        else:
            model = pybamm.lithium_ion.DFN()
            roots = [
                *model.rhs.values(),
                *model.algebraic.values(),
                *model.variables.values(),
            ]
            target = pybamm.Parameter(
                "Maximum concentration in negative electrode [mol.m-3]"
            )
        if not matched:
            target = pybamm.Parameter("absent replacement target")
        mapping = {target: 1234}
        for root in roots:
            hash(root)

        def run():
            cache = {} if shared_cache else None
            return [pybamm.replace(root, mapping, cache=cache) for root in roots]

        benchmark(run)


def test_parameterise(benchmark):
    def setup():
        R, model = _create_expression()
        return (R, model), {}

    benchmark.pedantic(_parameterise, setup=setup, rounds=2000, iterations=1)


def test_discretise(benchmark):
    def setup():
        R, model = _create_expression()
        r, geometry = _parameterise(R, model)
        return (r, geometry, model), {}

    benchmark.pedantic(_discretise, setup=setup, rounds=2000, iterations=1)


def test_solve(benchmark):
    R, model = _create_expression()
    r, geometry = _parameterise(R, model)
    _discretise(r, geometry, model)

    def run():
        solver = pybamm.IDAKLUSolver()
        t = np.linspace(0, 3600, 600)
        solver.solve(model, t)

    benchmark(run)
