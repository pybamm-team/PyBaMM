import pybamm
import numpy as np


def time_create_expression():
    global model
    model = pybamm.BaseModel()
    global R
    R = pybamm.Parameter("Particle radius [m]")
    D = pybamm.Parameter("Diffusion coefficient [m2.s-1]")
    j = pybamm.Parameter("Interfacial current density [A.m-2]")
    F = pybamm.Parameter("Faraday constant [C.mol-1]")
    c0 = pybamm.Parameter("Initial concentration [mol.m-3]")

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


def setup_parameterise():
    time_create_expression()


def time_parameterise():

    global param
    param = pybamm.ParameterValues(
        {
            "Particle radius [m]": 10e-6,
            "Diffusion coefficient [m2.s-1]": 3.9e-14,
            "Interfacial current density [A.m-2]": 1.4,
            "Faraday constant [C.mol-1]": 96485,
            "Initial concentration [mol.m-3]": 2.5e4,
        }
    )
    global r
    r = pybamm.SpatialVariable(
        "r", domain=["negative particle"], coord_sys="spherical polar"
    )
    global geometry
    geometry = {"negative particle": {r: {"min": pybamm.Scalar(0), "max": R}}}
    param.process_model(model)
    param.process_geometry(geometry)


time_parameterise.setup = setup_parameterise


def setup_discretise():
    time_create_expression()
    time_parameterise()


def time_discretise():
    time_create_expression()
    time_parameterise()

    submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
    var_pts = {r: 20}
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

    spatial_methods = {"negative particle": pybamm.FiniteVolume()}
    disc = pybamm.Discretisation(mesh, spatial_methods)
    disc.process_model(model)


time_discretise.setup = setup_discretise


def setup_solve():
    time_create_expression()
    time_parameterise()
    time_discretise()


def time_solve():
    solver = pybamm.ScipySolver()
    t = np.linspace(0, 3600, 600)
    solver.solve(model, t)


time_solve.setup = setup_solve
