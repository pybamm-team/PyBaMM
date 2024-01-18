import pybamm
import numpy as np
from benchmarks.benchmark_utils import set_random_seed


class TimeCreateExpression:
    R: pybamm.Parameter
    model: pybamm.BaseModel

    def setup(self):
        set_random_seed()

    def time_create_expression(self):
        self.R = pybamm.Parameter("Particle radius [m]")
        D = pybamm.Parameter("Diffusion coefficient [m2.s-1]")
        j = pybamm.Parameter("Interfacial current density [A.m-2]")
        F = pybamm.Parameter("Faraday constant [C.mol-1]")
        c0 = pybamm.Parameter("Initial concentration [mol.m-3]")
        self.model = pybamm.BaseModel()

        c = pybamm.Variable("Concentration [mol.m-3]", domain="negative particle")
        N = -D * pybamm.grad(c)
        dcdt = -pybamm.div(N)
        self.model.rhs = {c: dcdt}

        lbc = pybamm.Scalar(0)
        rbc = -j / F / D
        self.model.boundary_conditions = {
            c: {"left": (lbc, "Neumann"), "right": (rbc, "Neumann")}
        }

        self.model.initial_conditions = {c: c0}
        self.model.variables = {
            "Concentration [mol.m-3]": c,
            "Surface concentration [mol.m-3]": pybamm.surf(c),
            "Flux [mol.m-2.s-1]": N,
        }


class TimeParameteriseModel(TimeCreateExpression):
    r: pybamm.SpatialVariable
    geometry: dict

    def setup(self):
        set_random_seed()
        TimeCreateExpression.time_create_expression(self)

    def time_parameterise(self):
        param = pybamm.ParameterValues(
            {
                "Particle radius [m]": 10e-6,
                "Diffusion coefficient [m2.s-1]": 3.9e-14,
                "Interfacial current density [A.m-2]": 1.4,
                "Faraday constant [C.mol-1]": 96485,
                "Initial concentration [mol.m-3]": 2.5e4,
            }
        )

        self.r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )

        self.geometry = {
            "negative particle": {self.r: {"min": pybamm.Scalar(0), "max": self.R}}
        }
        param.process_model(self.model)
        param.process_geometry(self.geometry)


class TimeDiscretiseModel(TimeParameteriseModel):
    def setup(self):
        set_random_seed()
        TimeCreateExpression.time_create_expression(self)
        TimeParameteriseModel.time_parameterise(self)

    def time_discretise(self):
        TimeCreateExpression.time_create_expression(self)
        TimeParameteriseModel.time_parameterise(self)
        submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
        var_pts = {self.r: 20}
        mesh = pybamm.Mesh(self.geometry, submesh_types, var_pts)

        spatial_methods = {"negative particle": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(self.model)


class TimeSolveModel(TimeDiscretiseModel):
    def setup(self):
        set_random_seed()
        TimeCreateExpression.time_create_expression(self)
        TimeParameteriseModel.time_parameterise(self)
        TimeDiscretiseModel.time_discretise(self)

    def time_solve(self):
        solver = pybamm.ScipySolver()
        t = np.linspace(0, 3600, 600)
        solver.solve(self.model, t)
