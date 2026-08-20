from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

import pybamm

pybamm.set_logging_level("DEBUG")

model = pybamm.BaseModel()

R = pybamm.Parameter("Particle radius")
D = pybamm.Parameter("Diffusion coefficient")
c0 = pybamm.Parameter("Initial concentration")

c = pybamm.Variable("Concentration", domain="negative particle")

# governing equations
N = -D * pybamm.grad(c)  # flux
N.set_do_not_simplify()

dcdt = -pybamm.div(N)
model.rhs = {c: dcdt}


# initial conditions
model.initial_conditions = {c: c0}

model.variables = {
    "Concentration": c,
    "Flux": N,
    "Surface concentration": pybamm.surf(c),
}

model2 = deepcopy(model)

# boundary conditions
lbc = pybamm.Scalar(0)
rbc = pybamm.Scalar(1)
model.boundary_conditions = {
    c: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (pybamm.Scalar(-2), ("Flux", N)),
    }
}

param = pybamm.ParameterValues(
    {
        "Particle radius": 10e-6,
        "Diffusion coefficient": 3.9e-14,
        "Initial concentration": 2.5e4,
    }
)

r = pybamm.SpatialVariable(
    "r", domain=["negative particle"], coord_sys="spherical polar"
)
geometry = {"negative particle": {r: {"min": pybamm.Scalar(0), "max": R}}}


param.process_model(model)
param.process_geometry(geometry)

submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
var_pts = {r: 20}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

spatial_methods = {"negative particle": pybamm.FiniteVolume()}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# solve
solver = pybamm.ScipySolver()
t = np.linspace(0, 3600, 600)
solution = solver.solve(model, t)

# post-process, so that the solution can be called at any time t or spaceå r
# (using interpolation)
c_sol = solution["Concentration"]
c_surf = solution["Surface concentration"]

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(solution.t, c_surf(solution.t))
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Surface concentration")

r = mesh["negative particle"].nodes  # radial position
time = 1000  # time in seconds
ax2.plot(r * 1e6, c_sol(t=time, r=r), label=f"t={time}[s]")
ax2.set_xlabel("Particle radius")
ax2.set_ylabel("Concentration")


# Comparison with Neumann
model2.boundary_conditions = {
    c: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (pybamm.Scalar(2) / D, "Neumann"),
    }
}

param.process_model(model2)
disc.process_model(model2)

# solve
solver = pybamm.ScipySolver()
t = np.linspace(0, 3600, 600)
solution = solver.solve(model2, t)

# post-process, so that the solution can be called at any time t or spaceå r
# (using interpolation)
c = solution["Concentration"]
c_surf = solution["Surface concentration"]

# Plot
ax1.plot(solution.t, c_surf(solution.t), "--")
ax2.plot(r * 1e6, c(t=time, r=r), "--", label=f"t={time}[s]")
ax2.legend()

plt.tight_layout()
plt.show()
