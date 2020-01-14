#
# Ex3: Adding parameters and additional output variables in PyBaMM
#

import pybamm
import numpy as np
import matplotlib.pyplot as plt

"--------------------------------------------------------------------------------------"
"Setting up the model"

# 1. Initialise an empty model ---------------------------------------------------------
model = pybamm.BaseModel()

# 2. Define parameters and variables ---------------------------------------------------
D = pybamm.Parameter("Diffusion coefficient")
c = pybamm.Variable("Concentration", domain="negative particle")


# 3. State governing equations ---------------------------------------------------------
N = -D * pybamm.grad(c)  # flux
dcdt = -pybamm.div(N)

model.rhs = {c: dcdt}  # add equation to rhs dictionary

# 4. State boundary conditions ---------------------------------------------------------
lbc = pybamm.Scalar(0)
rbc = pybamm.Scalar(2)
model.boundary_conditions = {c: {"left": (lbc, "Dirichlet"), "right": (rbc, "Neumann")}}

# 5. State initial conditions ----------------------------------------------------------
model.initial_conditions = {c: pybamm.Scalar(1)}


# 6. State output variables ------------------------------------------------------------
model.variables = {"Concentration": c, "Flux": N}

"--------------------------------------------------------------------------------------"
"Using the model"

# define geometry
r = pybamm.SpatialVariable(
    "r", domain=["negative particle"], coord_sys="spherical polar"
)
geometry = {
    "negative particle": {
        "primary": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
    }
}

# parameter values
param = pybamm.ParameterValues({"Diffusion coefficient": 0.5})

# process model and geometry
param.process_model(model)
param.process_geometry(geometry)

# mesh and discretise
submesh_types = {"negative particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)}
var_pts = {r: 20}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

spatial_methods = {"negative particle": pybamm.FiniteVolume()}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# solve
solver = pybamm.ScipySolver()
t = np.linspace(0, 1, 100)
solution = solver.solve(model, t)

# post-process, so that the solutions can be called at any time t or space r
# (using interpolation)
c = pybamm.ProcessedVariable(
    model.variables["Concentration"], solution.t, solution.y, mesh
)
N = pybamm.ProcessedVariable(model.variables["Flux"], solution.t, solution.y, mesh)

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(solution.t, c(solution.t, r=1))
ax1.set_xlabel("t")
ax1.set_ylabel("Surface concentration")
ax2.plot(solution.t, N(solution.t, r=0.5))
ax2.set_xlabel("t")
ax2.set_ylabel("Flux through r=0.5")
plt.tight_layout()
plt.show()
