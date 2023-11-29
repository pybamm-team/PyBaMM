# This script is intended to be a stripped back version of the
# 'docs/source/examples/notebooks/create-model.ipnb' so for more details please see
# that notebook

import pybamm
import numpy as np
import matplotlib.pyplot as plt

# 1. Initialise model ------------------------------------------------------------------
model = pybamm.BaseModel()

# 2. Define parameters and variables ---------------------------------------------------
# dimensional parameters
k_dim = pybamm.Parameter("Reaction rate constant")
L_0_dim = pybamm.Parameter("Initial thickness")
V_hat_dim = pybamm.Parameter("Partial molar volume")
c_inf_dim = pybamm.Parameter("Bulk electrolyte solvent concentration")


def D_dim(cc):
    return pybamm.FunctionParameter("Diffusivity", {"Concentration [mol.m-3]": cc})


# dimensionless parameters
k = k_dim * L_0_dim / D_dim(c_inf_dim)
V_hat = V_hat_dim * c_inf_dim


def D(cc):
    c_dim = c_inf_dim * cc
    return D_dim(c_dim) / D_dim(c_inf_dim)


# variables
x = pybamm.SpatialVariable("x", domain="SEI layer", coord_sys="cartesian")
c = pybamm.Variable("Solvent concentration", domain="SEI layer")
L = pybamm.Variable("SEI thickness")

# 3. State governing equations ---------------------------------------------------------
R = k * pybamm.BoundaryValue(c, "left")  # SEI reaction flux
N = -(1 / L) * D(c) * pybamm.grad(c)  # solvent flux
dcdt = (V_hat * R) * pybamm.inner(x / L, pybamm.grad(c)) - (1 / L) * pybamm.div(
    N
)  # solvent concentration governing equation
dLdt = V_hat * R  # SEI thickness governing equation

model.rhs = {c: dcdt, L: dLdt}  # add to model

# 4. State boundary conditions ---------------------------------------------------------
D_left = pybamm.BoundaryValue(
    D(c), "left"
)  # pybamm requires BoundaryValue(D(c)) and not D(BoundaryValue(c))
grad_c_left = L * R / D_left  # left bc
c_right = pybamm.Scalar(1)  # right bc

# add to model
model.boundary_conditions = {
    c: {"left": (grad_c_left, "Neumann"), "right": (c_right, "Dirichlet")}
}

# 5. State initial conditions ----------------------------------------------------------
model.initial_conditions = {c: pybamm.Scalar(1), L: pybamm.Scalar(1)}

# 6. State output variables ------------------------------------------------------------
model.variables = {
    "SEI thickness": L,
    "SEI growth rate": dLdt,
    "Solvent concentration": c,
    "SEI thickness [m]": L_0_dim * L,
    "SEI growth rate [m.s-1]": (D_dim(c_inf_dim) / L_0_dim) * dLdt,
    "Solvent concentration [mol.m-3]": c_inf_dim * c,
}

"--------------------------------------------------------------------------------------"
"Using the model"

# define geometry
geometry = {"SEI layer": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}


# diffusivity function
def Diffusivity(cc):
    return cc * 10 ** (-5)


# parameter values (not physically based, for example only!)
param = pybamm.ParameterValues(
    {
        "Reaction rate constant": 20,
        "Initial thickness": 1e-6,
        "Partial molar volume": 10,
        "Bulk electrolyte solvent concentration": 1,
        "Diffusivity": Diffusivity,
    }
)

# process model and geometry
param.process_model(model)
param.process_geometry(geometry)

# mesh and discretise
submesh_types = {"SEI layer": pybamm.Uniform1DSubMesh}
var_pts = {x: 50}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

spatial_methods = {"SEI layer": pybamm.FiniteVolume()}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# solve
solver = pybamm.ScipySolver()
t = np.linspace(0, 1, 100)
solution = solver.solve(model, t)

# Extract output variables
L_out = solution["SEI thickness [m]"]

# plot
plt.plot(solution.t, L_out(solution.t))
plt.xlabel("Time")
plt.ylabel("SEI thickness")
plt.show()
