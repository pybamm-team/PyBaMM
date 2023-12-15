#
# Solve the transient heat equation with a spatially-dependent source term
#

import pybamm
import numpy as np
import matplotlib.pyplot as plt

# Numerical solution ----------------------------------------------------------

# Start with a base model
model = pybamm.BaseModel()

# Define the variables and parameters
# Note: we need to define the spatial variable x here too, so we can use it
# to write down the source term
x = pybamm.SpatialVariable("x", domain="rod", coord_sys="cartesian")
T = pybamm.Variable("Temperature", domain="rod")
k = pybamm.Parameter("Thermal diffusivity")

# Write the governing equations
N = -k * pybamm.grad(T)  # Heat flux
Q = 1 - pybamm.Function(np.abs, x - 1)  # Source term
dTdt = -pybamm.div(N) + Q
model.rhs = {T: dTdt}  # add to model

# Add the boundary and initial conditions
model.boundary_conditions = {
    T: {
        "left": (pybamm.Scalar(0), "Dirichlet"),
        "right": (pybamm.Scalar(0), "Dirichlet"),
    }
}
model.initial_conditions = {T: 2 * x - x**2}

# Add desired output variables
model.variables = {"Temperature": T, "Heat flux": N, "Heat source": Q}


# Define geometry
geometry = {"rod": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(2)}}}

# Set parameter values
param = pybamm.ParameterValues({"Thermal diffusivity": 0.75})

# Process model and geometry
param.process_model(model)
param.process_geometry(geometry)

# Pick mesh, spatial method, and discretise
submesh_types = {"rod": pybamm.Uniform1DSubMesh}
var_pts = {x: 30}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
spatial_methods = {"rod": pybamm.FiniteVolume()}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# Solve
solver = pybamm.ScipySolver()
t = np.linspace(0, 1, 100)
solution = solver.solve(model, t)

# Extract output variables
T_out = solution["Temperature"]

# Exact solution -------------------------------------------------------
N = 100  # number of Fourier modes to sum
k_val = param["Thermal diffusivity"]  # extract value of diffusivity


# Fourier coefficients
def q(n):
    return (8 / (n**2 * np.pi**2)) * np.sin(n * np.pi / 2)


def c(n):
    return (16 / (n**3 * np.pi**3)) * (1 - np.cos(n * np.pi))


def b(n):
    return c(n) - 4 * q(n) / (k_val * n**2 * np.pi**2)


def T_n(t, n):
    return (4 * q(n) / (k_val * n**2 * np.pi**2)) + b(n) * np.exp(
        -k_val * (n * np.pi / 2) ** 2 * t
    )


# Sum series to get the source term
def Q_exact(n):
    out = 0
    for n in np.arange(1, N):
        out += q(n) * np.sin(n * np.pi * x / 2)
    return out


# Sum series to get the temperature
def T_exact(x, t):
    out = 0
    for n in np.arange(1, N):
        out += T_n(t, n) * np.sin(n * np.pi * x / 2)
    return out


# Plot ------------------------------------------------------------------------
x_nodes = mesh["rod"].nodes  # numerical gridpoints
xx = np.linspace(0, 2, 101)  # fine mesh to plot exact solution
plot_times = np.linspace(0, 1, 5)

plt.figure(figsize=(15, 8))
cmap = plt.get_cmap("inferno")
for i, t in enumerate(plot_times):
    color = cmap(float(i) / len(plot_times))
    plt.plot(
        x_nodes,
        T_out(t, x=x_nodes),
        "o",
        color=color,
        label="Numerical" if i == 0 else "",
    )
    plt.plot(
        xx, T_exact(xx, t), "-", color=color, label=f"Exact (t={plot_times[i]})"
    )
plt.xlabel("x", fontsize=16)
plt.ylabel("T", fontsize=16)
plt.legend()
plt.show()
