import numpy as np
import matplotlib.pyplot as plt
import pybamm
from pybamm import (
    BaseModel,
    Variable,
    grad,
    div,
    SpatialVariable,
    Scalar,
    Discretisation,
    FiniteVolume3D,
)

Lx, Ly, Lz = 1.0, 0.8, 0.6
kappa = 0.1
t_max = 0.2


def Q_function(x, y, z):
    return 10.0 * np.exp(
        -5.0 * ((x - Lx / 2) ** 2 + (y - Ly / 2) ** 2 + (z - Lz / 2) ** 2)
    )


model = BaseModel()

x = SpatialVariable("x", ["prism"], coord_sys="cartesian", direction="x")
y = SpatialVariable("y", ["prism"], coord_sys="cartesian", direction="y")
z = SpatialVariable("z", ["prism"], coord_sys="cartesian", direction="z")

T = Variable("T", domain=["prism"])


def Q_eval(var):
    return 10 * pybamm.exp(
        -5 * ((x - Lx / 2) ** 2 + (y - Ly / 2) ** 2 + (z - Lz / 2) ** 2)
    )


Q = Q_eval(T)

model.rhs = {T: kappa * div(grad(T)) + Q}

model.initial_conditions = {T: Scalar(0)}

bcs = {
    T: {
        "left": (Scalar(0), "Dirichlet"),
        "right": (Scalar(0), "Neumann"),
        "negative electrode": None,
    }
}

model.boundary_conditions = {
    T: {
        ("x", "left"): (Scalar(0), "Dirichlet"),
        ("x", "right"): (Scalar(0), "Neumann"),
        ("y", "left"): (Scalar(100), "Dirichlet"),
        ("y", "right"): (Scalar(0), "Neumann"),
        ("z", "left"): (Scalar(50), "Dirichlet"),
        ("z", "right"): (Scalar(0), "Neumann"),
    }
}

Nx, Ny, Nz = 16, 16, 16

geometry = {
    "prism": {
        x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(Lx)},
        y: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(Ly)},
        z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(Lz)},
    }
}
submesh_types = {"prism": pybamm.Uniform3DSubMesh}
var_pts = {x: Nx, y: Ny, z: Nz}

mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

spatial_methods = {"prism": FiniteVolume3D()}
disc = Discretisation(mesh, spatial_methods)
disc.set_variable_slices([T])
disc.process_model(model)

t_eval = np.linspace(0, t_max, 50)
solver = pybamm.CasadiSolver()
sim = pybamm.Simulation(model, mesh=mesh, solver=solver)
sim.solve(t_eval)

t_sol = sim.solution["T"].entries

submesh = mesh["prism"]
nodes = submesh.nodes
coords = nodes.reshape((Nx, Ny, Nz, 3))

T_end = t_sol.reshape((Nx, Ny, Nz))

mid_j = Ny // 2
X_plane = coords[:, mid_j, :, 0]
Z_plane = coords[:, mid_j, :, 2]
T_plane = T_end[:, mid_j, :]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 5))
surf = ax.plot_surface(
    X_plane, Z_plane, T_plane, cmap="inferno", edgecolor="none", rcount=60, ccount=60
)
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.set_zlabel("T [°C]")
ax.set_title(f"T(x,z) at y = {Ly / 2:.2f},  t = {t_max:.2f}\n")
fig.colorbar(surf, ax=ax, shrink=0.5, label="Temperature")
plt.tight_layout()
plt.show()

center_idx = (Nx // 2, Ny // 2, Nz // 2)
corner_idx = (0, 0, 0)
print("T(center) ≈", T_end[center_idx], "°C")
print("T(corner) ≈", T_end[corner_idx], "°C")
