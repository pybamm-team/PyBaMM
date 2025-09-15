import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

import pybamm
from pybamm import (
    BaseModel,
    Discretisation,
    Scalar,
    ScikitFiniteElement3D,
    SpatialVariable,
)

Lx, Ly, Lz = 1.0, 0.8, 0.6
alpha = 0.1
t_max = 0.2

model = BaseModel()

x = SpatialVariable("x", ["current collector"], coord_sys="cartesian")
y = SpatialVariable("y", ["current collector"], coord_sys="cartesian")
z = SpatialVariable("z", ["current collector"], coord_sys="cartesian")

T = pybamm.Variable("T", domain="current collector")
model.variables = {"T": T}

model.algebraic = {T: alpha * pybamm.laplacian(T) - pybamm.source(0, T)}

model.initial_conditions = {T: Scalar(0)}

model.boundary_conditions = {
    T: {
        "x_min": (Scalar(100), "Dirichlet"),
        "x_max": (Scalar(0), "Dirichlet"),
        "y_min": (Scalar(0), "Neumann"),
        "y_max": (Scalar(0), "Neumann"),
        "z_min": (Scalar(0), "Neumann"),
        "z_max": (Scalar(0), "Neumann"),
    }
}

geometry = {
    "current collector": {
        x: {"min": Scalar(0), "max": Scalar(Lx)},
        y: {"min": Scalar(0), "max": Scalar(Ly)},
        z: {"min": Scalar(0), "max": Scalar(Lz)},
    }
}

submesh_types = {
    "current collector": pybamm.ScikitFemGenerator3D(geom_type="pouch", h=0.15)
}
var_pts = {x: None, y: None, z: None}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

spatial_methods = {"current collector": ScikitFiniteElement3D()}
disc = Discretisation(mesh, spatial_methods)
disc.process_model(model)

solver = pybamm.AlgebraicSolver()
solution = solver.solve(model)

nodes = mesh["current collector"].nodes
T_solution = solution.y.flatten()

print(f"Number of FEM nodes: {nodes.shape[0]}")
print(f"Temperature range: [{T_solution.min():.2f}, {T_solution.max():.2f}]°C")

Nx_vis, Ny_vis, Nz_vis = 25, 20, 15
x_grid = np.linspace(0, Lx, Nx_vis)
y_grid = np.linspace(0, Ly, Ny_vis)
z_grid = np.linspace(0, Lz, Nz_vis)
X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
T_interp_3d = griddata(
    nodes, T_solution, grid_points, method="linear", fill_value=0
).reshape((Nx_vis, Ny_vis, Nz_vis))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
mid_y = Ny_vis // 2
mid_z = Nz_vis // 2
mid_x = Nx_vis // 2

im0 = axes[0].contourf(
    X[:, mid_y, :], Z[:, mid_y, :], T_interp_3d[:, mid_y, :], levels=20, cmap="inferno"
)
axes[0].set_xlabel("x [m]")
axes[0].set_ylabel("z [m]")
axes[0].set_title(f"T(x,z) at y={y_grid[mid_y]:.2f}")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].contourf(
    X[:, :, mid_z], Y[:, :, mid_z], T_interp_3d[:, :, mid_z], levels=20, cmap="inferno"
)
axes[1].set_xlabel("x [m]")
axes[1].set_ylabel("y [m]")
axes[1].set_title(f"T(x,y) at z={z_grid[mid_z]:.2f}")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].contourf(
    Y[mid_x, :, :], Z[mid_x, :, :], T_interp_3d[mid_x, :, :], levels=20, cmap="inferno"
)
axes[2].set_xlabel("y [m]")
axes[2].set_ylabel("z [m]")
axes[2].set_title(f"T(y,z) at x={x_grid[mid_x]:.2f}")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(
    X[:, mid_y, :],
    Z[:, mid_y, :],
    T_interp_3d[:, mid_y, :],
    cmap="inferno",
    edgecolor="none",
    alpha=0.9,
)
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.set_zlabel("T [°C]")
ax.set_title(f"T(x,z) at y={y_grid[mid_y]:.2f} (3D Surface)")
fig.colorbar(surf, ax=ax, shrink=0.5, label="Temperature [°C]")
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    nodes[:, 0], nodes[:, 1], nodes[:, 2], c=T_solution, cmap="inferno", s=15, alpha=0.6
)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_title("3D Temperature Distribution (FEM Nodes)")
fig.colorbar(scatter, ax=ax, shrink=0.5, label="Temperature [°C]")
plt.tight_layout()
plt.show()


def find_closest_node(target_point, nodes):
    distances = np.linalg.norm(nodes - target_point, axis=1)
    return np.argmin(distances)


center_point = np.array([Lx / 2, Ly / 2, Lz / 2])
left_edge = np.array([0.1, Ly / 2, Lz / 2])
right_edge = np.array([Lx - 0.1, Ly / 2, Lz / 2])

center_idx = find_closest_node(center_point, nodes)
left_idx = find_closest_node(left_edge, nodes)
right_idx = find_closest_node(right_edge, nodes)

x_line_points = np.column_stack(
    [np.linspace(0, Lx, 50), np.full(50, Ly / 2), np.full(50, Lz / 2)]
)
T_line = griddata(nodes, T_solution, x_line_points, method="linear")

plt.figure(figsize=(8, 5))
plt.plot(x_line_points[:, 0], T_line, "b-", linewidth=2, label="FEM solution")
plt.axhline(100, color="r", linestyle="--", alpha=0.7, label="Left BC (100°C)")
plt.axhline(0, color="g", linestyle="--", alpha=0.7, label="Right BC (0°C)")
plt.xlabel("x [m]")
plt.ylabel("Temperature [°C]")
plt.title("Temperature Profile Along x-axis (centerline)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nSTEADY-STATE HEAT DISTRIBUTION")
print(f"Center temperature: {T_solution[center_idx]:.1f}°C")
print(f"Near left (hot): {T_solution[left_idx]:.1f}°C")
print(f"Near right (cold): {T_solution[right_idx]:.1f}°C")
print("Heat flows from left (100°C) to right (0°C)")
