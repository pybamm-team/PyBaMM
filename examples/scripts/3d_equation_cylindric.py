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

R = 0.5
H = 1.0
alpha = 0.1

model = BaseModel()

x = SpatialVariable("x", ["current collector"], coord_sys="cartesian", direction="x")
y = SpatialVariable("y", ["current collector"], coord_sys="cartesian", direction="y")
z = SpatialVariable("z", ["current collector"], coord_sys="cartesian", direction="z")

T = pybamm.Variable("T", domain="current collector")
model.variables = {"T": T}

model.algebraic = {T: alpha * pybamm.laplacian(T) - pybamm.source(0, T)}

model.initial_conditions = {T: Scalar(0)}

model.boundary_conditions = {
    T: {
        "bottom cap": (Scalar(100), "Dirichlet"),
        "top cap": (Scalar(0), "Dirichlet"),
        "side wall": (Scalar(0), "Neumann"),
    }
}

geometry = {
    "current collector": {
        x: {"min": pybamm.Scalar(-R), "max": pybamm.Scalar(R)},
        y: {"min": pybamm.Scalar(-R), "max": pybamm.Scalar(R)},
        z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(H)},
    }
}

submesh_types = {
    "current collector": pybamm.ScikitFemGenerator3D(
        geom_type="cylinder", h=0.1, radius=R, height=H
    )
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

Nr_vis, Ntheta_vis, Nz_vis = 15, 32, 20
r_grid = np.linspace(0, R * 0.99, Nr_vis)
theta_grid = np.linspace(0, 2 * np.pi, Ntheta_vis)
z_grid = np.linspace(0, H, Nz_vis)

R_grid, Theta_grid, Z_grid = np.meshgrid(r_grid, theta_grid, z_grid, indexing="ij")
X_grid = R_grid * np.cos(Theta_grid)
Y_grid = R_grid * np.sin(Theta_grid)
grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()])

T_interpolated = griddata(
    nodes, T_solution, grid_points, method="linear", fill_value=0
).reshape((Nr_vis, Ntheta_vis, Nz_vis))

mid_k = Nz_vis // 2
R_plane = R_grid[:, :, mid_k]
Theta_plane = Theta_grid[:, :, mid_k]
T_plane = T_interpolated[:, :, mid_k]

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection="polar"))
pcm = ax.contourf(Theta_plane, R_plane, T_plane, levels=20, cmap="inferno")
ax.set_title(f"T(r,θ) at z = {H / 2:.2f} m")
ax.set_ylim(0, R)
plt.colorbar(pcm, ax=ax, label="Temperature [°C]", shrink=0.8)
plt.show()

mid_j = 0
R_axial = R_grid[:, mid_j, :]
Z_axial = Z_grid[:, mid_j, :]
T_axial = T_interpolated[:, mid_j, :]

fig, ax = plt.subplots(figsize=(8, 6))
pcm = ax.contourf(R_axial, Z_axial, T_axial, levels=20, cmap="inferno")
ax.set_xlabel("r [m]")
ax.set_ylabel("z [m]")
ax.set_title("T(r,z) at θ = 0")
ax.set_aspect("equal")
plt.colorbar(pcm, ax=ax, label="Temperature [°C]")
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

r_surface = R * 0.9
theta_surface = np.linspace(0, 2 * np.pi, 64)
z_surface = np.linspace(0, H, 32)
Theta_surf, Z_surf = np.meshgrid(theta_surface, z_surface)
X_surf = r_surface * np.cos(Theta_surf)
Y_surf = r_surface * np.sin(Theta_surf)

surface_points = np.column_stack([X_surf.ravel(), Y_surf.ravel(), Z_surf.ravel()])
T_surface = griddata(
    nodes, T_solution, surface_points, method="linear", fill_value=0
).reshape(X_surf.shape)

surf = ax.plot_surface(
    X_surf,
    Y_surf,
    Z_surf,
    facecolors=plt.cm.inferno(T_surface / T_surface.max()),
    alpha=0.8,
    edgecolor="none",
)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_title("3D Cylinder Surface")

node_mask = np.random.choice(len(nodes), size=min(200, len(nodes)), replace=False)
ax.scatter(
    nodes[node_mask, 0],
    nodes[node_mask, 1],
    nodes[node_mask, 2],
    c="red",
    s=1,
    alpha=0.3,
)

plt.tight_layout()
plt.show()


def find_closest_node(target_point, nodes):
    distances = np.linalg.norm(nodes - target_point, axis=1)
    return np.argmin(distances)


center_point = np.array([0, 0, H / 2])
edge_point = np.array([R * 0.8, 0, H / 2])
bottom_point = np.array([0, 0, 0.05])
top_point = np.array([0, 0, H - 0.05])

center_idx = find_closest_node(center_point, nodes)
edge_idx = find_closest_node(edge_point, nodes)
bottom_idx = find_closest_node(bottom_point, nodes)
top_idx = find_closest_node(top_point, nodes)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

pcm1 = ax1.contourf(Theta_plane, R_plane, T_plane, levels=15, cmap="inferno")
ax1.set_title(f"T(r,θ) at z={H / 2:.1f}m")
ax1.set_xlabel("θ [rad]")
ax1.set_ylabel("r [m]")
plt.colorbar(pcm1, ax=ax1)

pcm2 = ax2.contourf(R_axial, Z_axial, T_axial, levels=15, cmap="inferno")
ax2.set_xlabel("r [m]")
ax2.set_ylabel("z [m]")
ax2.set_title("T(r,z) at θ=0")
ax2.set_aspect("equal")
plt.colorbar(pcm2, ax=ax2)

scatter = ax3.scatter(
    nodes[:, 0], nodes[:, 1], c=T_solution, cmap="inferno", s=5, alpha=0.7
)
circle = plt.Circle((0, 0), R, fill=False, color="black", linewidth=2)
ax3.add_patch(circle)
ax3.set_xlabel("x [m]")
ax3.set_ylabel("y [m]")
ax3.set_title("FEM Nodes (x-y plane)")
ax3.set_aspect("equal")
plt.colorbar(scatter, ax=ax3)

z_line_points = np.column_stack([np.full(50, 0), np.full(50, 0), np.linspace(0, H, 50)])
T_line = griddata(nodes, T_solution, z_line_points, method="linear")

ax4.plot(z_line_points[:, 2], T_line, "b-", linewidth=2, label="FEM solution")
ax4.axhline(100, color="r", linestyle="--", alpha=0.7, label="Bottom BC (100°C)")
ax4.axhline(0, color="g", linestyle="--", alpha=0.7, label="Top BC (0°C)")
ax4.set_xlabel("z [m]")
ax4.set_ylabel("Temperature [°C]")
ax4.set_title("Axial Temperature Profile")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nSTEADY-STATE HEAT DISTRIBUTION")
print(f"Center temperature: {T_solution[center_idx]:.1f}°C")
print(f"Edge temperature: {T_solution[edge_idx]:.1f}°C")
print(f"Bottom temperature: {T_solution[bottom_idx]:.1f}°C")
print(f"Top temperature: {T_solution[top_idx]:.1f}°C")
print("Heat flows from bottom (100°C) to top (0°C)")
