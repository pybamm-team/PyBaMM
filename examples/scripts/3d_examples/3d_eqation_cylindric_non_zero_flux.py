import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from scipy.interpolate import griddata

import pybamm
from pybamm import (
    BaseModel,
    Discretisation,
    Scalar,
    ScikitFiniteElement3D,
)
from pybamm.meshes.scikit_fem_submeshes_3d import ScikitFemGenerator3D

R_outer = 0.6
R_inner = 0.1
H = 1.0
alpha = 0.1

heat_generation = 1000.0
heat_flux_top = 500.0
heat_flux_bottom = -300.0

model = BaseModel()

r = pybamm.SpatialVariable("r", ["current collector"], coord_sys="cylindrical polar")
theta = pybamm.SpatialVariable(
    "theta", ["current collector"], coord_sys="cylindrical polar"
)
z = pybamm.SpatialVariable("z", ["current collector"], coord_sys="cylindrical polar")
T = pybamm.Variable("T", domain="current collector")

source_term = pybamm.source(heat_generation, T)

model.algebraic = {T: alpha * pybamm.laplacian(T) + source_term}
model.initial_conditions = {T: Scalar(25)}

model.boundary_conditions = {
    T: {
        "r_min": (Scalar(100), "Dirichlet"),
        "r_max": (Scalar(20), "Dirichlet"),
        "z_min": (Scalar(heat_flux_bottom), "Neumann"),
        "z_max": (Scalar(heat_flux_top), "Neumann"),
    }
}

geometry = {
    "current collector": {
        r: {"min": pybamm.Scalar(R_inner), "max": pybamm.Scalar(R_outer)},
        z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(H)},
    }
}

submesh_types = {
    "current collector": ScikitFemGenerator3D(geom_type="cylinder", h=0.08)
}

var_pts = {r: None, theta: None, z: None}
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

Nr_vis, Ntheta_vis, Nz_vis = 30, 40, 25
r_grid = np.linspace(R_inner, R_outer, Nr_vis)
theta_grid = np.linspace(0, 2 * np.pi, Ntheta_vis)
z_grid = np.linspace(0, H, Nz_vis)
R_grid, Theta_grid, Z_grid = np.meshgrid(r_grid, theta_grid, z_grid, indexing="ij")

X_grid = R_grid * np.cos(Theta_grid)
Y_grid = R_grid * np.sin(Theta_grid)
grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()])

T_interpolated = griddata(
    nodes, T_solution, grid_points, method="linear", fill_value=np.nan
).reshape((Nr_vis, Ntheta_vis, Nz_vis))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Heat Distribution with Source Term and Neumann BCs", fontsize=16)

mid_k = Nz_vis // 2
R_plane = R_grid[:, :, mid_k]
Theta_plane = Theta_grid[:, :, mid_k]
T_plane = T_interpolated[:, :, mid_k]
ax1.remove()
ax1 = fig.add_subplot(2, 2, 1, projection="polar")
pcm1 = ax1.contourf(Theta_plane, R_plane, T_plane, levels=20, cmap="plasma")
ax1.set_title(f"T(r,θ) at z={H / 2:.1f}m")
ax1.set_ylim(R_inner, R_outer)
fig.colorbar(pcm1, ax=ax1)

mid_j = 0
R_axial = R_grid[:, mid_j, :]
Z_axial = Z_grid[:, mid_j, :]
T_axial = T_interpolated[:, mid_j, :]
pcm2 = ax2.contourf(R_axial, Z_axial, T_axial, levels=20, cmap="plasma")
ax2.set_xlabel("r [m]")
ax2.set_ylabel("z [m]")
ax2.set_title("T(r,z) - Shows effect of Neumann BCs")
ax2.set_aspect("equal")
fig.colorbar(pcm2, ax=ax2)

unique_z_coords = np.unique(nodes[:, 2])
mid_plane_z = unique_z_coords[np.argmin(np.abs(unique_z_coords - H / 2))]

z_mid_indices = nodes[:, 2] == mid_plane_z
z_mid_nodes = nodes[z_mid_indices]
T_mid = T_solution[z_mid_indices]

triang = tri.Triangulation(z_mid_nodes[:, 0], z_mid_nodes[:, 1])
pcm3 = ax3.tricontourf(triang, T_mid, levels=20, cmap="plasma")
ax3.plot(
    R_outer * np.cos(np.linspace(0, 2 * np.pi, 100)),
    R_outer * np.sin(np.linspace(0, 2 * np.pi, 100)),
    "k-",
    linewidth=2,
)
ax3.plot(
    R_inner * np.cos(np.linspace(0, 2 * np.pi, 100)),
    R_inner * np.sin(np.linspace(0, 2 * np.pi, 100)),
    "w-",
    linewidth=2,
)
ax3.set_xlabel("x [m]")
ax3.set_ylabel("y [m]")
ax3.set_title("FEM Solution at Mid-Height")
ax3.set_aspect("equal")
fig.colorbar(pcm3, ax=ax3, label="Temperature [°C]")

z_line = np.linspace(0, H, 100)
mid_radius = (R_inner + R_outer) / 2
axial_points = np.column_stack(
    [np.full_like(z_line, mid_radius), np.zeros_like(z_line), z_line]
)
T_axial_profile = griddata(nodes, T_solution, axial_points, method="linear")
ax4.plot(z_line, T_axial_profile, "b-", linewidth=3, label="FEM solution")
ax4.axhline(100, color="r", linestyle="--", alpha=0.7, label="Inner BC (100°C)")
ax4.axhline(20, color="g", linestyle="--", alpha=0.7, label="Outer BC (20°C)")
ax4.set_xlabel("z [m]")
ax4.set_ylabel("Temperature [°C]")
ax4.set_title("Axial Temperature Profile\n(shows Neumann BC effects)")
ax4.legend()
ax4.grid(True, alpha=0.3)

ax4.text(
    0.05,
    T_axial_profile[5],
    f"q = {heat_flux_bottom} W/m²\n(cooling)",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
)
ax4.text(
    0.85,
    T_axial_profile[-5],
    f"q = {heat_flux_top} W/m²\n(heating)",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"),
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


def find_closest_node(target_point, nodes):
    distances = np.linalg.norm(nodes - target_point, axis=1)
    return np.argmin(distances)


inner_point = np.array([R_inner, 0, H / 2])
outer_point = np.array([R_outer, 0, H / 2])
mid_point = np.array([(R_inner + R_outer) / 2, 0, H / 2])
bottom_point = np.array([(R_inner + R_outer) / 2, 0, 0])
top_point = np.array([(R_inner + R_outer) / 2, 0, H])

inner_idx = find_closest_node(inner_point, nodes)
outer_idx = find_closest_node(outer_point, nodes)
mid_idx = find_closest_node(mid_point, nodes)
bottom_idx = find_closest_node(bottom_point, nodes)
top_idx = find_closest_node(top_point, nodes)

print("\n" + "=" * 60)
print("HEAT DISTRIBUTION WITH SOURCE TERM AND NEUMANN BCs")
print("=" * 60)
print("Boundary Conditions:")
print(f"  Inner radius (r={R_inner}m): T = 100°C (Dirichlet)")
print(f"  Outer radius (r={R_outer}m): T = 20°C (Dirichlet)")
print(f"  Bottom surface (z=0): q = {heat_flux_bottom} W/m² (Neumann - cooling)")
print(f"  Top surface (z={H}m): q = {heat_flux_top} W/m² (Neumann - heating)")
print(f"  Volumetric heat source: Q = {heat_generation} W/m³")
print("\nTemperature Results:")
print(f"  Inner surface: {T_solution[inner_idx]:.1f}°C")
print(f"  Outer surface: {T_solution[outer_idx]:.1f}°C")
print(f"  Mid-radius center: {T_solution[mid_idx]:.1f}°C")
print(f"  Bottom center: {T_solution[bottom_idx]:.1f}°C")
print(f"  Top center: {T_solution[top_idx]:.1f}°C")
print(f"  Overall range: [{T_solution.min():.1f}, {T_solution.max():.1f}]°C")

temp_rise_axial = T_solution[top_idx] - T_solution[bottom_idx]
print(f"\nAxial temperature rise due to Neumann BCs: {temp_rise_axial:.1f}°C")
print("This demonstrates how:")
print("  1. Positive Neumann flux (top) heats the surface")
print("  2. Negative Neumann flux (bottom) cools the surface")
print("  3. Source term raises overall temperature level")
