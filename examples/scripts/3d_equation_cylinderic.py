import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

import pybamm
from pybamm import (
    BaseModel,
    Discretisation,
    ScikitFiniteElement3D,
    SpatialVariable,
)

R = 0.5
H = 1
kappa = 0.1
t_max = 0.2

model = BaseModel()

x = SpatialVariable("x", ["cylinder"], coord_sys="cartesian", direction="x")
y = SpatialVariable("y", ["cylinder"], coord_sys="cartesian", direction="y")
z = SpatialVariable("z", ["cylinder"], coord_sys="cartesian", direction="z")

T = x**2 + y**2 + 2 * z
model.variables = {"T": T}

# Cylinder geometry: centered at origin, radius R, height H
geometry = {
    "cylinder": {
        x: {"min": pybamm.Scalar(-R), "max": pybamm.Scalar(R)},
        y: {"min": pybamm.Scalar(-R), "max": pybamm.Scalar(R)},
        z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(H)},
    }
}

submesh_types = {
    "cylinder": pybamm.ScikitFemGenerator3D(
        geom_type="cylinder", h=0.1, radius=R, height=H
    )
}

var_pts = {x: None, y: None, z: None}

mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

spatial_methods = {"cylinder": ScikitFiniteElement3D()}
disc = Discretisation(mesh, spatial_methods)

disc.process_model(model)

t_disc = model.variables["T"].entries
submesh = mesh["cylinder"]
nodes = submesh.nodes

print(f"Number of FEM nodes: {nodes.shape[0]}")
print(f"Cylinder: radius={R}, height={H}")
print(
    f"Mesh bounds: x=[{nodes[:, 0].min():.3f}, {nodes[:, 0].max():.3f}], "
    f"y=[{nodes[:, 1].min():.3f}, {nodes[:, 1].max():.3f}], "
    f"z=[{nodes[:, 2].min():.3f}, {nodes[:, 2].max():.3f}]"
)

Nr_vis, Ntheta_vis, Nz_vis = 15, 32, 20

r_grid = np.linspace(0, R * 0.99, Nr_vis)  # Slightly inside to avoid boundary issues
theta_grid = np.linspace(0, 2 * np.pi, Ntheta_vis)
z_grid = np.linspace(0, H, Nz_vis)

# Convert to Cartesian for interpolation
R_grid, Theta_grid, Z_grid = np.meshgrid(r_grid, theta_grid, z_grid, indexing="ij")
X_grid = R_grid * np.cos(Theta_grid)
Y_grid = R_grid * np.sin(Theta_grid)
grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()])

T_interpolated = griddata(
    nodes, t_disc, grid_points, method="linear", fill_value=0
).reshape((Nr_vis, Ntheta_vis, Nz_vis))

mid_k = Nz_vis // 2
R_plane = R_grid[:, :, mid_k]
Theta_plane = Theta_grid[:, :, mid_k]
T_plane = T_interpolated[:, :, mid_k]

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection="polar"))
pcm = ax.contourf(Theta_plane, R_plane, T_plane, levels=20, cmap="inferno")
ax.set_title(f"T(r,θ) at z = {H / 2:.2f} m (Cylinder Cross-Section)")
ax.set_ylim(0, R)
plt.colorbar(pcm, ax=ax, label="Temperature [°C]", shrink=0.8)
plt.show()

# 2. Plot axial cross-section (r-z plane at θ=0)
mid_j = 0  # θ = 0 plane
R_axial = R_grid[:, mid_j, :]
Z_axial = Z_grid[:, mid_j, :]
T_axial = T_interpolated[:, mid_j, :]

fig, ax = plt.subplots(figsize=(8, 6))
pcm = ax.contourf(R_axial, Z_axial, T_axial, levels=20, cmap="inferno")
ax.set_xlabel("r [m]")
ax.set_ylabel("z [m]")
ax.set_title("T(r,z) at θ = 0 (Axial Cross-Section)")
ax.set_aspect("equal")
plt.colorbar(pcm, ax=ax, label="Temperature [°C]")
plt.show()

# 3. 3D surface plot of cylinder surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot cylinder surface at r = R*0.9
r_surface = R * 0.9
theta_surface = np.linspace(0, 2 * np.pi, 64)
z_surface = np.linspace(0, H, 32)
Theta_surf, Z_surf = np.meshgrid(theta_surface, z_surface)
X_surf = r_surface * np.cos(Theta_surf)
Y_surf = r_surface * np.sin(Theta_surf)

# Interpolate temperature on surface
surface_points = np.column_stack([X_surf.ravel(), Y_surf.ravel(), Z_surf.ravel()])
T_surface = griddata(
    nodes, t_disc, surface_points, method="linear", fill_value=0
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
ax.set_title("3D Cylinder Surface (colored by temperature)")

node_mask = np.random.choice(len(nodes), size=min(200, len(nodes)), replace=False)
ax.scatter(
    nodes[node_mask, 0],
    nodes[node_mask, 1],
    nodes[node_mask, 2],
    c="red",
    s=1,
    alpha=0.3,
    label="FEM nodes",
)

plt.tight_layout()
plt.show()


def find_closest_node(target_point, nodes):
    distances = np.linalg.norm(nodes - target_point, axis=1)
    return np.argmin(distances)


center_point = np.array([0, 0, H / 2])
edge_point = np.array([R * 0.8, 0, H / 2])
bottom_point = np.array([0, 0, 0])
top_point = np.array([0, 0, H])

center_idx = find_closest_node(center_point, nodes)
edge_idx = find_closest_node(edge_point, nodes)
bottom_idx = find_closest_node(bottom_point, nodes)
top_idx = find_closest_node(top_point, nodes)

print("\nFEM Results:")
print(f"T(center) ≈ {float(t_disc[center_idx]):.4f} °C at {nodes[center_idx]}")
print(f"T(edge) ≈ {float(t_disc[edge_idx]):.4f} °C at {nodes[edge_idx]}")
print(f"T(bottom) ≈ {float(t_disc[bottom_idx]):.4f} °C at {nodes[bottom_idx]}")
print(f"T(top) ≈ {float(t_disc[top_idx]):.4f} °C at {nodes[top_idx]}")

T_center_analytical = 0**2 + 0**2 + 2 * (H / 2)
T_edge_analytical = (R * 0.8) ** 2 + 2 * (H / 2)
T_bottom_analytical = 0**2 + 2 * 0
T_top_analytical = 0**2 + 2 * H

print("\nAnalytical comparison:")
print(f"T(center) analytical = {T_center_analytical:.4f} °C")
print(f"T(edge) analytical = {T_edge_analytical:.4f} °C")
print(f"T(bottom) analytical = {T_bottom_analytical:.4f} °C")
print(f"T(top) analytical = {T_top_analytical:.4f} °C")

print("\nErrors:")
print(f"Center error: {abs(float(t_disc[center_idx]) - T_center_analytical):.6f}")
print(f"Edge error: {abs(float(t_disc[edge_idx]) - T_edge_analytical):.6f}")
print(f"Bottom error: {abs(float(t_disc[bottom_idx]) - T_bottom_analytical):.6f}")
print(f"Top error: {abs(float(t_disc[top_idx]) - T_top_analytical):.6f}")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Top-left: Polar cross-section
pcm1 = ax1.contourf(Theta_plane, R_plane, T_plane, levels=15, cmap="inferno")
ax1.set_title(f"T(r,θ) at z={H / 2:.1f}m")
ax1.set_xlabel("θ [rad]")
ax1.set_ylabel("r [m]")
plt.colorbar(pcm1, ax=ax1)

# Top-right: Axial cross-section
pcm2 = ax2.contourf(R_axial, Z_axial, T_axial, levels=15, cmap="inferno")
ax2.set_xlabel("r [m]")
ax2.set_ylabel("z [m]")
ax2.set_title("T(r,z) at θ=0")
ax2.set_aspect("equal")
plt.colorbar(pcm2, ax=ax2)

# Bottom-left: Node distribution (x-y view)
scatter = ax3.scatter(
    nodes[:, 0], nodes[:, 1], c=t_disc, cmap="inferno", s=5, alpha=0.7
)
circle = plt.Circle((0, 0), R, fill=False, color="black", linewidth=2)
ax3.add_patch(circle)
ax3.set_xlabel("x [m]")
ax3.set_ylabel("y [m]")
ax3.set_title("FEM Nodes (x-y plane)")
ax3.set_aspect("equal")
plt.colorbar(scatter, ax=ax3)

# Bottom-right: Radial temperature profile at mid-height
r_nodes = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
z_mid_mask = np.abs(nodes[:, 2] - H / 2) < 0.1
if np.any(z_mid_mask):
    r_mid = r_nodes[z_mid_mask]
    t_mid = t_disc[z_mid_mask]
    ax4.scatter(r_mid, t_mid, alpha=0.6, s=10)

    # Analytical curve
    r_analytical = np.linspace(0, R, 100)
    t_analytical = r_analytical**2 + 2 * (H / 2)
    ax4.plot(r_analytical, t_analytical, "r-", linewidth=2, label="Analytical")

ax4.set_xlabel("r [m]")
ax4.set_ylabel("T [°C]")
ax4.set_title(f"Radial Profile at z={H / 2:.1f}m")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
