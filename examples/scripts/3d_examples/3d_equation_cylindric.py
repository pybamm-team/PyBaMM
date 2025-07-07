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

model = BaseModel()

r = pybamm.SpatialVariable("r", ["current collector"], coord_sys="cylindrical polar")
theta = pybamm.SpatialVariable(
    "theta", ["current collector"], coord_sys="cylindrical polar"
)
z = pybamm.SpatialVariable("z", ["current collector"], coord_sys="cylindrical polar")
T = pybamm.Variable("T", domain="current collector")

model.algebraic = {T: alpha * pybamm.laplacian(T)}
model.initial_conditions = {T: Scalar(25)}
model.boundary_conditions = {
    T: {
        "r_min": (Scalar(100), "Dirichlet"),
        "r_max": (Scalar(0), "Dirichlet"),
        "z_min": (Scalar(0), "Neumann"),
        "z_max": (Scalar(0), "Neumann"),
    }
}

geometry = {
    "current collector": {
        r: {"min": pybamm.Scalar(R_inner), "max": pybamm.Scalar(R_outer)},
        z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(H)},
    }
}

# For unstructured 3D FEM meshes, we control the density and quality using a
# characteristic length parameter 'h'. The generator will try to create
# tetrahedral elements with edge lengths around this value.
submesh_types = {
    "current collector": ScikitFemGenerator3D(geom_type="cylinder", h=0.08)
}

# The 'var_pts' dictionary is required by the pybamm.Mesh class for validation,
# but our ScikitFemGenerator3D completely IGNORES it. The number of points
# is determined by the meshing algorithm based on 'h', not by 'var_pts'.
# We pass placeholder values to satisfy the class constructor.
# We may want to pass var_pts anyway since other domains may not be in 3D in a simulation
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

# Create a regular grid for visualization purposes.
Nr_vis, Ntheta_vis, Nz_vis = 30, 40, 25
r_grid = np.linspace(R_inner, R_outer, Nr_vis)
theta_grid = np.linspace(0, 2 * np.pi, Ntheta_vis)
z_grid = np.linspace(0, H, Nz_vis)
R_grid, Theta_grid, Z_grid = np.meshgrid(r_grid, theta_grid, z_grid, indexing="ij")

X_grid = R_grid * np.cos(Theta_grid)
Y_grid = R_grid * np.sin(Theta_grid)
grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()])

# Interpolate the unstructured FEM solution onto the regular visualization grid.
# This can sometimes make the plot look "lumpy" if the mesh is coarse.
T_interpolated = griddata(
    nodes, T_solution, grid_points, method="linear", fill_value=np.nan
).reshape((Nr_vis, Ntheta_vis, Nz_vis))

# Create the 4-panel plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Steady-State Heat Distribution Analysis", fontsize=16)

# Plot 1: Polar plot (r-theta plane) at mid-height
mid_k = Nz_vis // 2
R_plane = R_grid[:, :, mid_k]
Theta_plane = Theta_grid[:, :, mid_k]
T_plane = T_interpolated[:, :, mid_k]
ax1.remove()
ax1 = fig.add_subplot(2, 2, 1, projection="polar")
pcm1 = ax1.contourf(Theta_plane, R_plane, T_plane, levels=20, cmap="viridis")
ax1.set_title(f"T(r,θ) at z={H / 2:.1f}m (Interpolated)")
ax1.set_ylim(R_inner, R_outer)
fig.colorbar(pcm1, ax=ax1)

# Plot 2: Cross-section (r-z plane) at theta=0
mid_j = 0
R_axial = R_grid[:, mid_j, :]
Z_axial = Z_grid[:, mid_j, :]
T_axial = T_interpolated[:, mid_j, :]
pcm2 = ax2.contourf(R_axial, Z_axial, T_axial, levels=20, cmap="viridis")
ax2.set_xlabel("r [m]")
ax2.set_ylabel("z [m]")
ax2.set_title("T(r,z) at θ=0 (Interpolated)")
ax2.set_aspect("equal")
fig.colorbar(pcm2, ax=ax2)

# Plot 3: Direct FEM solution plot at mid-height (no interpolation)
unique_z_coords = np.unique(nodes[:, 2])
mid_plane_z = unique_z_coords[np.argmin(np.abs(unique_z_coords - H / 2))]
print(
    f"\nPlotting direct FEM solution on the mesh plane closest to mid-height (z={mid_plane_z:.3f}m)."
)

z_mid_indices = nodes[:, 2] == mid_plane_z
z_mid_nodes = nodes[z_mid_indices]
T_mid = T_solution[z_mid_indices]

triang = tri.Triangulation(z_mid_nodes[:, 0], z_mid_nodes[:, 1])
pcm3 = ax3.tricontourf(triang, T_mid, levels=20, cmap="viridis")
ax3.plot(
    R_outer * np.cos(np.linspace(0, 2 * np.pi, 100)),
    R_outer * np.sin(np.linspace(0, 2 * np.pi, 100)),
    "k-",
)
ax3.plot(
    R_inner * np.cos(np.linspace(0, 2 * np.pi, 100)),
    R_inner * np.sin(np.linspace(0, 2 * np.pi, 100)),
    "r-",
)
ax3.set_xlabel("x [m]")
ax3.set_ylabel("y [m]")
ax3.set_title("FEM Solution at Mid-Height (Direct)")
ax3.set_aspect("equal")
fig.colorbar(pcm3, ax=ax3, label="Temperature [°C]")

# Plot 4: Radial temperature profile vs. analytical solution
r_line = np.linspace(R_inner, R_outer, 100)
radial_points = np.column_stack(
    [r_line, np.zeros_like(r_line), np.full_like(r_line, H / 2)]
)
T_radial = griddata(nodes, T_solution, radial_points, method="linear")
ax4.plot(r_line, T_radial, "b-", linewidth=3, label="FEM solution")
A = -100 / np.log(R_outer / R_inner)
B = 100 - A * np.log(R_inner)
T_analytical = A * np.log(r_line) + B
ax4.plot(r_line, T_analytical, "r--", linewidth=2, label="Analytical solution")
ax4.set_xlabel("r [m]")
ax4.set_ylabel("Temperature [°C]")
ax4.set_title("Radial Temperature Profile")
ax4.legend()
ax4.grid(True, alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Mesh Visualisation
print("\nGenerating mesh visualization...")
fig_mesh, ax_mesh = plt.subplots(figsize=(8, 8))
ax_mesh.triplot(triang, "k-", lw=0.5, alpha=0.7)
ax_mesh.plot(z_mid_nodes[:, 0], z_mid_nodes[:, 1], "o", markersize=2, color="blue")
ax_mesh.set_title(f"FEM Mesh Visualization (Slice at z={mid_plane_z:.3f}m)")
ax_mesh.set_xlabel("x [m]")
ax_mesh.set_ylabel("y [m]")
ax_mesh.set_aspect("equal")
plt.show()


def find_closest_node(target_point, nodes):
    distances = np.linalg.norm(nodes - target_point, axis=1)
    return np.argmin(distances)


inner_point = np.array([R_inner, 0, H / 2])
outer_point = np.array([R_outer, 0, H / 2])
mid_point = np.array([(R_inner + R_outer) / 2, 0, H / 2])
inner_idx = find_closest_node(inner_point, nodes)
outer_idx = find_closest_node(outer_point, nodes)
mid_idx = find_closest_node(mid_point, nodes)

print("\n" + "=" * 50)
print("STEADY-STATE HEAT DISTRIBUTION ANALYSIS")
print("=" * 50)
print(f"FEM Inner surface temperature: {T_solution[inner_idx]:.2f}°C")
print(f"FEM Outer surface temperature: {T_solution[outer_idx]:.2f}°C")
print(f"FEM Mid-radius temperature: {T_solution[mid_idx]:.2f}°C")
T_mid_analytical = A * np.log((R_inner + R_outer) / 2) + B
print(f"Analytical mid-radius temperature: {T_mid_analytical:.2f}°C")
error = abs(T_solution[mid_idx] - T_mid_analytical)
print(f"Error at mid-radius: {error:.3f}°C")
