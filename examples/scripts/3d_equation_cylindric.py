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

R_outer = 0.6
R_inner = 0.1
H = 1.0
alpha = 0.1

model = BaseModel()

r = pybamm.SpatialVariable(
    "r", ["current collector"], coord_sys="cylindrical polar", direction="r"
)
theta = pybamm.SpatialVariable(
    "theta", ["current collector"], coord_sys="cylindrical polar", direction="theta"
)
z = pybamm.SpatialVariable(
    "z", ["current collector"], coord_sys="cylindrical polar", direction="z"
)
T = pybamm.Variable("T", domain="current collector")

model.algebraic = {T: alpha * pybamm.laplacian(T)}
model.initial_conditions = {T: Scalar(25)}
model.boundary_conditions = {
    T: {
        "inner radius": (Scalar(100), "Dirichlet"),
        "outer radius": (Scalar(0), "Dirichlet"),
        "bottom": (Scalar(0), "Neumann"),
        "top": (Scalar(0), "Neumann"),
    }
}

geometry = {
    "current collector": {
        r: {"min": pybamm.Scalar(R_inner), "max": pybamm.Scalar(R_outer)},
        z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(H)},
    }
}

submesh_types = {
    "current collector": pybamm.ScikitFemGenerator3D(geom_type="cylinder", h=0.1)
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
fig.suptitle("Steady-State Heat Distribution Analysis", fontsize=16)

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

z_mid_indices = np.abs(nodes[:, 2] - H / 2) < (H / 20)
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
