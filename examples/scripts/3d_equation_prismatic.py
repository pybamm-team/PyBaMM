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

Lx, Ly, Lz = 1.0, 0.8, 0.6
kappa = 0.1
t_max = 0.2

model = BaseModel()

x = SpatialVariable("x", ["prism"], coord_sys="cartesian", direction="x")
y = SpatialVariable("y", ["prism"], coord_sys="cartesian", direction="y")
z = SpatialVariable("z", ["prism"], coord_sys="cartesian", direction="z")

T = x * 2 * y + 3 * z
model.variables = {"T": T}

geometry = {
    "prism": {
        x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(Lx)},
        y: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(Ly)},
        z: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(Lz)},
    }
}

submesh_types = {
    "prism": pybamm.ScikitFemGenerator3D(geom_type="box", gen_params={"h": 0.1})
}

var_pts = {x: None, y: None, z: None}

mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

spatial_methods = {"prism": ScikitFiniteElement3D()}
disc = Discretisation(mesh, spatial_methods)

disc.process_model(model)

t_disc = model.variables["T"].entries
submesh = mesh["prism"]
nodes = submesh.nodes

print(f"Number of FEM nodes: {nodes.shape[0]}")
print(
    f"Mesh bounds: x=[{nodes[:, 0].min():.3f}, {nodes[:, 0].max():.3f}], "
    f"y=[{nodes[:, 1].min():.3f}, {nodes[:, 1].max():.3f}], "
    f"z=[{nodes[:, 2].min():.3f}, {nodes[:, 2].max():.3f}]"
)

Nx_vis, Ny_vis, Nz_vis = 20, 20, 20
x_grid = np.linspace(0, Lx, Nx_vis)
y_grid = np.linspace(0, Ly, Ny_vis)
z_grid = np.linspace(0, Lz, Nz_vis)

X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()])

T_interpolated = griddata(
    nodes, t_disc, grid_points, method="linear", fill_value=0
).reshape((Nx_vis, Ny_vis, Nz_vis))

mid_j = Ny_vis // 2
X_plane = X_grid[:, mid_j, :]
Z_plane = Z_grid[:, mid_j, :]
T_plane = T_interpolated[:, mid_j, :]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))
surf = ax.plot_surface(
    X_plane,
    Z_plane,
    T_plane,
    cmap="inferno",
    edgecolor="none",
    rcount=40,
    ccount=40,
    alpha=0.9,
)

ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.set_zlabel("T [°C]")
ax.set_title(f"T(x,z) at y = {Ly / 2:.2f} (3D FEM)")
fig.colorbar(surf, ax=ax, shrink=0.5, label="Temperature [°C]")

y_mid_mask = np.abs(nodes[:, 1] - Ly / 2) < 0.05
if np.any(y_mid_mask):
    nodes_mid = nodes[y_mid_mask]
    t_mid = t_disc[y_mid_mask]
    ax.scatter(
        nodes_mid[:, 0],
        nodes_mid[:, 2],
        t_mid,
        c="red",
        s=20,
        alpha=0.6,
        label="FEM nodes",
    )
    ax.legend()

plt.tight_layout()
plt.show()


def find_closest_node(target_point, nodes):
    distances = np.linalg.norm(nodes - target_point, axis=1)
    return np.argmin(distances)


center_point = np.array([Lx / 2, Ly / 2, Lz / 2])
corner_point = np.array([0, 0, 0])

center_idx = find_closest_node(center_point, nodes)
corner_idx = find_closest_node(corner_point, nodes)

print("\nFEM Results:")
print(f"T(center) ≈ {float(t_disc[center_idx]):.4f} °C at node {nodes[center_idx]}")
print(f"T(corner) ≈ {float(t_disc[corner_idx]):.4f} °C at node {nodes[corner_idx]}")

T_center_analytical = (Lx / 2) * 2 * (Ly / 2) + 3 * (Lz / 2)
T_corner_analytical = 0 * 2 * 0 + 3 * 0

print("\nAnalytical comparison:")
print(f"T(center) analytical = {T_center_analytical:.4f} °C")
print(f"T(corner) analytical = {T_corner_analytical:.4f} °C")
print(f"Center error: {abs(float(t_disc[center_idx]) - T_center_analytical):.6f}")
print(f"Corner error: {abs(float(t_disc[corner_idx]) - T_corner_analytical):.6f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax1.contourf(X_plane, Z_plane, T_plane, levels=20, cmap="inferno")
ax1.set_xlabel("x [m]")
ax1.set_ylabel("z [m]")
ax1.set_title(f"T(x,z) at y = {Ly / 2:.2f}")
plt.colorbar(im1, ax=ax1, label="Temperature [°C]")

ax2.scatter(nodes[:, 0], nodes[:, 1], c=t_disc, cmap="inferno", s=10, alpha=0.7)
ax2.set_xlabel("x [m]")
ax2.set_ylabel("y [m]")
ax2.set_title("FEM Node Distribution (x-y plane)")
ax2.set_aspect("equal")
plt.colorbar(ax2.collections[0], ax=ax2, label="Temperature [°C]")

plt.tight_layout()
plt.show()

print("\nMesh Information:")
print(f"Number of nodes: {submesh.npts}")
print(f"Number of elements: {submesh.nelements}")
print("Mesh type: Unstructured tetrahedral (scikit-fem)")
