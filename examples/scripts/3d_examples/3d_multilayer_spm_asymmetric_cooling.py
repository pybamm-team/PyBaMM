"""
Multilayer 3D-thermal SPM: asymmetric-cooling demonstration.

Runs a large pouch cell in parallel connection. The left face is cooled
aggressively while the other faces are insulated, so the zone closest to the
cold plate (zone 0) stays cooler than the zone closest to the insulated
face (zone num_layers - 1). This illustrates the through-stack temperature
gradient that ``MultiLayer3DThermalSPM`` resolves, which ``Basic3DThermalSPM``
cannot capture.

The physical stack has ``num_layers * layers_per_zone`` unit cells, but we
model only ``num_layers`` SPM zones (each representing ``layers_per_zone``
adjacent unit cells lumped into one thermal zone). See
``MultiLayer3DThermalSPM.layers_per_zone`` for the semantics: electrode
thicknesses and per-unit-cell current/overpotentials are preserved, only the
zone thermal x-extent and the stack capacity are rescaled.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

import pybamm

print("Building multi-layer 3D-thermal SPM (parallel connection)...")
# A larger thermal contact resistance (1e-2 K.m^2/W, e.g. an adhesive/air gap
# between layers) gives the stack enough thermal impedance for a visible
# through-stack gradient under asymmetric cooling. Reduce towards 1e-4 to
# approximate perfect thermal contact.
#
# ``mesh_h`` controls the 3D FEM mesh density per zone. Smaller h => finer
# mesh, slower solve. ``0.03`` yields ~96 nodes per zone (~1920 total for
# 20 zones) and solves in a few seconds; ``0.02`` triggers a steep jump in
# solve time because the 3D meshes become large enough to dominate the
# DAE, so stay at 0.03 unless you need finer visualisation and are willing
# to wait.
#
# ``num_physical_layers`` is the total number of unit cells in the real
# stack (physically meaningful); ``num_subdivisions`` chooses how many SPM
# zones we solve, each lumping ``num_physical_layers / num_subdivisions``
# adjacent unit cells into one thermal zone. Electrode parameters and
# per-unit-cell overpotentials are preserved; only the zone thermal extent
# and the stack capacity are rescaled. Here we represent a 120-unit-cell
# stack with 20 zones (i.e. 6 physical layers/zone).
model = pybamm.lithium_ion.MultiLayer3DThermalSPM(
    num_physical_layers=120,
    num_subdivisions=20,
    connection="parallel",
    mesh_h=0.03,
)

parameter_values = pybamm.ParameterValues("Marquis2019")

# ----------------------------------------------------------------- #
# Scale capacity to the full physical stack so the C-rate in the
# Experiment refers to the full physical stack (``num_physical_layers``)
# and not a single unit cell. apply_stack_scaling handles this in one
# step and prints the resolved scaling.
# ----------------------------------------------------------------- #
model.apply_stack_scaling(parameter_values)

# Inter-layer thermal contact resistance (K.m2.W-1). This is a proper
# pybamm.Parameter, so it can be set/swept here in the parameter values.
parameter_values.update({model.CONTACT_RESISTANCE_PARAM: 1e-2})

# Asymmetric cooling: cold plate on the left, insulated everywhere else.
h_cold = 50.0  # W.m-2.K-1
h_insulated = 0.1  # W.m-2.K-1
parameter_values.update(
    {
        "Total heat transfer coefficient [W.m-2.K-1]": h_insulated,
        "Left face heat transfer coefficient [W.m-2.K-1]": h_cold,
        "Right face heat transfer coefficient [W.m-2.K-1]": h_insulated,
        "Front face heat transfer coefficient [W.m-2.K-1]": h_insulated,
        "Back face heat transfer coefficient [W.m-2.K-1]": h_insulated,
        "Bottom face heat transfer coefficient [W.m-2.K-1]": h_insulated,
        "Top face heat transfer coefficient [W.m-2.K-1]": h_insulated,
    }
)

# High discharge rate to generate visible heating.
experiment = pybamm.Experiment([("Discharge at 3C until 2.8V",)])

var_pts = {
    "x_n": 10,
    "x_s": 10,
    "x_p": 10,
    "r_n": 15,
    "r_p": 15,
    "x": None,
    "y": None,
    "z": None,
}

sim = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    var_pts=var_pts,
    experiment=experiment,
)

print("Solving...")
solution = sim.solve()
print(f"Solve complete. Final t = {solution.t[-1]:.1f} s")

# --------------------------------------------------------------- #
# Summary print-out
# --------------------------------------------------------------- #
print("\n--- End-of-discharge summary ---")
num_layers = model.num_layers
num_physical_layers = model.num_physical_layers
print(
    f"  {model.num_subdivisions} SPM zones x {model.layers_per_zone} "
    f"physical layers/zone = {num_physical_layers} physical unit cells"
)
for i in range(num_layers):
    T_i = float(solution[f"Layer {i} average temperature [K]"].data[-1])
    f_i = float(solution[f"Layer {i} current fraction"].data[-1])
    I_cell = float(solution[f"Layer {i} per-unit-cell current [A]"].data[-1])
    print(
        f"  Zone {i}: T_av = {T_i:7.3f} K,  current fraction = {f_i:.4f},  "
        f"per-unit-cell current = {I_cell:+.3f} A"
    )

T_max = float(solution["Maximum layer-averaged temperature [K]"].data[-1])
T_min = float(solution["Minimum layer-averaged temperature [K]"].data[-1])
spread = float(solution["Temperature spread [K]"].data[-1])
print(f"  Stack spread: T_max - T_min = {T_max:.3f} - {T_min:.3f} = {spread:.3f} K")
V_end = float(solution["Voltage [V]"].data[-1])
print(f"  Terminal voltage (end): {V_end:.3f} V")

# --------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------- #
print("\nGenerating plots...")

t_s = solution["Time [s]"].data
t_min = t_s / 60.0

fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)

# Shared colormap + normalization for per-layer traces; the colorbar
# replaces per-line legend entries when num_layers is large.
cmap = cm.viridis
norm = Normalize(vmin=0, vmax=max(num_layers - 1, 1))
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # required by older matplotlib versions

# (a) Terminal voltage
axes[0, 0].plot(t_min, solution["Voltage [V]"].data, color="C0")
axes[0, 0].set_xlabel("Time [min]")
axes[0, 0].set_ylabel("Terminal voltage [V]")
axes[0, 0].set_title("Terminal voltage")
axes[0, 0].grid(True, alpha=0.3)

# (b) Per-layer average temperature overlaid, coloured by layer index
for i in range(num_layers):
    T_i = solution[f"Layer {i} average temperature [K]"].data
    axes[0, 1].plot(t_min, T_i, color=cmap(norm(i)), lw=1.2)
axes[0, 1].set_xlabel("Time [min]")
axes[0, 1].set_ylabel("Layer-averaged temperature [K]")
axes[0, 1].set_title("Per-layer temperature")
axes[0, 1].grid(True, alpha=0.3)
cbar_T = fig.colorbar(sm, ax=axes[0, 1], pad=0.02)
cbar_T.set_label("Zone index (0 = cooled side)")

# (c) Per-layer current fraction overlaid, coloured by layer index
for i in range(num_layers):
    f_i = solution[f"Layer {i} current fraction"].data
    axes[1, 0].plot(t_min, f_i, color=cmap(norm(i)), lw=1.2)
axes[1, 0].axhline(
    1.0 / num_layers, color="k", ls="--", alpha=0.5, lw=1, label="even split"
)
axes[1, 0].set_xlabel("Time [min]")
axes[1, 0].set_ylabel("Current fraction")
axes[1, 0].set_title("Per-layer current fraction")
axes[1, 0].legend(loc="best")
axes[1, 0].grid(True, alpha=0.3)
cbar_f = fig.colorbar(sm, ax=axes[1, 0], pad=0.02)
cbar_f.set_label("Zone index (0 = cooled side)")

# (d) Through-stack profile at initial, middle, and final time
idx_snapshots = [0, len(t_s) // 2, -1]
for j in idx_snapshots:
    T_profile = [
        float(solution[f"Layer {i} average temperature [K]"].data[j])
        for i in range(num_layers)
    ]
    axes[1, 1].plot(
        np.arange(num_layers), T_profile, marker="o", label=f"t = {t_min[j]:.1f} min"
    )
axes[1, 1].set_xlabel("Zone index (0 = cooled side)")
axes[1, 1].set_ylabel("Layer-averaged temperature [K]")
axes[1, 1].set_title("Through-stack temperature profile")
axes[1, 1].legend(loc="best")
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle(
    f"MultiLayer3DThermalSPM — {model.num_subdivisions} zones x "
    f"{model.layers_per_zone} layers/zone = {num_physical_layers} "
    f"physical layers, {model.connection}, asymmetric cooling",
    fontsize=12,
)
plt.show()

# --------------------------------------------------------------- #
# 3D spatial plots at end of discharge
# --------------------------------------------------------------- #
# The built-in ``pybamm.plot_3d_heatmap`` / ``plot_3d_cross_section``
# helpers assume a single ``cell`` domain with ``dimensionality == 3``.
# This model instead uses one FEM mesh per ``cell layer i`` domain, so
# we build the 3-D visualisation directly from the per-layer meshes.
print("\nGenerating 3D spatial plots...")

t_final = solution.t[-1]

# Collect (x, y, z, T) for every FEM node across every layer. The mesh
# generator already places each layer at its correct absolute x, so we
# do NOT need to shift the x coordinates further.
all_x, all_y, all_z, all_T = [], [], [], []
total_nodes = 0
for i in range(num_layers):
    v_i = solution[f"Layer {i} temperature [K]"]
    nodes_i = v_i.mesh.nodes  # shape (n_nodes_i, 3), absolute coords
    total_nodes += nodes_i.shape[0]
    T_vals = v_i(t=t_final, x=nodes_i[:, 0], y=nodes_i[:, 1], z=nodes_i[:, 2])
    all_x.append(nodes_i[:, 0])
    all_y.append(nodes_i[:, 1])
    all_z.append(nodes_i[:, 2])
    all_T.append(np.asarray(T_vals).ravel())

print(
    f"  Total FEM nodes across {num_layers} zones: {total_nodes} "
    f"(~{total_nodes // num_layers} per zone)"
)

x_stack = np.concatenate(all_x)
y_all = np.concatenate(all_y)
z_all = np.concatenate(all_z)
T_all = np.concatenate(all_T)

# --- Figure 2a: 3D scatter of the full stack ---
fig2 = plt.figure(figsize=(12, 5.5), constrained_layout=True)

ax_3d = fig2.add_subplot(1, 2, 1, projection="3d")
sc = ax_3d.scatter(
    x_stack * 1e3,  # mm for readability
    y_all * 1e3,
    z_all * 1e3,
    c=T_all,
    cmap="inferno",
    s=10,
    alpha=0.8,
)
ax_3d.set_xlabel("x (through-stack) [mm]")
ax_3d.set_ylabel("y [mm]")
ax_3d.set_zlabel("z [mm]")
ax_3d.set_title(f"3D temperature at t = {t_final / 60:.1f} min\n(cold plate at x = 0)")
cb = fig2.colorbar(sc, ax=ax_3d, shrink=0.7, pad=0.08, label="T [K]")
cb.formatter.set_useOffset(False)
ax_3d.view_init(elev=18, azim=-60)

# --- Figure 2b: mid-plane slice (y ~ W/2) showing T(x_stack, z) ---
# Each layer's mesh is already at its absolute x-position in the stack,
# so the layer-centre x we record below is already on the global axis.
# Evaluate each layer at its own internal midpoint to avoid NaNs at
# mesh boundaries.
n_z = 40
T_slice = np.zeros((n_z, num_layers))
z_grid = None
xc = np.zeros(num_layers)
for i in range(num_layers):
    v_i = solution[f"Layer {i} temperature [K]"]
    nodes_i = v_i.mesh.nodes
    x_lo, x_hi = float(nodes_i[:, 0].min()), float(nodes_i[:, 0].max())
    y_lo, y_hi = float(nodes_i[:, 1].min()), float(nodes_i[:, 1].max())
    z_lo, z_hi = float(nodes_i[:, 2].min()), float(nodes_i[:, 2].max())
    if z_grid is None:
        eps = 1e-6 * (z_hi - z_lo)
        z_grid = np.linspace(z_lo + eps, z_hi - eps, n_z)
    x_mid = 0.5 * (x_lo + x_hi)
    y_mid = 0.5 * (y_lo + y_hi)
    T_slice[:, i] = np.asarray(
        v_i(
            t=t_final,
            x=np.full_like(z_grid, x_mid),
            y=np.full_like(z_grid, y_mid),
            z=z_grid,
        )
    ).ravel()
    xc[i] = x_mid  # absolute x of this layer's centre

X_grid, Z_grid = np.meshgrid(xc, z_grid)

ax_slice = fig2.add_subplot(1, 2, 2)
pcm = ax_slice.pcolormesh(
    X_grid * 1e3, Z_grid * 1e3, T_slice, cmap="inferno", shading="auto"
)
ax_slice.set_xlabel("x (through-stack) [mm]")
ax_slice.set_ylabel("z [mm]")
ax_slice.set_title(f"Mid-plane slice (y = W/2) at t = {t_final / 60:.1f} min")
cb2 = fig2.colorbar(pcm, ax=ax_slice, label="T [K]")
cb2.formatter.set_useOffset(False)

plt.show()
