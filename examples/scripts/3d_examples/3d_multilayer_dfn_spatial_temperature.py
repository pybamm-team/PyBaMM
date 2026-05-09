"""
Multilayer DFN with full layer resolution — spatial T(x) through the stack.

This example uses a DFN model with 30 layers where EVERY layer is
individually resolved (no coarsening). The key output is the 3D
temperature field ``Layer i temperature [K]`` which lives on a FEM
mesh for each layer. By evaluating this field along the stack
x-direction at the midplane (y=L_y/2, z=L_z/2), we can visualise
the continuous through-stack temperature profile — including the
intra-cell gradient within each layer.

Symmetric cooling (same h on both x-faces) so the profile is
symmetric about the stack centre, making intra-layer gradients
easier to see.

Configuration:
  - 30 physical layers, each individually resolved (layers_per_zone=1)
  - DFN electrochemistry (spatially-resolved particles, electrolyte PDE,
    solid/electrolyte charge conservation, implicit Butler-Volmer)
  - Symmetric cooling: h=50 W/m²K on both left and right faces
  - 3C discharge to 2.8 V cutoff
  - Contact resistance: 5e-3 K.m²/W
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

import pybamm

print("=" * 65)
print("DFN full-resolution — spatial T(x) through stack")
print("=" * 65)

NUM_LAYERS_FULL = 30  # Every layer individually resolved
print(f"  Building DFN with {NUM_LAYERS_FULL} individually-resolved layers...")

model_full = pybamm.lithium_ion.MultiLayer3DThermalDFN(
    num_physical_layers=NUM_LAYERS_FULL,
    num_subdivisions=NUM_LAYERS_FULL,  # No coarsening: 1 physical cell per zone
    connection="parallel",
    mesh_h=0.03,  # Coarse FEM mesh to keep runtime tractable
)

param_full = pybamm.ParameterValues("Marquis2019")
model_full.apply_stack_scaling(param_full, verbose=True)

# Symmetric cooling: same h on left and right x-faces (cold plates on both sides)
h_sym = 50.0  # W.m-2.K-1 — active cooling on both faces
h_other = 0.5  # W.m-2.K-1 — mild convection on other faces
param_full.update(
    {
        "Total heat transfer coefficient [W.m-2.K-1]": h_other,
        "Left face heat transfer coefficient [W.m-2.K-1]": h_sym,
        "Right face heat transfer coefficient [W.m-2.K-1]": h_sym,
        "Front face heat transfer coefficient [W.m-2.K-1]": h_other,
        "Back face heat transfer coefficient [W.m-2.K-1]": h_other,
        "Bottom face heat transfer coefficient [W.m-2.K-1]": h_other,
        "Top face heat transfer coefficient [W.m-2.K-1]": h_other,
    }
)
# Moderate contact resistance
param_full.update({model_full.CONTACT_RESISTANCE_PARAM: 5e-3})

exp_full = pybamm.Experiment([("Discharge at 3C until 2.8V",)])

var_pts_full = {
    "x_n": 10,
    "x_s": 10,
    "x_p": 10,
    "r_n": 15,
    "r_p": 15,
    "x": None,
    "y": None,
    "z": None,
}

sim_full = pybamm.Simulation(
    model_full,
    parameter_values=param_full,
    var_pts=var_pts_full,
    experiment=exp_full,
)

print("  Solving (this may take a few minutes with 30 DFN layers)...")
sol_full = sim_full.solve()
print(f"  Solve complete. Final t = {sol_full.t[-1]:.1f} s")

# --------------------------------------------------------------- #
# Extract cell geometry for x-coordinate calculations
# --------------------------------------------------------------- #
L_n = float(param_full["Negative electrode thickness [m]"])
L_s = float(param_full["Separator thickness [m]"])
L_p = float(param_full["Positive electrode thickness [m]"])
L_cc_n = float(param_full["Negative current collector thickness [m]"])
L_cc_p = float(param_full["Positive current collector thickness [m]"])
L_cell = L_cc_n + L_n + L_s + L_p + L_cc_p  # Single unit cell thickness
L_y_val = float(param_full["Electrode width [m]"])
L_z_val = float(param_full["Electrode height [m]"])

print(f"  Unit cell thickness: {L_cell * 1e3:.4f} mm")
print(f"  Total stack thickness: {NUM_LAYERS_FULL * L_cell * 1e3:.2f} mm")

# --------------------------------------------------------------- #
# Build spatial T(x) profile at the midplane for selected times
# --------------------------------------------------------------- #
# For each layer, the 3D temperature variable lives on a FEM mesh
# spanning x in [i*L_cell, (i+1)*L_cell]. We evaluate it along a
# line at y = L_y/2, z = L_z/2 to get the through-thickness profile.
print("\n  Extracting spatial T(x) profile at midplane...")

t_end = sol_full.t[-1]
# Select time snapshots: start, 25%, 50%, 75%, end
t_snapshots = [0.0, 0.25 * t_end, 0.5 * t_end, 0.75 * t_end, t_end]

# Number of x-points to sample within each layer
nx_per_layer = 5
y_mid = L_y_val / 2.0
z_mid = L_z_val / 2.0

fig_tx, (ax_tx, ax_volt) = plt.subplots(
    1,
    2,
    figsize=(14, 5.5),
    gridspec_kw={"width_ratios": [3, 1]},
    constrained_layout=True,
)

cmap_time = cm.plasma
norm_time = Normalize(vmin=0, vmax=t_end)
sm_time = cm.ScalarMappable(cmap=cmap_time, norm=norm_time)
sm_time.set_array([])

for t_snap in t_snapshots:
    x_all = []
    T_all = []

    for i in range(NUM_LAYERS_FULL):
        # x-range for this layer
        x_min_layer = i * L_cell
        x_max_layer = (i + 1) * L_cell
        # Sample points (avoid exact boundaries to stay within the mesh)
        x_pts = np.linspace(
            x_min_layer + 0.05 * L_cell,
            x_max_layer - 0.05 * L_cell,
            nx_per_layer,
        )
        y_pts = np.full_like(x_pts, y_mid)
        z_pts = np.full_like(x_pts, z_mid)

        # Evaluate the 3D temperature field at these points
        T_var = sol_full[f"Layer {i} temperature [K]"]
        T_vals = T_var(t=t_snap, x=x_pts, y=y_pts, z=z_pts)

        x_all.extend(x_pts)
        T_all.extend(T_vals)

    x_all = np.array(x_all)
    T_all = np.array(T_all) - 273.15  # Convert to Celsius
    # Fill any NaN from interpolation edge effects
    mask = np.isnan(T_all)
    if mask.any() and not mask.all():
        T_all[mask] = np.interp(
            np.flatnonzero(mask), np.flatnonzero(~mask), T_all[~mask]
        )

    color = cmap_time(norm_time(t_snap))
    ax_tx.plot(
        x_all * 1e3,  # Convert to mm
        T_all,
        color=color,
        lw=1.5,
        label=f"t = {t_snap:.0f} s",
    )

ax_tx.set_xlabel("Position through stack [mm]")
ax_tx.set_ylabel("Temperature [\u00b0C]")
ax_tx.set_title(
    f"Spatial T(x) through {NUM_LAYERS_FULL}-layer DFN stack\n"
    f"(midplane y={L_y_val * 1e3:.0f}/2 mm, z={L_z_val * 1e3:.0f}/2 mm, "
    f"symmetric cooling h={h_sym} W/m\u00b2K)"
)
ax_tx.legend(loc="upper left", fontsize=8)
ax_tx.grid(True, alpha=0.3)
fig_tx.colorbar(sm_time, ax=ax_tx, label="Time [s]")

# Add layer boundary markers (faint vertical lines)
for i in range(1, NUM_LAYERS_FULL):
    ax_tx.axvline(i * L_cell * 1e3, color="gray", alpha=0.15, lw=0.5)

# Voltage vs time in the side panel
t_full = sol_full["Time [s]"].data
V_full = sol_full["Voltage [V]"].data
ax_volt.plot(t_full, V_full, "k-", lw=1.5)
for t_snap in t_snapshots:
    color = cmap_time(norm_time(t_snap))
    ax_volt.axvline(t_snap, color=color, lw=1.5, alpha=0.7)
ax_volt.set_xlabel("Time [s]")
ax_volt.set_ylabel("Terminal voltage [V]")
ax_volt.set_title("Voltage")
ax_volt.grid(True, alpha=0.3)

plt.savefig("multilayer_dfn_spatial_temperature.png", dpi=150)
print("  Saved: multilayer_dfn_spatial_temperature.png")
plt.show()

# --------------------------------------------------------------- #
# Animated GIF: T(x) evolving over time
# --------------------------------------------------------------- #
print("\n  Generating animated T(x) GIF...")

num_frames_tx = 60
t_frames_tx = np.linspace(0, t_end, num_frames_tx)

fig_gif_tx, (ax_gif_tx, ax_gif_v) = plt.subplots(
    1,
    2,
    figsize=(13, 5),
    gridspec_kw={"width_ratios": [3, 1]},
    constrained_layout=True,
)

# Pre-compute T(x) for all frames to set axis limits
print("    Pre-computing temperature profiles for animation...")
all_T_profiles = []
all_x_profiles = []

for frame_t in t_frames_tx:
    x_frame = []
    T_frame = []
    for i in range(NUM_LAYERS_FULL):
        x_min_layer = i * L_cell
        x_max_layer = (i + 1) * L_cell
        x_pts = np.linspace(
            x_min_layer + 0.05 * L_cell,
            x_max_layer - 0.05 * L_cell,
            nx_per_layer,
        )
        y_pts = np.full_like(x_pts, y_mid)
        z_pts = np.full_like(x_pts, z_mid)
        T_var = sol_full[f"Layer {i} temperature [K]"]
        T_vals = T_var(t=frame_t, x=x_pts, y=y_pts, z=z_pts)
        x_frame.extend(x_pts)
        T_frame.extend(T_vals)
    x_arr = np.array(x_frame) * 1e3  # mm
    T_arr = np.array(T_frame) - 273.15  # Celsius
    # Replace any NaN from interpolation with linear fill from neighbours
    mask = np.isnan(T_arr)
    if mask.any() and not mask.all():
        T_arr[mask] = np.interp(
            np.flatnonzero(mask), np.flatnonzero(~mask), T_arr[~mask]
        )
    all_x_profiles.append(x_arr)
    all_T_profiles.append(T_arr)

T_min_all = min(np.nanmin(T) for T in all_T_profiles)
T_max_all = max(np.nanmax(T) for T in all_T_profiles)
T_pad = max((T_max_all - T_min_all) * 0.05, 0.1)

# Set up initial plot
(line_tx,) = ax_gif_tx.plot([], [], "C0-", lw=1.8)
ax_gif_tx.set_xlim(0, NUM_LAYERS_FULL * L_cell * 1e3)
ax_gif_tx.set_ylim(T_min_all - T_pad, T_max_all + T_pad)
ax_gif_tx.set_xlabel("Position through stack [mm]")
ax_gif_tx.set_ylabel("Temperature [\u00b0C]")
ax_gif_tx.set_title(
    f"DFN {NUM_LAYERS_FULL}-layer spatial T(x) \u2014 symmetric cooling, 3C discharge"
)
ax_gif_tx.grid(True, alpha=0.3)
# Layer boundaries
for i in range(1, NUM_LAYERS_FULL):
    ax_gif_tx.axvline(i * L_cell * 1e3, color="gray", alpha=0.15, lw=0.5)

# Voltage panel
ax_gif_v.plot(t_full, V_full, "k-", lw=1.2, alpha=0.5)
(vline_tx,) = ax_gif_v.plot(
    [0, 0], [V_full.min() - 0.05, V_full.max() + 0.05], "r-", lw=2
)
ax_gif_v.set_xlim(0, t_end * 1.02)
ax_gif_v.set_ylim(V_full.min() - 0.05, V_full.max() + 0.05)
ax_gif_v.set_xlabel("Time [s]")
ax_gif_v.set_ylabel("Voltage [V]")
ax_gif_v.set_title("Voltage")
ax_gif_v.grid(True, alpha=0.3)

time_text_tx = ax_gif_tx.text(
    0.02,
    0.95,
    "",
    transform=ax_gif_tx.transAxes,
    fontsize=10,
    verticalalignment="top",
    fontweight="bold",
)


def update_frame_tx(frame_idx):
    t_now = t_frames_tx[frame_idx]
    line_tx.set_data(all_x_profiles[frame_idx], all_T_profiles[frame_idx])
    vline_tx.set_xdata([t_now, t_now])
    time_text_tx.set_text(f"t = {t_now:.1f} s")
    return [line_tx, vline_tx, time_text_tx]


ani_tx = FuncAnimation(
    fig_gif_tx,
    update_frame_tx,
    frames=num_frames_tx,
    interval=100,
    blit=False,
)

gif_filename_tx = "multilayer_dfn_spatial_T.gif"
ani_tx.save(gif_filename_tx, dpi=150, writer="pillow")
print(f"  GIF saved: {gif_filename_tx}")
print(f"  Frames: {num_frames_tx}, Duration: {num_frames_tx * 0.1:.1f} s")
print("  The GIF shows the continuous T(x) profile through all 30 DFN layers,")
print("  revealing intra-cell temperature gradients and inter-layer jumps.")
plt.close(fig_gif_tx)

# --------------------------------------------------------------- #
# Print final summary
# --------------------------------------------------------------- #
print("\n--- End-of-discharge summary ---")
for i in range(NUM_LAYERS_FULL):
    T_i = float(sol_full[f"Layer {i} average temperature [K]"].data[-1])
    f_i = float(sol_full[f"Layer {i} current fraction"].data[-1])
    if i == 0 or i == NUM_LAYERS_FULL - 1 or i == NUM_LAYERS_FULL // 2:
        print(
            f"  Layer {i:2d}: T_av = {T_i:.3f} K ({T_i - 273.15:.2f} \u00b0C), "
            f"current fraction = {f_i:.5f}"
        )
T_max_full = float(sol_full["Maximum layer-averaged temperature [K]"].data[-1])
T_min_full = float(sol_full["Minimum layer-averaged temperature [K]"].data[-1])
spread_full = float(sol_full["Temperature spread [K]"].data[-1])
print(f"  Stack spread: {spread_full:.3f} K")
print(
    f"  T_min = {T_min_full:.3f} K ({T_min_full - 273.15:.2f} \u00b0C) \u2014 edge layers"
)
print(
    f"  T_max = {T_max_full:.3f} K ({T_max_full - 273.15:.2f} \u00b0C) \u2014 centre layers"
)
V_end_full = float(sol_full["Voltage [V]"].data[-1])
print(f"  Terminal voltage (end): {V_end_full:.3f} V")
