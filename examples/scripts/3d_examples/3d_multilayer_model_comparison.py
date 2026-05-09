"""
Multilayer 3D-thermal model comparison: SPM vs SPMe vs DFN.

Compares the three electrochemistry fidelity levels available in the
multilayer thermal framework under asymmetric cooling (cold plate on the
left face, insulated elsewhere). The comparison shows how model fidelity
affects:

1. The predicted through-stack temperature gradient
2. Current redistribution between layers (parallel connection)
3. Terminal voltage

Configuration:
  - 120 physical layers, 10 zones (12 physical layers per zone)
  - Parallel connection
  - Asymmetric cooling: h_cold=50 W/m²K (left face), h_insulated=0.5 elsewhere
  - 3C discharge to 2.8 V cutoff
  - Contact resistance: 5e-3 K.m²/W

Outputs:
  - Static comparison plots (temperature, spread, current, voltage)
  - Animated GIF showing temperature profile & current distribution evolving
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import pybamm

print("=" * 65)
print(
    "SPMe and DFN model variant comparison (120 layers, 10 zones, asymmetric cooling)"
)
print("=" * 65)

NUM_PHYSICAL_LAYERS = 120
NUM_SUBDIVISIONS = 10  # 12 physical layers per zone


def run_model_comparison_asymmetric():
    """Compare SPM, SPMe, and DFN with asymmetric cooling to show
    temperature gradients and current redistribution differences."""
    results = {}
    var_pts_small = {
        "x_n": 10,
        "x_s": 10,
        "x_p": 10,
        "r_n": 15,
        "r_p": 15,
        "x": None,
        "y": None,
        "z": None,
    }
    # Higher C-rate discharge to cutoff voltage
    exp_compare = pybamm.Experiment([("Discharge at 3C until 2.8V",)])

    model_classes = {
        "SPM": pybamm.lithium_ion.MultiLayer3DThermalSPM,
        "SPMe": pybamm.lithium_ion.MultiLayer3DThermalSPMe,
        "DFN": pybamm.lithium_ion.MultiLayer3DThermalDFN,
    }

    for name, ModelClass in model_classes.items():
        print(f"  Building and solving {name}...")
        model_cmp = ModelClass(
            num_physical_layers=NUM_PHYSICAL_LAYERS,
            num_subdivisions=NUM_SUBDIVISIONS,
            connection="parallel",
        )
        param_cmp = pybamm.ParameterValues("Marquis2019")
        model_cmp.apply_stack_scaling(param_cmp, verbose=False)

        # Asymmetric cooling: cold plate on left (x_min), insulated elsewhere
        h_cold_cmp = 50.0
        h_insulated_cmp = 0.5
        param_cmp.update(
            {
                "Total heat transfer coefficient [W.m-2.K-1]": h_insulated_cmp,
                "Left face heat transfer coefficient [W.m-2.K-1]": h_cold_cmp,
                "Right face heat transfer coefficient [W.m-2.K-1]": h_insulated_cmp,
                "Front face heat transfer coefficient [W.m-2.K-1]": h_insulated_cmp,
                "Back face heat transfer coefficient [W.m-2.K-1]": h_insulated_cmp,
                "Bottom face heat transfer coefficient [W.m-2.K-1]": h_insulated_cmp,
                "Top face heat transfer coefficient [W.m-2.K-1]": h_insulated_cmp,
            }
        )
        # Moderate contact resistance to allow some temperature difference
        param_cmp.update({model_cmp.CONTACT_RESISTANCE_PARAM: 5e-3})

        sim_cmp = pybamm.Simulation(
            model_cmp,
            parameter_values=param_cmp,
            var_pts=var_pts_small,
            experiment=exp_compare,
        )
        sol_cmp = sim_cmp.solve()
        results[name] = sol_cmp

    return results


results_compare = run_model_comparison_asymmetric()

# --------------------------------------------------------------- #
# Print summary
# --------------------------------------------------------------- #
print("\n--- End-of-discharge comparison (3C, asymmetric cooling) ---")
print(f"{'Model':<6} {'Voltage':>8} {'T_spread':>9} {'T_min':>8} {'T_max':>8}")
for name, sol in results_compare.items():
    V = float(sol["Voltage [V]"].data[-1])
    T_spread = float(sol["Temperature spread [K]"].data[-1])
    T_min = float(sol["Minimum layer-averaged temperature [K]"].data[-1])
    T_max = float(sol["Maximum layer-averaged temperature [K]"].data[-1])
    print(f"  {name:<4} {V:8.4f} V {T_spread:7.3f} K  {T_min:7.2f} K {T_max:7.2f} K")

print("\n--- Per-layer temperature at end of discharge ---")
print(f"{'Layer':<6}", end="")
for name in results_compare:
    print(f"  {name:>10}", end="")
print()
for i in range(NUM_SUBDIVISIONS):
    print(f"  {i:<4}", end="")
    for _name, sol in results_compare.items():
        T_i = float(sol[f"Layer {i} average temperature [K]"].data[-1])
        print(f"  {T_i:8.3f} K", end="")
    print()

print("\n--- Per-layer current fraction at end of discharge ---")
print(f"{'Layer':<6}", end="")
for name in results_compare:
    print(f"  {name:>10}", end="")
print()
for i in range(NUM_SUBDIVISIONS):
    print(f"  {i:<4}", end="")
    for _name, sol in results_compare.items():
        f_i = float(sol[f"Layer {i} current fraction"].data[-1])
        print(f"  {f_i:10.6f}", end="")
    print()

# --------------------------------------------------------------- #
# Figure: Temperature gradient comparison
# --------------------------------------------------------------- #
print("\nGenerating comparison plots...")

fig_cmp, axes_cmp = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
colors_cmp = {"SPM": "C0", "SPMe": "C1", "DFN": "C2"}

# (a) Per-layer temperature vs time (all models overlaid)
ax = axes_cmp[0, 0]
for name, sol in results_compare.items():
    t = sol["Time [s]"].data
    for i in range(NUM_SUBDIVISIONS):
        T_i = sol[f"Layer {i} average temperature [K]"].data
        label = f"{name}" if i == 0 else None
        alpha = max(0.3, 1.0 - 0.7 * i / max(NUM_SUBDIVISIONS - 1, 1))
        ax.plot(t, T_i, color=colors_cmp[name], alpha=alpha, lw=1.3, label=label)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Layer-averaged temperature [K]")
ax.set_title("Per-layer temperature evolution")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# (b) Temperature spread vs time
ax = axes_cmp[0, 1]
for name, sol in results_compare.items():
    t = sol["Time [s]"].data
    spread = sol["Temperature spread [K]"].data
    ax.plot(t, spread, color=colors_cmp[name], label=name, lw=1.8)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Temperature spread [K]")
ax.set_title("Through-stack temperature gradient (T_max - T_min)")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)

# (c) Current fraction vs time (all layers, all models)
ax = axes_cmp[1, 0]
for name, sol in results_compare.items():
    t = sol["Time [s]"].data
    for i in range(NUM_SUBDIVISIONS):
        f_i = sol[f"Layer {i} current fraction"].data
        alpha = max(0.3, 1.0 - 0.7 * i / max(NUM_SUBDIVISIONS - 1, 1))
        label = f"{name}" if i == 0 else None
        ax.plot(t, f_i, color=colors_cmp[name], alpha=alpha, lw=1.3, label=label)
ax.axhline(
    1.0 / NUM_SUBDIVISIONS, color="k", ls="--", alpha=0.4, lw=1, label="even split"
)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Current fraction")
ax.set_title("Per-layer current distribution")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

# (d) Through-stack profile at final time (bar chart)
ax = axes_cmp[1, 1]
layer_indices = np.arange(NUM_SUBDIVISIONS)
bar_width = 0.25
for j, (name, sol) in enumerate(results_compare.items()):
    T_profile = [
        float(sol[f"Layer {i} average temperature [K]"].data[-1])
        for i in range(NUM_SUBDIVISIONS)
    ]
    ax.bar(
        layer_indices + j * bar_width,
        T_profile,
        bar_width,
        label=name,
        color=colors_cmp[name],
        alpha=0.8,
    )
ax.set_xlabel("Layer index (0 = cooled side)")
ax.set_ylabel("Layer-averaged temperature [K]")
ax.set_title("Final temperature profile across stack")
ax.set_xticks(layer_indices + bar_width)
ax.set_xticklabels([str(i) for i in layer_indices])
ax.legend(loc="best")
ax.grid(True, alpha=0.3, axis="y")

fig_cmp.suptitle(
    f"Model fidelity comparison \u2014 {NUM_PHYSICAL_LAYERS} physical layers, "
    f"{NUM_SUBDIVISIONS} zones, parallel, asymmetric cooling (3C to 2.8V)",
    fontsize=12,
)
plt.show()

# --------------------------------------------------------------- #
# Figure: Current redistribution detail
# --------------------------------------------------------------- #
fig_curr, axes_curr = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

# (a) Current fraction difference from uniform (shows redistribution magnitude)
ax = axes_curr[0]
uniform = 1.0 / NUM_SUBDIVISIONS
for name, sol in results_compare.items():
    t = sol["Time [s]"].data
    # Layer 0 (cold side) should draw more current
    f_cold = sol["Layer 0 current fraction"].data
    f_hot = sol[f"Layer {NUM_SUBDIVISIONS - 1} current fraction"].data
    ax.plot(
        t,
        (f_cold - uniform) * 100,
        color=colors_cmp[name],
        lw=1.8,
        label=f"{name} (cold layer)",
    )
    ax.plot(
        t, (f_hot - uniform) * 100, color=colors_cmp[name], lw=1.8, ls="--", alpha=0.6
    )
ax.axhline(0, color="k", ls="-", alpha=0.3, lw=0.8)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Deviation from uniform [%]")
ax.set_title("Current redistribution\n(solid = cold layer 0, dashed = hot layer)")
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.3)

# (b) Voltage curves
ax = axes_curr[1]
for name, sol in results_compare.items():
    t = sol["Time [s]"].data
    V = sol["Voltage [V]"].data
    ax.plot(t, V, color=colors_cmp[name], label=name, lw=1.8)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Voltage [V]")
ax.set_title("Terminal voltage")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)

fig_curr.suptitle(
    "Current redistribution and voltage under asymmetric cooling",
    fontsize=12,
)
plt.show()

# =============================================================== #
# Animated GIF: Temperature profile & current distribution vs time
# =============================================================== #
# Creates an animated GIF showing:
#   Left pane:  Through-stack temperature profile (layer index vs T)
#               with one line per model (SPM, SPMe, DFN), evolving in time
#   Right pane: Voltage vs time with a moving red vertical line
#               indicating the current frame time
#
# This follows PyBaMM's create_gif pattern (FuncAnimation + ani.save).

print("\n" + "=" * 65)
print("Generating animated GIF...")
print("=" * 65)

# Use the results_compare dict from the asymmetric comparison above
t_arrays = {name: sol["Time [s]"].data for name, sol in results_compare.items()}
# Use SPM time array as reference (all should be similar)
t_ref = t_arrays["SPM"]
num_frames = 60
frame_times = np.linspace(t_ref[0], t_ref[-1], num_frames)

# Pre-extract data for speed
T_data = {}  # T_data[name][layer_idx] = array of T values over time
f_data = {}  # f_data[name][layer_idx] = array of current fraction values
V_data = {}  # V_data[name] = array of voltage values

for name, sol in results_compare.items():
    T_data[name] = [
        sol[f"Layer {i} average temperature [K]"].data for i in range(NUM_SUBDIVISIONS)
    ]
    f_data[name] = [
        sol[f"Layer {i} current fraction"].data for i in range(NUM_SUBDIVISIONS)
    ]
    V_data[name] = sol["Voltage [V]"].data

colors_gif = {"SPM": "C0", "SPMe": "C1", "DFN": "C2"}
layer_x = np.arange(NUM_SUBDIVISIONS)

# Set up the figure
fig_gif, (ax_temp, ax_curr, ax_volt) = plt.subplots(
    1, 3, figsize=(14, 4.5), constrained_layout=True
)

# Initialize temperature profile plot
temp_lines = {}
for name in results_compare:
    (line,) = ax_temp.plot(
        layer_x,
        [298.15] * NUM_SUBDIVISIONS,
        "o-",
        color=colors_gif[name],
        lw=2,
        markersize=6,
        label=name,
    )
    temp_lines[name] = line
ax_temp.set_xlabel("Layer index (0 = cooled side)")
ax_temp.set_ylabel("Layer-averaged temperature [K]")
ax_temp.set_title("Through-stack temperature profile")
ax_temp.set_xticks(layer_x)
ax_temp.legend(loc="upper left", fontsize=9)
ax_temp.grid(True, alpha=0.3)
ax_temp.ticklabel_format(useOffset=False, axis="y")

# Initialize current fraction plot
curr_lines = {}
for name in results_compare:
    (line,) = ax_curr.plot(
        layer_x,
        [1.0 / NUM_SUBDIVISIONS] * NUM_SUBDIVISIONS,
        "s-",
        color=colors_gif[name],
        lw=2,
        markersize=6,
        label=name,
    )
    curr_lines[name] = line
ax_curr.axhline(1.0 / NUM_SUBDIVISIONS, color="k", ls="--", alpha=0.4, lw=1)
ax_curr.set_xlabel("Layer index (0 = cooled side)")
ax_curr.set_ylabel("Current fraction")
ax_curr.set_title("Per-layer current distribution")
ax_curr.set_xticks(layer_x)
ax_curr.legend(loc="upper left", fontsize=9)
ax_curr.grid(True, alpha=0.3)

# Initialize voltage vs time plot (static lines + moving marker)
for name, _sol in results_compare.items():
    ax_volt.plot(
        t_arrays[name], V_data[name], color=colors_gif[name], lw=1.5, label=name
    )
vline = ax_volt.axvline(0, color="red", lw=2, alpha=0.8)
ax_volt.set_xlabel("Time [s]")
ax_volt.set_ylabel("Voltage [V]")
ax_volt.set_title("Terminal voltage")
ax_volt.legend(loc="upper right", fontsize=9)
ax_volt.grid(True, alpha=0.3)

time_text = fig_gif.suptitle("t = 0.0 s", fontsize=12)


def interpolate_at_time(data_array, t_array, t_target):
    """Linearly interpolate a data array at a target time."""
    return float(np.interp(t_target, t_array, data_array))


def update_frame(frame_idx):
    """Update all axes for a given frame (called by FuncAnimation)."""
    t_now = frame_times[frame_idx]

    # Update temperature profiles
    T_all_now = []
    for name in results_compare:
        T_profile = [
            interpolate_at_time(T_data[name][i], t_arrays[name], t_now)
            for i in range(NUM_SUBDIVISIONS)
        ]
        temp_lines[name].set_ydata(T_profile)
        T_all_now.extend(T_profile)

    # Auto-scale y-axis for temperature
    if T_all_now:
        T_min_now = min(T_all_now) - 0.05
        T_max_now = max(T_all_now) + 0.05
        ax_temp.set_ylim(T_min_now, T_max_now)

    # Update current fraction profiles
    f_all_now = []
    for name in results_compare:
        f_profile = [
            interpolate_at_time(f_data[name][i], t_arrays[name], t_now)
            for i in range(NUM_SUBDIVISIONS)
        ]
        curr_lines[name].set_ydata(f_profile)
        f_all_now.extend(f_profile)

    # Auto-scale y-axis for current fraction
    if f_all_now:
        f_min_now = min(f_all_now) - 0.001
        f_max_now = max(f_all_now) + 0.001
        ax_curr.set_ylim(f_min_now, f_max_now)

    # Update voltage time marker
    vline.set_xdata([t_now, t_now])

    # Update title with current time
    time_text.set_text(f"t = {t_now:.1f} s")

    return list(temp_lines.values()) + list(curr_lines.values()) + [vline, time_text]


# Create and save the animation
ani = FuncAnimation(
    fig_gif,
    update_frame,
    frames=num_frames,
    interval=100,  # 100ms between frames
    blit=False,
)

gif_filename = "multilayer_model_comparison.gif"
ani.save(gif_filename, dpi=150, writer="pillow")
print(f"  GIF saved: {gif_filename}")
print(f"  Frames: {num_frames}, Duration: {num_frames * 0.1:.1f} s")
print("  The GIF shows how the temperature gradient and current")
print("  redistribution evolve differently across SPM, SPMe, and DFN.")
plt.close(fig_gif)
