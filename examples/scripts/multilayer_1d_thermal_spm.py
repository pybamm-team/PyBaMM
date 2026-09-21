"""
Multilayer 1D thermal SPM — through-stack T(x) profile.

This example demonstrates the MultiLayer1DThermalSPM model which gives
full 1D thermal resolution within each cell layer. Unlike the 3D thermal
model (which has only 2 x-nodes per layer due to mesh_h >> L_x), this
model resolves the intra-cell temperature gradient using the standard
finite-volume discretisation with var_pts resolution.

Configuration:
  - 6 layers, parallel connection
  - Symmetric cooling: h=50 W/m²K on both faces
  - 3C discharge
  - Contact resistance: 5e-3 K.m²/W
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pybamm

NUM_LAYERS = 6

print("=" * 65)
print("MultiLayer1DThermalSPM — through-stack T(x)")
print("=" * 65)

model = pybamm.lithium_ion.MultiLayer1DThermalSPM(
    num_physical_layers=NUM_LAYERS,
    num_subdivisions=NUM_LAYERS,
    connection="parallel",
)

param = pybamm.ParameterValues("Marquis2019")
model.apply_stack_scaling(param, verbose=True)

h_sym = 50.0
param.update(
    {
        "Left face heat transfer coefficient [W.m-2.K-1]": h_sym,
        "Right face heat transfer coefficient [W.m-2.K-1]": h_sym,
        model.CONTACT_RESISTANCE_PARAM: 5e-3,
        # Set 3C discharge current (positive = discharge in PyBaMM convention)
        "Current function [A]": 3 * param["Nominal cell capacity [A.h]"],
    }
)

solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
sim = pybamm.Simulation(model, parameter_values=param, solver=solver)

print("\nSolving...")
sol = sim.solve(t_eval=np.linspace(0, 1100, 100))
print(f"Solve complete. Final t = {sol.t[-1]:.1f} s")

# --------------------------------------------------------------- #
# Extract T(x) through the full stack
# --------------------------------------------------------------- #
# Each layer's "cell temperature" lives on the standard electrode
# domains (neg|sep|pos) with x in [0, L_x]. To plot T(x) through
# the full stack, we offset each layer's x by i * L_x.
L_n = float(param["Negative electrode thickness [m]"])
L_s = float(param["Separator thickness [m]"])
L_p = float(param["Positive electrode thickness [m]"])
L_x = L_n + L_s + L_p
L_cc_n = float(param["Negative current collector thickness [m]"])
L_cc_p = float(param["Positive current collector thickness [m]"])
L_cell = L_cc_n + L_x + L_cc_p  # Full unit cell including CCs

print(f"\n  L_x (electrode sandwich) = {L_x * 1e6:.1f} µm")
print(f"  L_cell (with CCs) = {L_cell * 1e6:.1f} µm")
print(f"  Total stack = {NUM_LAYERS * L_cell * 1e3:.3f} mm")

# Time snapshots
t_end = sol.t[-1]
t_snapshots = [0.0, 0.25 * t_end, 0.5 * t_end, 0.75 * t_end, t_end]

fig, (ax_t, ax_v) = plt.subplots(
    1,
    2,
    figsize=(14, 5.5),
    gridspec_kw={"width_ratios": [3, 1]},
    constrained_layout=True,
)

colors = plt.cm.plasma(np.linspace(0, 1, len(t_snapshots)))

for t_snap, color in zip(t_snapshots, colors, strict=False):
    x_stack = []
    T_stack = []

    for i in range(NUM_LAYERS):
        # Get the 1D temperature data for this layer
        T_var = sol[f"Layer {i} cell temperature [K]"]
        T_data = T_var(t=t_snap)

        # The x-coordinates for the electrode sandwich are [0, L_x]
        # Each layer is offset in the stack by i * L_cell
        n_pts = len(T_data)
        x_local = np.linspace(0, L_x, n_pts)
        x_global = x_local + i * L_cell + L_cc_n  # offset past neg CC

        # Add CC node temperatures as single points
        T_cn = float(sol[f"Layer {i} negative CC temperature [K]"](t=t_snap))
        T_cp = float(sol[f"Layer {i} positive CC temperature [K]"](t=t_snap))

        # Negative CC point
        x_cc_n = i * L_cell + L_cc_n / 2
        x_stack.append(x_cc_n)
        T_stack.append(T_cn - 273.15)

        # Electrode sandwich points
        x_stack.extend(x_global)
        T_stack.extend(T_data - 273.15)

        # Positive CC point
        x_cc_p = (i + 1) * L_cell - L_cc_p / 2
        x_stack.append(x_cc_p)
        T_stack.append(T_cp - 273.15)

    x_stack = np.array(x_stack) * 1e3  # Convert to mm
    T_stack = np.array(T_stack)

    ax_t.plot(
        x_stack,
        T_stack,
        "o-",
        color=color,
        markersize=2,
        lw=1.2,
        label=f"t = {t_snap:.0f} s",
    )

# Layer boundaries
for i in range(1, NUM_LAYERS):
    ax_t.axvline(i * L_cell * 1e3, color="gray", alpha=0.2, lw=0.5, ls="--")

ax_t.set_xlabel("Position through stack [mm]")
ax_t.set_ylabel("Temperature [°C]")
ax_t.set_title(
    f"Through-stack T(x) — {NUM_LAYERS}-layer 1D thermal SPM\n"
    f"(symmetric cooling h={h_sym} W/m²K, 3C discharge)"
)
ax_t.legend(fontsize=9)
ax_t.grid(True, alpha=0.3)

# Voltage panel
t_data = sol["Time [s]"].data
V_data = sol["Voltage [V]"].data
ax_v.plot(t_data, V_data, "k-", lw=1.5)
for t_snap, color in zip(t_snapshots, colors, strict=False):
    ax_v.axvline(t_snap, color=color, lw=1.5, alpha=0.7)
ax_v.set_xlabel("Time [s]")
ax_v.set_ylabel("Voltage [V]")
ax_v.set_title("Voltage")
ax_v.grid(True, alpha=0.3)

outpath = "multilayer_1d_thermal_spm_temperature.png"
plt.savefig(outpath, dpi=150)
print(f"\nSaved: {outpath}")

# Print summary
print("\n--- End-of-discharge summary ---")
for i in range(NUM_LAYERS):
    T_av = float(sol[f"Layer {i} average temperature [K]"].data[-1])
    print(f"  Layer {i}: T_av = {T_av:.3f} K ({T_av - 273.15:.2f} °C)")

T_avs = [
    float(sol[f"Layer {i} average temperature [K]"].data[-1]) for i in range(NUM_LAYERS)
]
print(f"  Stack spread: {max(T_avs) - min(T_avs):.3f} K")
print(f"  Points per layer: {len(sol['Layer 0 cell temperature [K]'](t=0))}")
