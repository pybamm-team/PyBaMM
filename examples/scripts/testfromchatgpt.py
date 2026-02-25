import matplotlib.pyplot as plt

import pybamm

# --------------------
# Basic settings
# --------------------
pybamm.set_logging_level("NOTICE")

N_cycles = 200

# Simple CCCV cycling protocol
experiment = pybamm.Experiment(
    [
        (
            "Discharge at C/8 until 3.2 V",
            "Rest for 15 minutes",
            "Charge at C/6 until 4.1 V",
            "Hold at 4.1 V until C/37",
            "Rest for 15 minutes",
        )
    ]
    * N_cycles
)

# Parameter set and a slightly boosted SEI rate to see capacity fade
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update({"SEI kinetic rate constant [m.s-1]": 1e-15})

# Start from 100% SOC (as in PyBaMM "simulating long experiments" example)
parameter_values.set_initial_stoichiometries(1.0)

# --------------------
# SEI mechanisms to compare
# --------------------
sei_mechanisms = {
    "reaction limited": {
        "SEI": "reaction limited",
    },
    "solvent-diffusion limited": {
        "SEI": "solvent-diffusion limited",
    },
    "electron-migration limited": {
        "SEI": "electron-migration limited",
    },
    "interstitial-diffusion limited": {
        "SEI": "interstitial-diffusion limited",
    },
    "EC reaction limited": {
        "SEI": "ec reaction limited",
    },
}

solutions = {}
solver = pybamm.IDAKLUSolver(atol=1e-9, rtol=1e-9)
# --------------------
# Run simulations
# --------------------
for label, options in sei_mechanisms.items():
    print(f"Running SPM with SEI mechanism: {label}")

    model = pybamm.lithium_ion.SPMe(options)

    sim = pybamm.Simulation(
        model,
        experiment=experiment,
        parameter_values=parameter_values,
        solver=solver,
    )

    # save_at_cycles=1 → store summary variables each cycle
    sol = sim.solve(save_at_cycles=1)
    solutions[label] = sol

# --------------------
# Extract Detailed Metrics (CC/CV/Discharge)
# --------------------
metrics = {}

for label, sol in solutions.items():
    print(f"Processing results for: {label}")
    m = {
        "cycles": [],
        "dis_time": [],
        "cc_cap": [],
        "cc_time": [],
        "cv_cap": [],
        "cv_time": [],
    }

    # Iterate through cycles to extract step data
    # Experiment: Discharge(0), Rest(1), CC(2), CV(3), Rest(4)
    for i, cycle in enumerate(sol.cycles):
        # Step 0: Discharge
        step_dis = cycle.steps[0]
        dis_time = step_dis["Time [h]"].entries[-1] - step_dis["Time [h]"].entries[0]

        # Step 2: CC Charge
        step_cc = cycle.steps[2]
        cc_cap = abs(
            step_cc["Discharge capacity [A.h]"].entries[-1]
            - step_cc["Discharge capacity [A.h]"].entries[0]
        )
        cc_time = step_cc["Time [h]"].entries[-1] - step_cc["Time [h]"].entries[0]

        # Step 3: CV Charge
        step_cv = cycle.steps[3]
        cv_cap = abs(
            step_cv["Discharge capacity [A.h]"].entries[-1]
            - step_cv["Discharge capacity [A.h]"].entries[0]
        )
        cv_time = step_cv["Time [h]"].entries[-1] - step_cv["Time [h]"].entries[0]

        m["cycles"].append(i + 1)
        m["dis_time"].append(dis_time)
        m["cc_cap"].append(cc_cap)
        m["cc_time"].append(cc_time)
        m["cv_cap"].append(cv_cap)
        m["cv_time"].append(cv_time)

    metrics[label] = m

# --------------------
# Plot Detailed Comparison
# --------------------
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle(f"SEI Mechanism Comparison (SPM, {N_cycles} Cycles)", fontsize=16)

# 1. CC Capacity
ax = axs[0, 0]
for label, m in metrics.items():
    ax.plot(m["cycles"], m["cc_cap"], label=label)
ax.set_ylabel("CC Capacity [A.h]")
ax.set_title("CC Charge Capacity")
ax.legend(fontsize="small")
ax.grid(True)

# 2. CC Time
ax = axs[0, 1]
for label, m in metrics.items():
    ax.plot(m["cycles"], m["cc_time"], label=label)
ax.set_ylabel("Time [h]")
ax.set_title("CC Charge Time")
ax.grid(True)

# 3. CV Capacity
ax = axs[1, 0]
for label, m in metrics.items():
    ax.plot(m["cycles"], m["cv_cap"], label=label)
ax.set_ylabel("CV Capacity [A.h]")
ax.set_title("CV Charge Capacity")
ax.grid(True)

# 4. CV Time
ax = axs[1, 1]
for label, m in metrics.items():
    ax.plot(m["cycles"], m["cv_time"], label=label)
ax.set_ylabel("Time [h]")
ax.set_title("CV Charge Time")
ax.grid(True)

# 5. Discharge Time
ax = axs[2, 0]
for label, m in metrics.items():
    ax.plot(m["cycles"], m["dis_time"], label=label)
ax.set_ylabel("Time [h]")
ax.set_title("Discharge Time")
ax.grid(True)

# 6. Empty or Summary (Optional)
axs[2, 1].axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("testfromchatgpt_detailed.png")
print("Saved testfromchatgpt_detailed.png")

print("\n" + "=" * 40)
print("SEI THICKNESS ANALYSIS")
print("=" * 40)
for label, sol in solutions.items():
    try:
        # Check standard variable for total SEI thickness
        if "X-averaged total SEI thickness [m]" in sol.all_models[0].variables:
            var_name = "X-averaged total SEI thickness [m]"
        elif "X-averaged negative SEI thickness [m]" in sol.all_models[0].variables:
            var_name = "X-averaged negative SEI thickness [m]"
        else:
            # Fallback or specific check
            var_name = "Negative SEI thickness [m]"

        sei_data = sol[var_name].entries
        init_val = sei_data[0]
        final_val = sei_data[-1]
        print(f"{label}:")
        print(f"  Initial: {init_val:.6e} m")
        print(f"  Final:   {final_val:.6e} m")
        print(f"  Growth:  {final_val - init_val:.6e} m")
    except Exception as e:
        print(f"{label}: Could not extract SEI thickness ({e})")
