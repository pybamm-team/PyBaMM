#
# Constant-current constant-voltage charge
#
import plotly.graph_objects as go

import pybamm

pybamm.set_logging_level("NOTICE")

# Define a SINGLE cycle experiment
experiment_cycle = pybamm.Experiment(
    [
        (
            "Discharge at C/8 until 3.2 V",
            "Rest for 15 minutes",
            "Charge at C/6 until 4.1 V",
            "Hold at 4.1 V until C/30",
            "Rest for 15 minutes",
        ),
    ]
)

# model = pybamm.lithium_ion.DFN({"SEI": "ec reaction limited"})
model = pybamm.lithium_ion.DFN(
    {
        "SEI": "solvent-diffusion limited",
        "SEI porosity change": "true",
        "lithium plating": "partially reversible",
        "lithium plating porosity change": "true",
        "particle mechanics": ("swelling and cracking", "swelling only"),
        "SEI on cracks": "true",
        "loss of active material": "stress-driven",
    }
)
# parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values = pybamm.ParameterValues("OKane2022")
# parameter_values["Initial SEI thickness [m]"] = 1e-9
# parameter_values["SEI kinetic rate constant [m.s-1]"] = 1e-11

# Print relevant parameters for "ec reaction limited" model
sei_parameters = {k: v for k, v in parameter_values.items() if "SEI" in k or "sei" in k}
print("SEI Parameters:")
for k, v in sei_parameters.items():
    print(f"  {k}: {v}")

# Use IDAKLUSolver for better convergence and smoothness
solver = pybamm.IDAKLUSolver(atol=1e-9, rtol=1e-9)
submesh_types = model.default_submesh_types.copy()
submesh_types["negative particle"] = pybamm.MeshGenerator(
    pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}
)
submesh_types["positive particle"] = pybamm.MeshGenerator(
    pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}
)

# Reduced mesh points to 20 as per previous optimization (could likely increase back to 30-40 with this new approach if desired)
var_pts = {"x_n": 10, "x_s": 10, "x_p": 10, "r_n": 20, "r_p": 20}

# Create simulation object ONCE
sim = pybamm.Simulation(
    model,
    experiment=experiment_cycle,
    parameter_values=parameter_values,
    solver=solver,
    var_pts=var_pts,
    submesh_types=submesh_types,
)

# --- Cycle Loop ---
TOTAL_CYCLES = 100
discharge_capacities = []
cc_charge_capacities = []
cv_charge_capacities = []
cycle_numbers = []

print(f"Starting simulation for {TOTAL_CYCLES} cycles (Cycle-by-Cycle Mode)...")

starting_sol = None

for i in range(TOTAL_CYCLES):
    print(f"--- Running Cycle {i + 1}/{TOTAL_CYCLES} ---")

    # Solve for ONE cycle
    if i == 0:
        sol = sim.solve(initial_soc=0.3)
    else:
        sol = sim.solve(starting_solution=starting_sol)

    # --- Extract Data ---
    # Step 1: Discharge
    step_discharge = sol.steps[0]
    d_cap = (
        step_discharge["Discharge capacity [A.h]"].entries[-1]
        - step_discharge["Discharge capacity [A.h]"].entries[0]
    )

    # Step 3: Charge CC
    step_charge_cc = sol.steps[2]
    c_cap_cc = (
        step_charge_cc["Discharge capacity [A.h]"].entries[-1]
        - step_charge_cc["Discharge capacity [A.h]"].entries[0]
    )

    # Step 4: Charge CV
    step_charge_cv = sol.steps[3]
    c_cap_cv = (
        step_charge_cv["Discharge capacity [A.h]"].entries[-1]
        - step_charge_cv["Discharge capacity [A.h]"].entries[0]
    )

    discharge_capacities.append(abs(d_cap))
    cc_charge_capacities.append(abs(c_cap_cc))
    cv_charge_capacities.append(abs(c_cap_cv))
    cycle_numbers.append(i + 1)

    # --- Prepare for next cycle ---
    # We only keep the last state to start the next cycle
    starting_sol = sol.last_state

    # Explicitly clear the full solution from the simulation object to free memory
    # (Though re-assigning starting_sol = sol.last_state effectively drops the ref to the big 'sol' object once this loop iteration ends)


# --- Plotting ---
print("\nSimulation complete. Generating plots...")

# Create interactive Plotly figure
fig_plotly = go.Figure()
fig_plotly.add_trace(
    go.Scatter(
        x=cycle_numbers,
        y=discharge_capacities,
        mode="lines+markers",
        name="Discharge Capacity",
    )
)
fig_plotly.add_trace(
    go.Scatter(
        x=cycle_numbers,
        y=cc_charge_capacities,
        mode="lines+markers",
        name="CC Charge Capacity",
    )
)
fig_plotly.add_trace(
    go.Scatter(
        x=cycle_numbers,
        y=cv_charge_capacities,
        mode="lines+markers",
        name="CV Charge Capacity",
    )
)

fig_plotly.update_layout(
    title="Capacity per Cycle",
    xaxis_title="Cycle Number",
    yaxis_title="Capacity [A.h]",
    hovermode="x unified",
)
fig_plotly.write_html("cccv_capacity.html")
print("Plotly figure saved to cccv_capacity.html")

# Print capacity data summary
print("\nCapacity Data per Cycle (First 5 and Last 5):")
print(
    f"{'Cycle':<6} | {'Discharge [A.h]':<16} | {'CC Charge [A.h]':<16} | {'CV Charge [A.h]':<16}"
)
print("-" * 60)
for i in range(len(discharge_capacities)):
    if i < 5 or i >= len(discharge_capacities) - 5:
        print(
            f"{cycle_numbers[i]:<6} | {discharge_capacities[i]:<16.4f} | {cc_charge_capacities[i]:<16.4f} | {cv_charge_capacities[i]:<16.4f}"
        )
    elif i == 5:
        print("...")
