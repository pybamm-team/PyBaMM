#
# Constant-current constant-voltage charge
#
import matplotlib.pyplot as plt

import pybamm
import pandas as pd
import os
import plotly.graph_objects as go

pybamm.set_logging_level("NOTICE")

# =============================================================================
# CHUNKING SIMULATION SETUP
# =============================================================================

total_cycles = 200
cycles_per_chunk = 20
num_chunks = total_cycles // cycles_per_chunk

# Define experiment for a SINGLE chunk
experiment_step = (
    "Discharge at C/8 until 3.2 V",
    "Rest for 15 minutes",
    "Charge at C/6 until 4.1 V",
    "Hold at 4.1 V until C/37",
    "Rest for 15 minutes",
)
experiment_chunk = pybamm.Experiment([experiment_step] * cycles_per_chunk)

# Output files
capacity_file = "capacity_results.csv"
aging_file = "aging_results.csv"

# Initialize CSV files
# Capacity CSV
if os.path.exists(capacity_file):
    os.remove(capacity_file)
with open(capacity_file, "w") as f:
    f.write("Cycle,Discharge Capacity [A.h],CC Charge Capacity [A.h],CV Charge Capacity [A.h]\n")

# Aging CSV
aging_vars = [
    "X-averaged total SEI thickness [m]",
    "X-averaged negative electrode porosity",
    "X-averaged positive electrode porosity",
    "X-averaged lithium plating thickness [m]",
    "X-averaged negative electrode active material volume fraction",
    "X-averaged positive electrode active material volume fraction",
    "Loss of lithium to SEI [mol]",
    "Loss of lithium to lithium plating [mol]",
    "X-averaged negative particle surface concentration [mol.m-3]",
]

if os.path.exists(aging_file):
    os.remove(aging_file)
with open(aging_file, "w") as f:
    header = "Chunk_Start_Cycle,Chunk_End_Cycle,Parameter,Initial,Final,Change\n"
    f.write(header)

starting_solution = None

print(f"Starting simulation: {total_cycles} cycles in {num_chunks} chunks of {cycles_per_chunk}.")

# Initialize Plotly figure
fig_plotly = go.Figure()
all_discharge_caps = []
all_cc_charge_caps = []
all_cv_charge_caps = []
all_cycle_nums = []

# parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values = pybamm.ParameterValues("OKane2022")
# parameter_values["Initial SEI thickness [m]"] = 1e-9
# parameter_values["SEI kinetic rate constant [m.s-1]"] = 1e-11

# Print relevant parameters for "ec reaction limited" model
sei_parameters = {
    k: v for k, v in parameter_values.items() if "SEI" in k or "sei" in k
}
print("SEI Parameters:")
for k, v in sei_parameters.items():
    print(f"  {k}: {v}")

# Submesh types and var_pts are defined once as they are constant
submesh_types = pybamm.lithium_ion.DFN().default_submesh_types.copy() # Initialize from a dummy model
submesh_types["negative particle"] = pybamm.MeshGenerator(
    pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}
)
submesh_types["positive particle"] = pybamm.MeshGenerator(
    pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}
)

var_pts = {"x_n": 10, "x_s": 10, "x_p": 10, "r_n": 40, "r_p": 40}


for chunk_idx in range(num_chunks):
    start_cycle = chunk_idx * cycles_per_chunk + 1
    end_cycle = (chunk_idx + 1) * cycles_per_chunk
    print(f"\n--- Running Chunk {chunk_idx + 1}/{num_chunks} (Cycles {start_cycle}-{end_cycle}) ---")

    # Re-initialize model to ensure clean state, but we might need to be careful 
    # if model has internal state that isn't captured by set_initial_conditions_from.
    # Usually recreating the model object is safer.
    
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

    # Re-apply solver and mesh settings
    solver = pybamm.IDAKLUSolver()
    # var_pts is defined above globally

    sim = pybamm.Simulation(
        model,
        experiment=experiment_chunk,
        parameter_values=parameter_values,
        solver=solver,
        var_pts=var_pts, 
        submesh_types=submesh_types,
    )

    # Set initial conditions from previous chunk if applicable
    if starting_solution is not None:
        print("Setting initial conditions from previous chunk...")
        sim.model.set_initial_conditions_from(starting_solution)
    
    # Solve
    # Only use initial_soc for the VERY FIRST chunk
    if chunk_idx == 0:
        sim.solve(initial_soc=0.3)
    else:
        # For subsequent chunks, the initial conditions are set manually, so we don't pass initial_soc
        sim.solve()

    # Process Results for this Chunk
    sol = sim.solution
    
    # 1. Capacity Data
    cycle_offset = start_cycle - 1
    chunk_discharge_caps = []
    chunk_cc_charge_caps = []
    chunk_cv_charge_caps = []
    chunk_cycle_nums = []

    with open(capacity_file, "a") as f:
        for i, cycle_sol in enumerate(sol.cycles):
            # Calculate capacities
            step_discharge = cycle_sol.steps[0]
            d_cap = abs(step_discharge["Discharge capacity [A.h]"].entries[-1] - step_discharge["Discharge capacity [A.h]"].entries[0])
            
            step_charge_cc = cycle_sol.steps[2]
            c_cap_cc = abs(step_charge_cc["Discharge capacity [A.h]"].entries[-1] - step_charge_cc["Discharge capacity [A.h]"].entries[0])
            
            step_charge_cv = cycle_sol.steps[3]
            c_cap_cv = abs(step_charge_cv["Discharge capacity [A.h]"].entries[-1] - step_charge_cv["Discharge capacity [A.h]"].entries[0])

            current_cycle_num = cycle_offset + i + 1
            f.write(f"{current_cycle_num},{d_cap:.6f},{c_cap_cc:.6f},{c_cap_cv:.6f}\n")
            
            # Store for plotting later
            chunk_discharge_caps.append(d_cap)
            chunk_cc_charge_caps.append(c_cap_cc)
            chunk_cv_charge_caps.append(c_cap_cv)
            chunk_cycle_nums.append(current_cycle_num)

    all_discharge_caps.extend(chunk_discharge_caps)
    all_cc_charge_caps.extend(chunk_cc_charge_caps)
    all_cv_charge_caps.extend(chunk_cv_charge_caps)
    all_cycle_nums.extend(chunk_cycle_nums)

    # 2. Aging Data
    with open(aging_file, "a") as f:
        for var in aging_vars:
            try:
                if var in sol.all_models[0].variables:
                    data = sol[var].entries
                    init_val = data[0]
                    final_val = data[-1]
                    change = final_val - init_val
                    f.write(f"{start_cycle},{end_cycle},{var},{init_val:.6e},{final_val:.6e},{change:.6e}\n")
                else:
                    f.write(f"{start_cycle},{end_cycle},{var},N/A,N/A,N/A\n")
            except Exception as e:
                f.write(f"{start_cycle},{end_cycle},{var},Error,Error,{str(e)}\n")

    # Update starting solution for next chunk
    starting_solution = sol
    
    # Save Plotly figure incrementally (rewrite file)
    fig_plotly = go.Figure()
    fig_plotly.add_trace(go.Scatter(x=all_cycle_nums, y=all_discharge_caps, mode='lines+markers', name='Discharge Capacity'))
    fig_plotly.add_trace(go.Scatter(x=all_cycle_nums, y=all_cc_charge_caps, mode='lines+markers', name='CC Charge Capacity'))
    fig_plotly.add_trace(go.Scatter(x=all_cycle_nums, y=all_cv_charge_caps, mode='lines+markers', name='CV Charge Capacity'))
    fig_plotly.update_layout(title="Capacity per Cycle", xaxis_title="Cycle Number", yaxis_title="Capacity [A.h]", hovermode="x unified")
    fig_plotly.write_html("cccv_capacity.html")

    print(f"Chunk {chunk_idx + 1} completed. Data saved.")

print("\nSimulation completed successfully.")


# Plot voltages and OCVs from the discharge segments only
# This part will now only plot the last chunk's solution
fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# Extract sub solutions from the last chunk's solution
# Note: sol is the solution from the last chunk
for i in range(len(sol.cycles)):
    cycle_sol = sol.cycles[i]
    
    # Extract variables for plots
    t = cycle_sol["Time [h]"].entries
    V = cycle_sol["Voltage [V]"].entries
    ocv_p = cycle_sol["X-averaged positive electrode open-circuit potential [V]"].entries
    ocv_n = cycle_sol["X-averaged negative electrode open-circuit potential [V]"].entries

    # Plot Cell Voltage
    axs[0].plot(t - t[0], V, label=f"Cycle {start_cycle + i}")
    axs[0].set_ylabel("Voltage [V]")
    
    # Plot Cathode OCV
    axs[1].plot(t - t[0], ocv_p, label=f"Cycle {start_cycle + i}")
    axs[1].set_ylabel("Cathode OCV [V]")

    # Plot Anode OCV
    axs[2].plot(t - t[0], ocv_n, label=f"Cycle {start_cycle + i}")
    axs[2].set_ylabel("Anode OCV [V]")
    axs[2].set_xlabel("Time [h]")
    axs[2].set_xlim([0, t[-1] - t[0]])

axs[0].legend(loc="lower left")
plt.tight_layout()
fig.savefig("cccv_results.png")

# Plot Capacity per Cycle (using accumulated data)
plt.figure(figsize=(6, 4))
plt.plot(all_cycle_nums, all_discharge_caps, 'o-', label="Discharge Capacity")
plt.plot(all_cycle_nums, all_cc_charge_caps, 's-', label="CC Charge Capacity")
plt.plot(all_cycle_nums, all_cv_charge_caps, '^-', label="CV Charge Capacity")
plt.xlabel("Cycle Number")
plt.ylabel("Capacity [A.h]")
plt.title("Capacity per Cycle")
plt.legend()
plt.grid(True)

# Print capacity data for each cycle (from accumulated data)
print("\nCapacity Data per Cycle:")
print(f"{'Cycle':<6} | {'Discharge [A.h]':<16} | {'CC Charge [A.h]':<16} | {'CV Charge [A.h]':<16}")
print("-" * 60)
for i in range(len(all_discharge_caps)):
    print(f"{all_cycle_nums[i]:<6} | {all_discharge_caps[i]:<16.4f} | {all_cc_charge_caps[i]:<16.4f} | {all_cv_charge_caps[i]:<16.4f}")

plt.savefig("cccv_capacity.png")

# The Plotly figure is already saved incrementally within the loop.
print("Plotly figure saved to cccv_capacity.html")

# Save time, voltage, current, discharge capacity, temperature, and electrolyte
# concentration to csv and matlab formats
# These will save data from the *last* chunk's solution (sol)
sol.save_data(
    "output.mat",
    [
        "Time [h]",
        "Current [A]",
        "Voltage [V]",
        "Discharge capacity [A.h]",
        "X-averaged cell temperature [K]",
        "Electrolyte concentration [mol.m-3]",
    ],
    to_format="matlab",
    short_names={
        "Time [h]": "t",
        "Current [A]": "I",
        "Voltage [V]": "V",
        "Discharge capacity [A.h]": "Q",
        "X-averaged cell temperature [K]": "T",
        "Electrolyte concentration [mol.m-3]": "c_e",
    },
)
# We can only save 0D variables to csv
sol.save_data(
    "output.csv",
    [
        "Time [h]",
        "Current [A]",
        "Voltage [V]",
        "Discharge capacity [A.h]",
        "X-averaged cell temperature [K]",
        "X-averaged positive electrode open-circuit potential [V]",
        "X-averaged negative electrode open-circuit potential [V]",
    ],
    to_format="csv",
)

# Show all plots
# sim.plot() # This would plot the last chunk's solution

# Print Initial and Final values of Aging Parameters
# This is now handled by the aging_results.csv file
print("\n" + "=" * 80)
print("AGING PARAMETERS TRACKING - See aging_results.csv for full data")
print("=" * 80 + "\n")

