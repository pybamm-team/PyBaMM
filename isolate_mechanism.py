
import pybamm
import matplotlib.pyplot as plt
import pandas as pd
import gc
import tqdm
# Setup
total_cycles = 200
cycles_per_chunk = 10
num_chunks = total_cycles // cycles_per_chunk

parameter_values = pybamm.ParameterValues("OKane2022")
solver = pybamm.IDAKLUSolver(atol=1e-7, rtol=1e-7)

# Define Variants
variants = {
    "Baseline": {
        "SEI": "solvent-diffusion limited",
        "SEI porosity change": "true",
        "lithium plating": "partially reversible",
        "lithium plating porosity change": "true",
        "particle mechanics": ("swelling and cracking", "swelling only"),
        "SEI on cracks": "true",
        "loss of active material": "stress-driven",
    },
    "No LAM": {
        "SEI": "solvent-diffusion limited",
        "SEI porosity change": "true",
        "lithium plating": "partially reversible",
        "lithium plating porosity change": "true",
        "particle mechanics": ("swelling and cracking", "swelling only"),
        "SEI on cracks": "true",
        "loss of active material": "none",
    },
    "No Plating": {
        "SEI": "solvent-diffusion limited",
        "SEI porosity change": "true",
        "lithium plating": "none",
        "lithium plating porosity change": "false",
        "particle mechanics": ("swelling and cracking", "swelling only"),
        "SEI on cracks": "true",
        "loss of active material": "none",
    },
    "No Cracking": {
        "SEI": "solvent-diffusion limited",
        "SEI porosity change": "true",
        "lithium plating": "partially reversible",
        "lithium plating porosity change": "true",
        "particle mechanics": "none",
        "SEI on cracks": "false",
        "loss of active material": "none",
    },
    "Only SEI": {
        "SEI": "solvent-diffusion limited",
        "SEI porosity change": "true",
        "lithium plating": "none",
        "lithium plating porosity change": "false",
        "particle mechanics": "none",
        "SEI on cracks": "false",
        "loss of active material": "none",
    },
    "No SEI": {
        "SEI": "none",
        "SEI porosity change": "false",
        "lithium plating": "none",
        "lithium plating porosity change": "false",
        "particle mechanics": "none",
        "SEI on cracks": "false",
        "loss of active material": "none",
    }
}

experiment_step = (
    "Discharge at C/8 until 3.2 V",
    "Rest for 15 minutes",
    "Charge at C/6 until 4.1 V",
    "Hold at 4.1 V until C/37",
    "Rest for 15 minutes",
)
experiment_chunk = pybamm.Experiment([experiment_step] * cycles_per_chunk)

def run_chunked_simulation(name, options):
    print(f"\n--- Starting {name} ({total_cycles} cycles) ---")
    
    all_cc_caps = []
    all_cc_times = []
    all_cv_caps = []
    all_cv_times = []
    all_cycles = []
    all_sei_thickness = []
    all_crack_lengths = []
    all_rest_data = []
    
    starting_solution = None
    
    # Mesh settings (globally defined for consistency)
    var_pts = {"x_n": 10, "x_s": 10, "x_p": 10, "r_n": 20, "r_p": 20}
    
    for chunk_idx in range(num_chunks):
        start_cycle = chunk_idx * cycles_per_chunk + 1
        print(f"  {name}: Chunk {chunk_idx + 1}/{num_chunks}...")
        
        # Re-init model
        # Switch to SPMe for faster execution (valid at 1C)
        model = pybamm.lithium_ion.SPMe(options)
        submesh_types = model.default_submesh_types.copy()
        submesh_types["negative particle"] = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params={"side": "right"})
        submesh_types["positive particle"] = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params={"side": "right"})
        
        sim = pybamm.Simulation(
            model,
            experiment=experiment_chunk,
            parameter_values=parameter_values,
            solver=solver,
            var_pts=var_pts,
            submesh_types=submesh_types,
        )
        
        if starting_solution:
            sim.model.set_initial_conditions_from(starting_solution)
            
        try:
            if chunk_idx == 0:
                sim.solve(initial_soc=0.3)
            else:
                sim.solve()
                
            # Process results for this chunk
            # Extract SEI variable name dynamically once per chunk
            sei_var = None
            if "X-averaged total SEI thickness [m]" in sim.solution.all_models[0].variables:
                sei_var = "X-averaged total SEI thickness [m]"
            elif "X-averaged negative SEI thickness [m]" in sim.solution.all_models[0].variables:
                sei_var = "X-averaged negative SEI thickness [m]"
            
            for i, sol in enumerate(sim.solution.cycles):
                current_cycle_num = start_cycle + i
                
                # CC Charge is Step 2 (index 2)
                step_cc = sol.steps[2]
                cc_cap = abs(step_cc["Discharge capacity [A.h]"].entries[-1] - step_cc["Discharge capacity [A.h]"].entries[0])
                cc_time = step_cc["Time [h]"].entries[-1] - step_cc["Time [h]"].entries[0]
                
                # CV Charge is Step 3 (index 3)
                step_cv = sol.steps[3]
                cv_cap = abs(step_cv["Discharge capacity [A.h]"].entries[-1] - step_cv["Discharge capacity [A.h]"].entries[0])
                cv_time = step_cv["Time [h]"].entries[-1] - step_cv["Time [h]"].entries[0]

                # SEI Thickness (End of cycle)
                if sei_var:
                    sei_val = sol[sei_var].entries[-1]
                else:
                    sei_val = 0.0

                # Crack Length (End of cycle)
                # Check for crack variable presence
                crack_val = 0.0
                if "X-averaged negative particle crack length [m]" in sol.all_models[0].variables:
                     crack_val = sol["X-averaged negative particle crack length [m]"].entries[-1]

                all_cc_caps.append(cc_cap)
                all_cc_times.append(cc_time)
                all_cv_caps.append(cv_cap)
                all_cv_times.append(cv_time)
                all_cycles.append(current_cycle_num)
                all_sei_thickness.append(sei_val)
                all_crack_lengths.append(crack_val)
            
            starting_solution = sim.solution

            # Check for rest insertion (after 50, 100, 150 cycles)
            # cycles_per_chunk is 10, so chunk_idx 4 is cycles 41-50.
            if (chunk_idx + 1) % 5 == 0 and (chunk_idx + 1) < num_chunks:
                cycle_num = (chunk_idx + 1) * cycles_per_chunk
                print(f"  Inserting 4-day rest after cycle {cycle_num}...")
                
                # SEI before rest (end of current chunk)
                sei_before = 0.0
                if sei_var:
                     sei_before = starting_solution[sei_var].entries[-1]
                
                # Re-init model for rest
                rest_model = pybamm.lithium_ion.SPMe(options)
                rest_submesh_types = rest_model.default_submesh_types.copy()
                rest_submesh_types["negative particle"] = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params={"side": "right"})
                rest_submesh_types["positive particle"] = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params={"side": "right"})
                
                rest_sim = pybamm.Simulation(
                    rest_model,
                    experiment=pybamm.Experiment(["Rest for 96 hours"]),
                    parameter_values=parameter_values,
                    solver=solver,
                    var_pts=var_pts,
                    submesh_types=rest_submesh_types,
                )
                
                rest_sim.model.set_initial_conditions_from(starting_solution)
                rest_sim.solve()
                starting_solution = rest_sim.solution
                
                # SEI after rest
                sei_after = 0.0
                if sei_var and sei_var in rest_sim.solution.all_models[0].variables:
                     sei_after = rest_sim.solution[sei_var].entries[-1]
                
                all_rest_data.append({
                    "cycle": cycle_num,
                    "before": sei_before,
                    "after": sei_after
                })

                del rest_sim
                gc.collect()

            
        except Exception as e:
            print(f"  FAILED at chunk {chunk_idx + 1}: {e}")
            break
            
        # Clean up
        del sim
        gc.collect()
        
        
    return all_cycles, all_cc_caps, all_cc_times, all_cv_caps, all_cv_times, all_sei_thickness, all_crack_lengths, all_rest_data

results = {}

for name, options in tqdm.tqdm(variants.items()):
    cycles, cc_caps, cc_times, cv_caps, cv_times, sei_thickness, crack_lengths, rest_data = run_chunked_simulation(name, options)
    results[name] = {
        "cycles": cycles,
        "cc_caps": cc_caps,
        "cc_times": cc_times,
        "cv_caps": cv_caps,
        "cv_times": cv_times,
        "sei_thickness": sei_thickness,
        "crack_lengths": crack_lengths,
        "rest_data": rest_data
    }

# --------------------
# SEI Growth Analysis
# --------------------
print("\n" + "="*40)
print("SEI GROWTH RATE FROM CRACKING ANALYSIS")
print("="*40)

# Create DataFrame for export
export_data = []

# Update Plot to 3x2 (6 subplots) to include SEI Growth Rate
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

for name, data in results.items():
    if data["sei_thickness"]:
        cycles = data["cycles"]
        thickness = data["sei_thickness"]
        cracks = data["crack_lengths"]
        
        # Calculate Rate (dSEI/dCycle)
        rates = [0.0] * len(thickness)
        for i in range(1, len(thickness)):
            rates[i] = thickness[i] - thickness[i-1]
        
        data["sei_rates"] = rates
        
        # Add to export
        for c, t, r, cr in zip(cycles, thickness, rates, cracks):
            period = "Cycling"
            if c in [51, 101, 151]:
                 period = "Post-Rest (96hr)"
            
            export_data.append({
                "Variant": name,
                "Cycle": c,
                "SEI Thickness [m]": t,
                "Growth Rate [m/cycle]": r,
                "Crack Length [m]": cr,
                "Period": period
            })
            
        # Plot Thickness
        axs[2, 0].plot(cycles, thickness, marker='.', label=name)
        
        # Plot Rate
        axs[2, 1].plot(cycles, rates, marker='.', label=name)
    else:
        axs[2, 0].text(0.5, 0.5, f"{name}: No SEI Data", ha='center')
        axs[2, 1].text(0.5, 0.5, f"{name}: No SEI Data", ha='center')

# Save CSV
df = pd.DataFrame(export_data)
csv_filename = "sei_growth_analysis.csv"
df.to_csv(csv_filename, index=False)
print(f"Detailed SEI data saved to {csv_filename}")


# Finalize Plots
# 1. CC Capacity
for name, data in results.items():
    axs[0, 0].plot(data["cycles"], data["cc_caps"], marker='.', label=name)
axs[0, 0].set_ylabel("CC Capacity [A.h]")
axs[0, 0].set_title("CC Charge Capacity")
axs[0, 0].grid(True)
axs[0, 0].legend(fontsize='small')

# 2. CC Time
for name, data in results.items():
    axs[0, 1].plot(data["cycles"], data["cc_times"], marker='.', label=name)
axs[0, 1].set_ylabel("CC Time [h]")
axs[0, 1].set_title("CC Charge Time")
axs[0, 1].grid(True)

# 3. CV Capacity
for name, data in results.items():
    axs[1, 0].plot(data["cycles"], data["cv_caps"], marker='.', label=name)
axs[1, 0].set_ylabel("CV Capacity [A.h]")
axs[1, 0].set_title("CV Charge Capacity")
axs[1, 0].grid(True)

# 4. CV Time
for name, data in results.items():
    axs[1, 1].plot(data["cycles"], data["cv_times"], marker='.', label=name)
axs[1, 1].set_ylabel("CV Time [h]")
axs[1, 1].set_title("CV Charge Time")
axs[1, 1].grid(True)

# 5. SEI Thickness
axs[2, 0].set_ylabel("SEI Thickness [m]")
axs[2, 0].set_title("Total SEI Thickness")
axs[2, 0].grid(True)

# 6. SEI Growth Rate
axs[2, 1].set_ylabel("Growth Rate [m/cycle]")
axs[2, 1].set_title("SEI Growth Rate")
axs[2, 1].grid(True)
axs[2, 1].set_yscale('log') # Log scale might be better for rates

plt.tight_layout()
plt.savefig("mechanism_isolation_detailed.png")
print("\nPlot saved to mechanism_isolation_detailed.png")

# Print Summary Table (Initial, Final, Average Rate)
print(f"\n{'Variant':<20} | {'Initial [m]':<12} | {'Final [m]':<12} | {'Avg Rate [m/cyc]':<15}")
print("-" * 70)
for name, data in results.items():
    if data["sei_thickness"]:
        init = data["sei_thickness"][0]
        final = data["sei_thickness"][-1]
        avg_rate = (final - init) / len(data["sei_thickness"])
        print(f"{name:<20} | {init:.4e}   | {final:.4e}   | {avg_rate:.4e}")

print("\n" + "="*40)
print("REST STEP ANALYSIS (Before vs After 4 Days)")
print("="*40)
print(f"{'Variant':<20} | {'Cycle':<5} | {'Before Rest [m]':<15} | {'After Rest [m]':<15} | {'Change [m]':<15}")
print("-" * 80)

for name, data in results.items():
    if "rest_data" in data and data["rest_data"]:
        for r in data["rest_data"]:
            change = r["after"] - r["before"]
            print(f"{name:<20} | {r['cycle']:<5} | {r['before']:.4e}        | {r['after']:.4e}        | {change:.4e}")
    else:
         print(f"{name:<20} | N/A   | N/A             | N/A             | N/A")



