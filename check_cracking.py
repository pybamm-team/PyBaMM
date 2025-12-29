import pybamm
import matplotlib.pyplot as plt
import numpy as np

# Setup similar to isolate_mechanism.py
parameter_values = pybamm.ParameterValues("OKane2022")
solver = pybamm.IDAKLUSolver()

# Baseline options (with cracking)
options = {
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "partially reversible",
    "lithium plating porosity change": "true",
    "particle mechanics": ("swelling and cracking", "swelling only"),
    "SEI on cracks": "true",
    "loss of active material": "stress-driven",
}

# Experiment: C/8 discharge, C/6 charge (same as isolate_mechanism.py)
experiment = pybamm.Experiment([
    (
        "Discharge at C/8 until 3.2 V",
        "Rest for 15 minutes",
        "Charge at C/6 until 4.1 V",
        "Hold at 4.1 V until C/37",
        "Rest for 15 minutes",
    )
] * 20) # Run 20 cycles

print("Running Baseline simulation to check cracking...")
model = pybamm.lithium_ion.SPMe(options)
sim = pybamm.Simulation(
    model, 
    experiment=experiment, 
    parameter_values=parameter_values, 
    solver=solver
)
sim.solve()

# Extract Cracking Variables
# OKane2022 typically has cracking on negative electrode
vars_to_check = [
    "X-averaged negative particle crack length [m]",
    "X-averaged negative particle surface area to volume ratio [m-1]",
    "X-averaged negative particle tangental stress [Pa]",
]

found_vars = {}
for var in vars_to_check:
    if var in sim.solution.all_models[0].variables:
        found_vars[var] = sim.solution[var].entries
    else:
        print(f"Variable not found: {var}")

# Print Analysis
print("\n--- Cracking Analysis (20 Cycles) ---")
for var, data in found_vars.items():
    init = data[0]
    final = data[-1]
    change = final - init
    print(f"{var}:")
    print(f"  Initial: {init:.4e}")
    print(f"  Final:   {final:.4e}")
    print(f"  Change:  {change:.4e}")

# Plot
if found_vars:
    t = sim.solution["Time [h]"].entries
    plt.figure(figsize=(10, 6))
    for var, data in found_vars.items():
        # Normalize to see relative change
        if np.max(np.abs(data)) > 0:
            plt.plot(t, data/data[0], label=var + " (normalized)")
        else:
             plt.plot(t, data, label=var)
    
    plt.xlabel("Time [h]")
    plt.ylabel("Relative Change")
    plt.title("Particle Mechanics Variables (Baseline @ C/8)")
    plt.legend()
    plt.grid(True)
    plt.savefig("check_cracking.png")
    print("Saved check_cracking.png")
