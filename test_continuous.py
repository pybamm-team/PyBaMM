
import pybamm
import pandas as pd
import os

# Setup simulation for 40 cycles CONTINUOUS
total_cycles = 40

experiment_step = (
    "Discharge at C/8 until 3.2 V",
    "Rest for 15 minutes",
    "Charge at C/6 until 4.1 V",
    "Hold at 4.1 V until C/37",
    "Rest for 15 minutes",
)
experiment = pybamm.Experiment([experiment_step] * total_cycles)

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
parameter_values = pybamm.ParameterValues("OKane2022")

solver = pybamm.IDAKLUSolver(atol=1e-9, rtol=1e-9)
submesh_types = model.default_submesh_types.copy()
submesh_types["negative particle"] = pybamm.MeshGenerator(
    pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}
)
submesh_types["positive particle"] = pybamm.MeshGenerator(
    pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}
)

var_pts = {"x_n": 10, "x_s": 10, "x_p": 10, "r_n": 40, "r_p": 40}

print("Running 40 cycles continuous simulation...")
sim = pybamm.Simulation(
    model,
    experiment=experiment,
    parameter_values=parameter_values,
    solver=solver,
    var_pts=var_pts, 
    submesh_types=submesh_types,
)
sim.solve(initial_soc=0.3)

print("\nResults (Cycle 15-30 focus):")
for i, sol in enumerate(sim.solution.cycles):
    cycle_num = i + 1
    if 10 <= cycle_num <= 30:
        step_charge_cv = sol.steps[3]
        c_cap_cv = abs(step_charge_cv["Discharge capacity [A.h]"].entries[-1] - step_charge_cv["Discharge capacity [A.h]"].entries[0])
        print(f"Cycle {cycle_num}: {c_cap_cv:.6f}")
