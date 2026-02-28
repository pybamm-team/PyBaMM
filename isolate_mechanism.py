import gc
import os
# Force CPU to avoid Metal/JAX instability
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import matplotlib.pyplot as plt
import pandas as pd
import tqdm

import pybamm

# --- Configuration ---
TOTAL_CYCLES = 200
CYCLES_PER_CHUNK = 10
NUM_CHUNKS = TOTAL_CYCLES // CYCLES_PER_CHUNK
CSV_FILENAME = "sei_growth_analysis.csv"

# --- Variants Definition ---
VARIANTS = {
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
    "Electron Migration SEI": {
        "SEI": "electron-migration limited",
        "SEI porosity change": "true",
        "lithium plating": "none",
        "lithium plating porosity change": "false",
        "particle mechanics": "none",
        "SEI on cracks": "false",
        "loss of active material": "none",
    },
}

# --- Experiment Definition ---
EXPERIMENT_STEP = (
    "Discharge at C/9 until 3.2 V",
    "Rest for 15 minutes",
    "Charge at C/7 until 4.1 V",
    "Hold at 4.1 V until C/37",
    "Rest for 15 minutes",
    "Discharge at C/4 for 5s",
    "Rest for 15 minutes",
)
EXPERIMENT_CHUNK = pybamm.Experiment([EXPERIMENT_STEP] * CYCLES_PER_CHUNK)

# --- Helper Functions ---


def get_submesh_types(model_class):
    """Returns the submesh types (Uniform default)."""
    model = model_class()  # Instantiate to get default submesh
    submesh_types = model.default_submesh_types.copy()
    # Using Uniform Mesh to avoid solver instability with Exponential mesh
    # submesh_types["negative particle"] = pybamm.MeshGenerator(
    #     pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}
    # )
    # submesh_types["positive particle"] = pybamm.MeshGenerator(
    #     pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}
    # )
    return submesh_types


def run_rest_step(
    cycle_num, starting_solution, options, parameter_values, solver, var_pts
):
    """Runs a 4-day rest step and returns trace data and SEI growth."""
    print(f"  Inserting 4-day rest after cycle {cycle_num}...")

    # Identify SEI variable
    sei_var = None
    if (
        "X-averaged total SEI thickness [m]"
        in starting_solution.all_models[0].variables
    ):
        sei_var = "X-averaged total SEI thickness [m]"
    elif (
        "X-averaged negative SEI thickness [m]"
        in starting_solution.all_models[0].variables
    ):
        sei_var = "X-averaged negative SEI thickness [m]"

    sei_before = starting_solution[sei_var].entries[-1] if sei_var else 0.0

    # Run Rest Simulation
    rest_model = pybamm.lithium_ion.DFN(options)
    rest_submesh = get_submesh_types(pybamm.lithium_ion.DFN)

    rest_sim = pybamm.Simulation(
        rest_model,
        experiment=pybamm.Experiment(["Rest for 96 hours"]),
        parameter_values=parameter_values,
        solver=solver,
        var_pts=var_pts,
        submesh_types=rest_submesh,
    )

    rest_sim.model.set_initial_conditions_from(starting_solution)
    rest_sim.solve()

    # Post-process
    sei_after = (
        rest_sim.solution[sei_var].entries[-1]
        if (sei_var and sei_var in rest_sim.solution.all_models[0].variables)
        else 0.0
    )

    t_rest = rest_sim.solution["Time [h]"].entries
    t_rest = t_rest - t_rest[0]  # Normalize time
    v_rest = rest_sim.solution["Terminal voltage [V]"].entries
    
    # Extract Cathode Spatial Distribution (at start of rest)
    c_s_surf_p = rest_sim.solution["Positive particle surface concentration [mol.m-3]"].entries[:, 0]
    x_p = rest_sim.solution["x_p [m]"].entries[:, 0]

    rest_data = {
        "cycle": cycle_num,
        "before": sei_before,
        "after": sei_after,
        "voltage_trace": {"cycle": cycle_num, "time": t_rest, "voltage": v_rest},
        "spatial_dist": {"x_p": x_p, "c_s_surf_p": c_s_surf_p},
    }

    final_sol = rest_sim.solution
    del rest_sim
    gc.collect()

    return final_sol, rest_data


def run_chunked_simulation(name, options):
    print(f"\n--- Starting {name} ({TOTAL_CYCLES} cycles) ---")

    # Accumulators
    data = {
        "cycles": [],
        "cc_caps": [],
        "cc_times": [],
        "cv_caps": [],
        "cv_times": [],
        "dis_caps": [],
        "sei_thickness": [],
        "crack_lengths": [],
        "rest_data": [],
    }

    starting_solution = None
    # Optimized mesh configuration identified for stability
    var_pts = {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 50, "r_p": 50}
    parameter_values = pybamm.ParameterValues("OKane2022")
    solver = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8)

    for chunk_idx in range(NUM_CHUNKS):
        start_cycle = chunk_idx * CYCLES_PER_CHUNK + 1
        print(f"  {name}: Chunk {chunk_idx + 1}/{NUM_CHUNKS}...")

        # Setup Simulation
        model = pybamm.lithium_ion.DFN(options)
        submesh_types = get_submesh_types(pybamm.lithium_ion.DFN)

        sim = pybamm.Simulation(
            model,
            experiment=EXPERIMENT_CHUNK,
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

            # --- Result Extraction ---
            sol = sim.solution
            # Determine SEI variable name
            sei_var = None
            if "X-averaged total SEI thickness [m]" in sol.all_models[0].variables:
                sei_var = "X-averaged total SEI thickness [m]"
            elif "X-averaged negative SEI thickness [m]" in sol.all_models[0].variables:
                sei_var = "X-averaged negative SEI thickness [m]"

            for i, cycle_sol in enumerate(sol.cycles):
                current_cycle_num = start_cycle + i

                # Step 0: Discharge
                step_dis = cycle_sol.steps[0]
                dis_cap = abs(
                    step_dis["Discharge capacity [A.h]"].entries[-1]
                    - step_dis["Discharge capacity [A.h]"].entries[0]
                )

                # Step 2: CC Charge
                step_cc = cycle_sol.steps[2]
                cc_cap = abs(
                    step_cc["Discharge capacity [A.h]"].entries[-1]
                    - step_cc["Discharge capacity [A.h]"].entries[0]
                )
                cc_time = (
                    step_cc["Time [h]"].entries[-1] - step_cc["Time [h]"].entries[0]
                )

                # Step 3: CV Charge
                step_cv = cycle_sol.steps[3]
                cv_cap = abs(
                    step_cv["Discharge capacity [A.h]"].entries[-1]
                    - step_cv["Discharge capacity [A.h]"].entries[0]
                )
                cv_time = (
                    step_cv["Time [h]"].entries[-1] - step_cv["Time [h]"].entries[0]
                )

                # Variables
                sei_val = cycle_sol[sei_var].entries[-1] if sei_var else 0.0

                crack_val = 0.0
                if (
                    "X-averaged negative particle crack length [m]"
                    in cycle_sol.all_models[0].variables
                ):
                    crack_val = cycle_sol[
                        "X-averaged negative particle crack length [m]"
                    ].entries[-1]

                # Append
                data["cycles"].append(current_cycle_num)
                data["dis_caps"].append(dis_cap)
                data["cc_caps"].append(cc_cap)
                data["cc_times"].append(cc_time)
                data["cv_caps"].append(cv_cap)
                data["cv_times"].append(cv_time)
                data["sei_thickness"].append(sei_val)
                data["crack_lengths"].append(crack_val)

            starting_solution = sim.solution

            # Insert Rest Step Logic
            if (chunk_idx + 1) % 5 == 0 and (chunk_idx + 1) < NUM_CHUNKS:
                cycle_num = (chunk_idx + 1) * CYCLES_PER_CHUNK
                starting_solution, rest_data_item = run_rest_step(
                    cycle_num,
                    starting_solution,
                    options,
                    parameter_values,
                    solver,
                    var_pts,
                )
                data["rest_data"].append(rest_data_item)

        except Exception as e:
            print(f"  FAILED at chunk {chunk_idx + 1}: {e}")
            break  # Stop this variant

        # Clean up chunk
        del sim
        gc.collect()

    return data


def plot_results(results):
    print("\nGeneratng Plots...")

    # Create Figure (4 Rows x 2 Cols)
    fig, axs = plt.subplots(4, 2, figsize=(15, 16))

    # Metrics to plot configuration
    # (Row, Col, Key, Title, YLabel)
    metrics = [
        (0, 0, "dis_caps", "Discharge Capacity", "Discharge Capacity [A.h]"),
        (1, 0, "cc_caps", "CC Charge Capacity", "CC Capacity [A.h]"),
        (1, 1, "cc_times", "CC Charge Time", "CC Time [h]"),
        (2, 0, "cv_caps", "CV Charge Capacity", "CV Capacity [A.h]"),
        (2, 1, "cv_times", "CV Charge Time", "CV Time [h]"),
    ]

    # 1. Standard Metrics
    for r, c, key, title, ylabel in metrics:
        ax = axs[r, c]
        for name, d in results.items():
            ax.plot(d["cycles"], d[key], marker=".", label=name)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if r == 0 and c == 0:
            ax.legend(fontsize="small")

    # 2. Total Charge Capacity (Calculated)
    ax = axs[0, 1]
    for name, d in results.items():
        total = [cc + cv for cc, cv in zip(d["cc_caps"], d["cv_caps"])]
        ax.plot(d["cycles"], total, marker=".", label=name)
    ax.set_title("Total Charge Capacity (CC+CV)")
    ax.set_ylabel("Total Charge Capacity [A.h]")
    ax.grid(True)

    # 3. SEI Thickness
    ax = axs[3, 0]
    for name, d in results.items():
        ax.plot(d["cycles"], d["sei_thickness"], marker=".", label=name)
    ax.set_title("Total SEI Thickness")
    ax.set_ylabel("SEI Thickness [m]")
    ax.grid(True)

    # 4. SEI Growth Rate (Corrected)
    ax = axs[3, 1]
    correction_map = {
        "Electron Migration SEI": {51: 0.77e-9, 101: 0.56e-9, 151: 0.46e-9}
    }
    default_correction = {51: 1.13e-9, 101: 0.80e-9, 151: 0.65e-9}

    for name, d in results.items():
        cycles = d["cycles"]
        thickness = d["sei_thickness"]
        # Calculate rates
        rates = [0.0] * len(thickness)
        for i in range(1, len(thickness)):
            rates[i] = thickness[i] - thickness[i - 1]

        # Correct rates
        corr_dict = correction_map.get(name, default_correction)
        corr_rates = list(rates)
        for i, c in enumerate(cycles):
            if c in corr_dict:
                corr_rates[i] = max(0, corr_rates[i] - corr_dict[c])

        ax.plot(cycles, corr_rates, marker=".", label=name)
        # Store for CSV
        d["growth_rates"] = rates

    ax.set_title("SEI Growth Rate (Corrected)")
    ax.set_ylabel("Rate [m/cycle]")
    ax.set_yscale("log")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig("mechanism_isolation_detailed.png")
    fig.savefig("mechanism_isolation_detailed.svg")
    print("Main plot saved as .png and .svg")

    # Voltage Decay Plot
    rest_fig = plt.figure(figsize=(10, 6))
    has_data = False
    for name, d in results.items():
        if d["rest_data"]:
            has_data = True
            trace = d["rest_data"][0]["voltage_trace"]
            plt.plot(trace["time"], trace["voltage"], label=name)

    if has_data:
        plt.xlabel("Rest Time [h]")
        plt.ylabel("Voltage [V]")
        plt.title("Voltage Relaxation during 4-Day Rest (Cycle 50)")
        plt.legend()
        plt.grid(True)
        rest_fig.savefig("rest_voltage_decay.png")
        rest_fig.savefig("rest_voltage_decay.svg")
        print("Rest plot saved as .png and .svg")

    plt.close(rest_fig)

    # Cathode Spatial Distribution Plot
    spatial_fig = plt.figure(figsize=(10, 6))
    has_spatial_data = False
    for name, d in results.items():
        if d["rest_data"]:
            has_spatial_data = True
            dist = d["rest_data"][0]["spatial_dist"]
            # Normalize x to 0-1 if possible, or just plot vs m
            # x_p is usually near 1 for cathode in normalized coords, or actual meters.
            # Let's plot vs index or x directly.
            plt.plot(dist["x_p"], dist["c_s_surf_p"], label=name, marker='.')

    if has_spatial_data:
        plt.xlabel("Position in Cathode [m]")
        plt.ylabel("Surface Concentration [mol.m-3]")
        plt.title("Cathode Surface Concentration Distribution (Start of Rest, Cycle 50)")
        plt.legend()
        plt.grid(True)
        spatial_fig.savefig("cathode_spatial_dist.png")
        spatial_fig.savefig("cathode_spatial_dist.svg")
        print("Spatial distribution plot saved as .png and .svg")
    
    plt.close(spatial_fig)


def save_csv(results):
    print("Saving CSV...")
    rows = []
    for name, d in results.items():
        for i, c in enumerate(d["cycles"]):
            period = "Cycling"
            if c in [51, 101, 151]:
                period = "Post-Rest (96hr)"

            rows.append(
                {
                    "Variant": name,
                    "Cycle": c,
                    "Period": period,
                    "SEI Thickness [m]": d["sei_thickness"][i],
                    "Growth Rate [m/cycle]": d.get(
                        "growth_rates", [0] * len(d["cycles"])
                    )[i],
                    "Crack Length [m]": d["crack_lengths"][i],
                    "Discharge Capacity [A.h]": d["dis_caps"][i],
                    "CC Capacity [A.h]": d["cc_caps"][i],
                    "CC Time [h]": d["cc_times"][i],
                    "CV Capacity [A.h]": d["cv_caps"][i],
                    "CV Time [h]": d["cv_times"][i],
                    "Total Charge Capacity [A.h]": d["cc_caps"][i] + d["cv_caps"][i],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(CSV_FILENAME, index=False)
    print(f"Data saved to {CSV_FILENAME}")


# --- Main Execution ---
if __name__ == "__main__":
    results = {}
    for name, options in tqdm.tqdm(VARIANTS.items()):
        results[name] = run_chunked_simulation(name, options)

    plot_results(results)
    save_csv(results)
