"""
Calendar Ageing and Subsequent Cycling Analysis for PyBaMM.

This script simulates a period of calendar ageing (rest) followed by
continuous cycling chunks to evaluate degradation.
"""

import argparse
import gc
import logging
import os
import time
from pathlib import Path

# Force CPU to avoid Metal/JAX instability
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

import pybamm

# --- Configuration ---
# You can tweak these or override them using command line arguments
DEFAULT_STORAGE_DAYS = 30
DEFAULT_INITIAL_SOC = 0.9
DEFAULT_TOTAL_CYCLES = 100
DEFAULT_CYCLES_PER_CHUNK = 10
CSV_FILENAME = "ageing_and_cycling_analysis.csv"

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

def setup_logger(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    pybamm.set_logging_level("WARNING")
    return logging.getLogger(__name__)

def get_submesh_types(model_class):
    """Returns the submesh types (Uniform default)."""
    model = model_class()
    submesh_types = model.default_submesh_types.copy()
    return submesh_types

def run_ageing_phase(options, storage_days, initial_soc, logger):
    """Runs the initial calendar ageing (storage) phase."""
    logger.info(f"    Running initial {storage_days}-day storage ageing phase...")
    model = pybamm.lithium_ion.DFN(options)
    parameter_values = pybamm.ParameterValues("OKane2022")
    
    # Ensure no current during ageing if not implicitly handled
    parameter_values["Current function [A]"] = 0

    solver = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8)
    var_pts = {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 50, "r_p": 50}

    seconds = storage_days * 24 * 60 * 60
    t_eval = np.linspace(0, seconds, min(100, max(20, storage_days * 2)))
    
    sim = pybamm.Simulation(
        model, 
        parameter_values=parameter_values,
        solver=solver,
        var_pts=var_pts,
        submesh_types=get_submesh_types(pybamm.lithium_ion.DFN)
    )

    sol = sim.solve(t_eval=t_eval, initial_soc=initial_soc)
    
    sei_var = None
    if "X-averaged total SEI thickness [m]" in sol.all_models[0].variables:
        sei_var = "X-averaged total SEI thickness [m]"
    elif "X-averaged negative SEI thickness [m]" in sol.all_models[0].variables:
        sei_var = "X-averaged negative SEI thickness [m]"

    sei_thickness = sol[sei_var].entries[-1] if sei_var else 0.0
    loss_li = sol["Loss of lithium inventory [%]"].entries[-1] if "Loss of lithium inventory [%]" in sol.all_models[0].variables else 0.0
    
    logger.info(f"    Storage complete. SEI Thickness: {sei_thickness:.4e} m, LLI: {loss_li:.2f}%")
    
    return sim.solution

def run_cycling_phase(name, options, starting_solution, total_cycles, cycles_per_chunk, logger):
    """Runs the chunked cycling phase using the aged solution as the starting point."""
    logger.info(f"    Starting cycling phase ({total_cycles} cycles)...")

    num_chunks = total_cycles // cycles_per_chunk
    experiment_chunk = pybamm.Experiment([EXPERIMENT_STEP] * cycles_per_chunk)

    data = {
        "cycles": [],
        "cc_caps": [],
        "cc_times": [],
        "cv_caps": [],
        "cv_times": [],
        "dis_caps": [],
        "sei_thickness": [],
        "crack_lengths": [],
    }

    var_pts = {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 50, "r_p": 50}
    parameter_values = pybamm.ParameterValues("OKane2022")
    solver = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8)

    current_solution = starting_solution

    for chunk_idx in range(num_chunks):
        start_cycle = chunk_idx * cycles_per_chunk + 1
        logger.debug(f"      {name}: Chunk {chunk_idx + 1}/{num_chunks} (Cycles {start_cycle}-{start_cycle+cycles_per_chunk-1})...")

        model = pybamm.lithium_ion.DFN(options)
        submesh_types = get_submesh_types(pybamm.lithium_ion.DFN)

        sim = pybamm.Simulation(
            model,
            experiment=experiment_chunk,
            parameter_values=parameter_values,
            solver=solver,
            var_pts=var_pts,
            submesh_types=submesh_types,
        )

        sim.model.set_initial_conditions_from(current_solution)

        try:
            sim.solve()
            
            sol = sim.solution
            sei_var = None
            if "X-averaged total SEI thickness [m]" in sol.all_models[0].variables:
                sei_var = "X-averaged total SEI thickness [m]"
            elif "X-averaged negative SEI thickness [m]" in sol.all_models[0].variables:
                sei_var = "X-averaged negative SEI thickness [m]"

            for i, cycle_sol in enumerate(sol.cycles):
                current_cycle_num = start_cycle + i

                step_dis = cycle_sol.steps[0]
                dis_cap = abs(step_dis["Discharge capacity [A.h]"].entries[-1] - step_dis["Discharge capacity [A.h]"].entries[0])

                step_cc = cycle_sol.steps[2]
                cc_cap = abs(step_cc["Discharge capacity [A.h]"].entries[-1] - step_cc["Discharge capacity [A.h]"].entries[0])
                cc_time = step_cc["Time [h]"].entries[-1] - step_cc["Time [h]"].entries[0]

                step_cv = cycle_sol.steps[3]
                cv_cap = abs(step_cv["Discharge capacity [A.h]"].entries[-1] - step_cv["Discharge capacity [A.h]"].entries[0])
                cv_time = step_cv["Time [h]"].entries[-1] - step_cv["Time [h]"].entries[0]

                sei_val = cycle_sol[sei_var].entries[-1] if sei_var else 0.0

                crack_val = 0.0
                if "X-averaged negative particle crack length [m]" in cycle_sol.all_models[0].variables:
                    crack_val = cycle_sol["X-averaged negative particle crack length [m]"].entries[-1]

                data["cycles"].append(current_cycle_num)
                data["dis_caps"].append(dis_cap)
                data["cc_caps"].append(cc_cap)
                data["cc_times"].append(cc_time)
                data["cv_caps"].append(cv_cap)
                data["cv_times"].append(cv_time)
                data["sei_thickness"].append(sei_val)
                data["crack_lengths"].append(crack_val)

            current_solution = sim.solution

        except Exception as e:
            logger.error(f"      FAILED at chunk {chunk_idx + 1}: {e}")
            break

        del sim
        gc.collect()

    return data

def plot_results(results, output_dir, logger):
    logger.info("Generating plots...")

    fig, axs = plt.subplots(3, 2, figsize=(15, 12))

    metrics = [
        (0, 0, "dis_caps", "Discharge Capacity", "Discharge Capacity [A.h]"),
        (1, 0, "cc_caps", "CC Charge Capacity", "CC Capacity [A.h]"),
        (1, 1, "cc_times", "CC Charge Time", "CC Time [h]"),
        (2, 0, "cv_caps", "CV Charge Capacity", "CV Capacity [A.h]"),
        (2, 1, "cv_times", "CV Charge Time", "CV Time [h]"),
    ]

    for r, c, key, title, ylabel in metrics:
        ax = axs[r, c]
        for name, d in results.items():
            ax.plot(d["cycles"], d[key], marker=".", label=name)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if r == 0 and c == 0:
            ax.legend(fontsize="small")

    ax = axs[0, 1]
    for name, d in results.items():
        total = [cc + cv for cc, cv in zip(d["cc_caps"], d["cv_caps"])]
        ax.plot(d["cycles"], total, marker=".", label=name)
    ax.set_title("Total Charge Capacity (CC+CV)")
    ax.set_ylabel("Total Charge Capacity [A.h]")
    ax.grid(True)

    fig.tight_layout()
    plot_path = output_dir / "ageing_and_cycling_capacities.png"
    fig.savefig(plot_path)
    logger.info(f"Capacity plot saved to {plot_path}")
    plt.close(fig)

    fig2, axs2 = plt.subplots(1, 2, figsize=(15, 5))
    
    ax = axs2[0]
    for name, d in results.items():
        ax.plot(d["cycles"], d["sei_thickness"], marker=".", label=name)
    ax.set_title("Total SEI Thickness")
    ax.set_ylabel("SEI Thickness [m]")
    ax.grid(True)
    ax.legend(fontsize="small")

    ax = axs2[1]
    for name, d in results.items():
        cycles = d["cycles"]
        thickness = d["sei_thickness"]
        rates = [0.0] * len(thickness)
        for i in range(1, len(thickness)):
            rates[i] = thickness[i] - thickness[i - 1]
        
        # Minor correction for anomalous jumps like isolate_mechanism, but simplified here
        # or just plotted raw as we don't have interlaced rests
        ax.plot(cycles, rates, marker=".", label=name)
        d["growth_rates"] = rates

    ax.set_title("SEI Growth Rate")
    ax.set_ylabel("Rate [m/cycle]")
    ax.set_yscale("log")
    ax.grid(True)

    fig2.tight_layout()
    sei_plot_path = output_dir / "ageing_and_cycling_sei.png"
    fig2.savefig(sei_plot_path)
    logger.info(f"SEI plot saved to {sei_plot_path}")
    plt.close(fig2)

def save_csv(results, output_dir, logger):
    logger.info("Saving CSV data...")
    rows = []
    for name, d in results.items():
        for i, c in enumerate(d["cycles"]):
            rows.append(
                {
                    "Variant": name,
                    "Cycle": c,
                    "SEI Thickness [m]": d["sei_thickness"][i],
                    "Growth Rate [m/cycle]": d.get("growth_rates", [0] * len(d["cycles"]))[i],
                    "Crack Length [m]": d["crack_lengths"][i],
                    "Discharge Capacity [A.h]": d["dis_caps"][i],
                    "CC Capacity [A.h]": d["cc_caps"][i],
                    "CC Time [h]": d["cc_times"][i],
                    "CV Capacity [A.h]": d["cv_caps"][i],
                    "CV Time [h]": d["cv_times"][i],
                    "Total Charge Capacity [A.h]": d["cc_caps"][i] + d["cv_caps"][i],
                }
            )

    csv_path = output_dir / CSV_FILENAME
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"Data saved to {csv_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Calendar Ageing followed by Cycling Analysis")
    parser.add_argument("--storage-days", type=int, default=DEFAULT_STORAGE_DAYS, help="Number of days for initial calendar ageing")
    parser.add_argument("--soc", type=float, default=DEFAULT_INITIAL_SOC, help="Initial SOC before storage (default 0.9)")
    parser.add_argument("--cycles", type=int, default=DEFAULT_TOTAL_CYCLES, help="Total number of subsequent cycles")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CYCLES_PER_CHUNK, help="Cycles per chunk (for memory management)")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save outputs")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--test-mode", action="store_true", help="Run a fast 2-cycle variant sweep for testing")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger(args.verbose)
    
    total_cycles = 2 if args.test_mode else args.cycles
    chunk_size = 1 if args.test_mode else args.chunk_size
    storage_days = 1 if args.test_mode else args.storage_days
    
    output_dir = Path(__file__).parent if args.output_dir == "." else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting analysis. Mode: {'TEST' if args.test_mode else 'FULL'}")
    logger.info(f"  Storage Phase: {storage_days} days at {args.soc*100:.0f}% SOC")
    logger.info(f"  Cycling Phase: {total_cycles} cycles (chunked by {chunk_size})")

    start_time = time.time()
    results = {}

    for name, options in VARIANTS.items():
        logger.info(f"--- Processing Variant: {name} ---")
        
        # 1. Ageing Phase
        aged_solution = run_ageing_phase(options, storage_days, args.soc, logger)
        
        # 2. Cycling Phase
        cycling_data = run_cycling_phase(name, options, aged_solution, total_cycles, chunk_size, logger)
        if cycling_data["cycles"]:
            results[name] = cycling_data
            
        logger.info(f"--- Completed Variant: {name} ---")

    execution_time = time.time() - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds")

    if results:
        plot_results(results, output_dir, logger)
        save_csv(results, output_dir, logger)
    else:
        logger.warning("No data generated for any variant. Check errors.")

if __name__ == "__main__":
    main()
