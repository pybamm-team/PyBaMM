"""
Calendar Ageing Simulation Script for PyBaMM.

This script runs calendar ageing simulations using the Doyle-Fuller-Newman (DFN) model
with reaction-limited, solvent-diffusion limited, electron-migration limited, and
interstitial-diffusion limited SEI growth models. It evaluates voltage loss across specified rest
durations and Initial States of Charge (SOC), exporting results to a CSV and generating
a comparative plot.

Optimized for parallel execution using Python's concurrent.futures.
"""

import argparse
import concurrent.futures
import csv
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pybamm as pb


def setup_logger(verbose: bool) -> logging.Logger:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # PyBaMM's internal logging
    pb.set_logging_level("WARNING")
    return logging.getLogger(__name__)


def run_simulation(args: Tuple[str, int, float]) -> Tuple[str, float, int, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[str]]:
    """
    Run a single calendar ageing simulation for a given SEI model, duration, and SOC.

    Args:
        args: A tuple containing (sei_model, days, initial_soc).

    Returns:
        A tuple of (sei_model, initial_soc, days, voltage_before, voltage_after, delta_v, sei_thickness, lli, sei_rxn_rate, error_message).
    """
    sei_model, days, initial_soc = args
    try:
        # Create a new DFN model with reaction limited SEI
        model = pb.lithium_ion.DFN({"SEI": sei_model})
        parameter_values = model.default_parameter_values

        # Set current to 0 for rest (calendar ageing)
        parameter_values["Current function [A]"] = 0

        # Setup simulation
        sim = pb.Simulation(model, parameter_values=parameter_values)
        solver = pb.IDAKLUSolver()

        # Calculate time vector in seconds
        seconds = days * 24 * 60 * 60
        t_eval = np.linspace(0, seconds, 100)

        # Solve with specific initial SOC
        sol = sim.solve(t_eval=t_eval, solver=solver, initial_soc=initial_soc)

        # Extract voltage data
        voltage = sol["Voltage [V]"].entries
        voltage_before = float(voltage[0])
        voltage_after = float(voltage[-1])
        delta_v = voltage_before - voltage_after
        
        sei_thickness = float(sol["X-averaged negative SEI thickness [m]"].entries[-1])
        lli = float(sol["Loss of lithium inventory [%]"].entries[-1])
        
        if sei_model == "reaction limited":
            sei_rxn_rate = parameter_values["SEI kinetic rate constant [m.s-1]"]
        else:
            sei_rxn_rate = 0.0

        return sei_model, initial_soc, days, voltage_before, voltage_after, delta_v, sei_thickness, lli, sei_rxn_rate, None
    except Exception as e:
        return sei_model, initial_soc, days, None, None, None, None, None, None, str(e)


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run calendar ageing simulations in PyBaMM.")
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["reaction limited", "solvent-diffusion limited", "electron-migration limited", "interstitial-diffusion limited"],
        help="List of SEI models to test",
    )
    parser.add_argument(
        "--socs",
        nargs="+",
        type=float,
        default=[0.3, 0.9],
        help="List of Initial States of Charge to test (default: 0.3 0.9)",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=2,
        help="Minimum rest duration in days (default: 2)",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=80,
        help="Maximum rest duration in days (default: 80)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save output files (default: current directory)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logger = setup_logger(args.verbose)

    models_to_test: List[str] = args.models
    socs_to_test: List[float] = args.socs
    day_range: List[int] = list(range(args.min_days, args.max_days + 1))
    output_dir = Path(__file__).parent if args.output_dir == "." else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create an array of argument tuples for parallel execution
    simulation_args = [(model, days, soc) for model in models_to_test for soc in socs_to_test for days in day_range]

    logger.info(
        f"Starting simulations for {len(models_to_test)} models, SOCs {socs_to_test} across {len(day_range)} "
        f"durations ({args.min_days} to {args.max_days} days)..."
    )
    start_time = time.time()

    results_dict: dict[str, dict[float, dict[str, List[Any]]]] = {
        model: {soc: {"days": [], "delta_v": []} for soc in socs_to_test} for model in models_to_test
    }
    csv_rows: List[List[Any]] = []

    # Process all combinations in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_simulation, simulation_args))

    execution_time = time.time() - start_time
    logger.info(f"Total simulation execution time: {execution_time:.2f} seconds")

    # Process results locally and populate data structures for CSV and Plotting
    for res in results:
        model, soc, days, v_before, v_after, d_v, sei_thickness, lli, sei_rxn_rate, error = res
        if error or v_before is None or v_after is None or d_v is None:
            logger.error(f"Error for Model {model} SOC {soc:.0%} at {days} days: {error}")
            continue

        results_dict[model][soc]["days"].append(days)
        results_dict[model][soc]["delta_v"].append(d_v)
        csv_rows.append([model, soc, days, v_before, v_after, d_v, sei_thickness, lli, sei_rxn_rate])

    # 1. Write to CSV
    csv_filename = output_dir / "calendar_ageing_results_models.csv"
    try:
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["SEI Model", "Initial SOC", "Days", "Voltage Before (V)", "Voltage After (V)", "Delta V (V)", "SEI Thickness (m)", "Loss of Lithium Inventory (%)", "SEI rxn rate"]
            )
            writer.writerows(csv_rows)
        logger.info(f"Results successfully saved to: {csv_filename}")
    except IOError as e:
        logger.error(f"Failed to write CSV file to {csv_filename}: {e}")

    # 2. Plotting the results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for idx, soc in enumerate(socs_to_test):
        ax = axes[idx]
        for model in models_to_test:
            if not results_dict[model][soc]["days"]:
                logger.warning(f"No successful simulations to plot for {model} at SOC {soc:.0%}")
                continue

            # Convert delta_v to millivolts for better readability
            delta_v_mv = [dv * 1000 for dv in results_dict[model][soc]["delta_v"]]
            ax.plot(
                results_dict[model][soc]["days"],
                delta_v_mv,
                marker="o",
                linestyle="-",
                label=f"{model}",
            )

        ax.set_title(f"Voltage Loss vs Rest Duration for {soc * 100:.0f}% SOC")
        ax.set_xlabel("Rest Duration (Days)")
        ax.set_ylabel("Voltage Loss (mV)")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

    plt.tight_layout()

    plot_filename = output_dir / "calendar_ageing_models_plot.png"
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        logger.info(f"Plot successfully saved to: {plot_filename}")
    except IOError as e:
        logger.error(f"Failed to save plot to {plot_filename}: {e}")
    finally:
        plt.close()


if __name__ == "__main__":
    main()
