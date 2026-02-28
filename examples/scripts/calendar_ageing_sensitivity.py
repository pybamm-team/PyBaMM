"""
Calendar Ageing Sensitivity Analysis for PyBaMM.

This script runs calendar ageing simulations across the four major SEI models,
testing their primary driving parameters at 0.1x, 1.0x, and 10.0x multipliers.
Outputs results to CSV and generates a comparative visual plot.
"""

import argparse
import concurrent.futures
import csv
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import pybamm as pb

# Parameter mapping for each SEI model type
MODEL_PARAMS = {
    "reaction limited": "SEI kinetic rate constant [m.s-1]",
    "solvent-diffusion limited": "SEI solvent diffusivity [m2.s-1]",
    "electron-migration limited": "SEI electron conductivity [S.m-1]",
    "interstitial-diffusion limited": "SEI lithium interstitial diffusivity [m2.s-1]",
}


def setup_logger(verbose: bool) -> logging.Logger:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    pb.set_logging_level("WARNING")
    return logging.getLogger(__name__)


def run_simulation(
    args: tuple[str, float, int, float],
) -> tuple[
    str,
    float,
    float,
    float,
    int,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    str | None,
]:
    """
    Run a calendar ageing simulation for a given SEI model, multiplier, duration, and SOC.

    Returns:
        A tuple of (sei_model, multiplier, actual_param_value, initial_soc, days, voltage_before, voltage_after, delta_v, sei_thickness, lli, error_message).
    """
    sei_model, param_multiplier, days, initial_soc = args
    param_name = MODEL_PARAMS[sei_model]
    actual_param_value = 0.0

    try:
        model = pb.lithium_ion.DFN({"SEI": sei_model})
        parameter_values = model.default_parameter_values

        parameter_values["Current function [A]"] = 0

        # Apply the exact sensitivity multiplier to the specific driving parameter
        base_val = parameter_values[param_name]
        actual_param_value = base_val * param_multiplier
        parameter_values[param_name] = actual_param_value

        sim = pb.Simulation(model, parameter_values=parameter_values)
        solver = pb.IDAKLUSolver()

        seconds = days * 24 * 60 * 60
        t_eval = np.linspace(0, seconds, 100)

        sol = sim.solve(t_eval=t_eval, solver=solver, initial_soc=initial_soc)

        voltage = sol["Voltage [V]"].entries
        voltage_before = float(voltage[0])
        voltage_after = float(voltage[-1])
        delta_v = voltage_before - voltage_after

        sei_thickness = float(sol["X-averaged negative SEI thickness [m]"].entries[-1])
        lli = float(sol["Loss of lithium inventory [%]"].entries[-1])

        return (
            sei_model,
            param_multiplier,
            actual_param_value,
            initial_soc,
            days,
            voltage_before,
            voltage_after,
            delta_v,
            sei_thickness,
            lli,
            None,
        )
    except Exception as e:
        return (
            sei_model,
            param_multiplier,
            actual_param_value,
            initial_soc,
            days,
            None,
            None,
            None,
            None,
            None,
            str(e),
        )


def save_to_csv(
    csv_filename: Path, csv_rows: list[list[Any]], logger: logging.Logger
) -> None:
    """Save simulation results to CSV."""
    try:
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "SEI Model",
                    "Multiplier",
                    "Param Value",
                    "Initial SOC",
                    "Days",
                    "Voltage Before (V)",
                    "Voltage After (V)",
                    "Delta V (V)",
                    "SEI Thickness (m)",
                    "Loss of Lithium Inventory (%)",
                ]
            )
            writer.writerows(csv_rows)
        logger.info(f"Results successfully saved to: {csv_filename}")
    except OSError as e:
        logger.error(f"Failed to write CSV file to {csv_filename}: {e}")


def plot_results(
    plot_filename: Path,
    results_dict: dict[str, dict[float, dict[str, list[Any]]]],
    models_to_test: list[str],
    multipliers: list[float],
    logger: logging.Logger,
) -> None:
    """Generate and save comparative plots of the results."""
    # 2x2 Grid for the four models
    _fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, model in enumerate(models_to_test):
        ax = axes[idx]
        plotted_any = False

        for mult in multipliers:
            if not results_dict[model][mult]["days"]:
                logger.warning(
                    f"No successful simulations to plot for {model} at {mult}x"
                )
                continue

            delta_v_mv = [dv * 1000 for dv in results_dict[model][mult]["delta_v"]]
            ax.plot(
                results_dict[model][mult]["days"],
                delta_v_mv,
                marker="o",
                linestyle="-",
                label=f"{mult}x multiplier",
            )
            plotted_any = True

        ax.set_title(f"Voltage Loss vs Days\n{model}\n({MODEL_PARAMS[model]})")
        ax.set_xlabel("Rest Duration (Days)")
        ax.set_ylabel("Voltage Loss (mV)")
        ax.grid(True, linestyle="--", alpha=0.7)
        if plotted_any:
            ax.legend()

    plt.tight_layout()

    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        logger.info(f"Plot successfully saved to: {plot_filename}")
    except OSError as e:
        logger.error(f"Failed to save plot to {plot_filename}: {e}")
    finally:
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run calendar ageing parameter sensitivity simulations."
    )
    parser.add_argument(
        "--multipliers",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 10.0],
        help="List of multipliers to apply to the primary driving parameter corresponding to the SEI model",
    )
    parser.add_argument(
        "--soc", type=float, default=0.9, help="Initial State of Charge (default: 0.9)"
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
        "--output-dir", type=str, default=".", help="Directory to save output files"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.verbose)

    models_to_test = list(MODEL_PARAMS.keys())
    multipliers: list[float] = args.multipliers
    day_range: list[int] = list(range(args.min_days, args.max_days + 1))

    output_dir = (
        Path(__file__).parent if args.output_dir == "." else Path(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compile the matrix of parameter variations across all days and models
    simulation_args = [
        (model, mult, days, args.soc)
        for model in models_to_test
        for mult in multipliers
        for days in day_range
    ]

    logger.info(
        f"Starting sensitivity simulations across {len(models_to_test)} models and "
        f"{len(multipliers)} parameter multipliers at {args.soc*100:.0f}% SOC..."
    )
    start_time = time.time()

    # Dictionary Map format: results_dict[model][multiplier]["days"] = [...]
    results_dict: dict[str, dict[float, dict[str, list[Any]]]] = {
        model: {mult: {"days": [], "delta_v": []} for mult in multipliers}
        for model in models_to_test
    }
    csv_rows: list[list[Any]] = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_simulation, simulation_args))

    execution_time = time.time() - start_time
    logger.info(f"Total simulation execution time: {execution_time:.2f} seconds")

    for res in results:
        (
            model,
            mult,
            val,
            soc,
            days,
            v_before,
            v_after,
            d_v,
            sei_thickness,
            lli,
            error,
        ) = res
        if error or v_before is None or v_after is None or d_v is None:
            logger.error(f"Error for {model} at {mult}x on day {days}: {error}")
            continue

        results_dict[model][mult]["days"].append(days)
        results_dict[model][mult]["delta_v"].append(d_v)
        csv_rows.append(
            [model, mult, val, soc, days, v_before, v_after, d_v, sei_thickness, lli]
        )

    save_to_csv(
        output_dir / "calendar_ageing_sensitivity_results.csv", csv_rows, logger
    )
    plot_results(
        output_dir / "calendar_ageing_sensitivity_plot.png",
        results_dict,
        models_to_test,
        multipliers,
        logger,
    )


if __name__ == "__main__":
    main()
