"""
Single MSMR Simulation Comparison

This script runs a single MSMR simulation with default parameters and compares
the results with experimental data to understand baseline performance.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import pybamm

# Set up logging
pybamm.set_logging_level("WARNING")


def load_experimental_data():
    """Load experimental data from CSV file."""
    try:
        df = pd.read_csv("test_02.csv")

        # Extract data
        time_data = df["TestTime(s)"].values
        current_data = df["Current(A)"].values
        voltage_data = df["Voltage(V)"].values
        capacity_data = df["Capacity(Ah)"].values

        # Normalize capacity to 0-1 range
        capacity_data = capacity_data / capacity_data.max()

        print(f"✓ Loaded experimental data: {len(capacity_data):,} points")
        print(f"  Time range: {time_data[0]:.1f} to {time_data[-1]:.1f} seconds")
        print(f"  Current: {current_data[0]:.4f} A (constant)")
        print(
            f"  Voltage range: {voltage_data.min():.4f} to {voltage_data.max():.4f} V"
        )
        print(
            f"  Capacity range: {capacity_data.min():.6f} to {capacity_data.max():.6f}"
        )

        return time_data, current_data, voltage_data, capacity_data

    except Exception as e:
        print(f"✗ Error loading experimental data: {e}")
        return None, None, None, None


def c(t):
    return 5


def run_single_msmr_simulation(time_data, current_data, voltage_data):
    """Run a single MSMR simulation with default parameters."""
    try:
        print("\nCreating MSMR model...")
        model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})

        print("Loading default parameters...")
        param = pybamm.ParameterValues("MSMR_Example")
        # Modify the parameters of the MSMR model
        param["Positive electrode host site occupancy fraction (0)"] = 0.14442
        param["Positive electrode host site standard potential (0) [V]"] = 3.62274
        param["Positive electrode host site ideality factor (0)"] = c
        # param["Positive electrode host site charge transfer coefficient (0)"] = 0.5
        # param["Positive electrode host site reference exchange-current density (0) [A.m-2]"] = 5
        # param["Negative electrode host site occupancy fraction (0)"] = 0.43336
        # param["Negative electrode host site standard potential (0) [V]"] = 0.08843
        param["Negative electrode host site ideality factor (0)"] = 5
        # param["Negative electrode host site charge transfer coefficient (0)"] = 0.5
        # param["Negative electrode host site reference exchange-current density (0) [A.m-2]"] = 2.7
        for domain in ["negative", "positive"]:
            Electrode = domain.capitalize()
            # Loop over reactions
            N = int(param["Number of reactions in " + domain + " electrode"])
            for i in range(N):
                names = [
                    f"{Electrode} electrode host site occupancy fraction ({i})",
                    f"{Electrode} electrode host site standard potential ({i}) [V]",
                    f"{Electrode} electrode host site ideality factor ({i})",
                    f"{Electrode} electrode host site charge transfer coefficient ({i})",
                    f"{Electrode} electrode host site reference exchange-current density ({i}) [A.m-2]",
                ]
                for name in names:
                    print(f"{name} = {param[name]}")
        # Use constant current (mean of experimental current)
        # current_mean = np.mean(current_data)
        # print(f"Using constant current: {current_mean:.4f} A")
        # param["Current function [A]"] = 0.5

        print("Creating simulation...")
        sim = pybamm.Simulation(model, parameter_values=param)
        experiment = pybamm.Experiment(
            [
                ("Discharge at C/9 until 3 V",),
            ],
        )
        sim = pybamm.Simulation(model, experiment=experiment)
        solution = sim.solve()
        # solution = sim.solution
        voltage_sim = solution["Voltage [V]"].entries
        capacity_sim = solution["Discharge capacity [A.h]"].entries
        time_sim = solution["Time [s]"].entries

        print("Simulation completed successfully!")
        print(f"Simulation time range: {time_sim[0]:.2f} to {time_sim[-1]:.2f} seconds")
        print(
            f"Simulation voltage range: {voltage_sim.min():.4f} to {voltage_sim.max():.4f} V"
        )
        print(
            f"Simulation capacity range: {capacity_sim.min():.6f} to {capacity_sim.max():.6f} Ah"
        )

        # Normalize capacity to 0-1 range for comparison
        if len(capacity_sim) > 0 and capacity_sim.max() > 0:
            capacity_sim_norm = capacity_sim / capacity_sim.max()
        else:
            capacity_sim_norm = np.linspace(0, 1, len(voltage_sim))

        return time_sim, voltage_sim, capacity_sim_norm, capacity_sim

    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        return None, None, None, None


def calculate_mse(exp_data, sim_data):
    """Calculate Mean Squared Error between experimental and simulated data."""
    if len(exp_data) != len(sim_data):
        # Interpolate simulation data to match experimental data points
        sim_interp = interp1d(
            np.linspace(0, 1, len(sim_data)),
            sim_data,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        sim_data_interp = sim_interp(np.linspace(0, 1, len(exp_data)))
    else:
        sim_data_interp = sim_data

    mse = np.mean((exp_data - sim_data_interp) ** 2)
    return mse, sim_data_interp


def plot_comparison(
    time_exp,
    voltage_exp,
    capacity_exp,
    time_sim,
    voltage_sim,
    capacity_sim_norm,
    capacity_sim_raw,
):
    """Plot comparison between experimental and simulated data."""

    # Calculate MSE
    voltage_mse, voltage_sim_interp = calculate_mse(voltage_exp, voltage_sim)
    capacity_mse, capacity_sim_interp = calculate_mse(capacity_exp, capacity_sim_norm)
    total_mse = voltage_mse + capacity_mse

    print("\n📊 MSE Analysis:")
    print(f"  Voltage MSE: {voltage_mse:.6f}")
    print(f"  Capacity MSE: {capacity_mse:.6f}")
    print(f"  Total MSE: {total_mse:.6f}")

    # Create plots with better styling
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "MSMR Simulation vs Experimental Data Comparison",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Voltage vs Time
    ax1.plot(
        time_exp, voltage_exp, "b-", label="Experimental", linewidth=1.5, alpha=0.8
    )
    ax1.plot(time_sim, voltage_sim, "r--", label="MSMR Simulation", linewidth=2)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Voltage [V]")
    ax1.set_title("Voltage vs Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(
        0.02,
        0.98,
        f"MSE: {voltage_mse:.6f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    # Plot 2: Capacity vs Time
    ax2.plot(
        time_exp,
        capacity_exp,
        "g-",
        label="Experimental (Normalized)",
        linewidth=1.5,
        alpha=0.8,
    )
    ax2.plot(
        time_sim,
        capacity_sim_norm,
        "m--",
        label="MSMR Simulation (Normalized)",
        linewidth=2,
    )
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Normalized Capacity")
    ax2.set_title("Normalized Capacity vs Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(
        0.02,
        0.98,
        f"MSE: {capacity_mse:.6f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
    )

    # Plot 3: Voltage vs Capacity (Normalized)
    ax3.plot(
        capacity_exp, voltage_exp, "b-", label="Experimental", linewidth=1.5, alpha=0.8
    )
    ax3.plot(
        capacity_sim_norm, voltage_sim, "r--", label="MSMR Simulation", linewidth=2
    )
    ax3.set_xlabel("Normalized Capacity")
    ax3.set_ylabel("Voltage [V]")
    ax3.set_title("Voltage vs Normalized Capacity")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.text(
        0.02,
        0.98,
        f"Total MSE: {total_mse:.6f}",
        transform=ax3.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    # Plot 4: Raw Capacity vs Time
    ax4.plot(
        time_sim, capacity_sim_raw, "purple", label="MSMR Raw Capacity", linewidth=2
    )
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Capacity [Ah]")
    ax4.set_title("Raw Simulation Capacity vs Time")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.text(
        0.02,
        0.98,
        f"Max Capacity: {capacity_sim_raw.max():.4f} Ah",
        transform=ax4.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()

    return voltage_mse, capacity_mse


def main():
    """Main function to run the comparison."""
    print("🚀 === Single MSMR Simulation Comparison ===")

    # Load experimental data
    print("\n1️⃣ Loading experimental data...")
    time_exp, current_exp, voltage_exp, capacity_exp = load_experimental_data()

    if time_exp is None:
        print("❌ Failed to load experimental data. Exiting.")
        return

    # Run single MSMR simulation
    print("\n2️⃣ Running MSMR simulation...")
    time_sim, voltage_sim, capacity_sim_norm, capacity_sim_raw = (
        run_single_msmr_simulation(time_exp, current_exp, voltage_exp)
    )

    if time_sim is None:
        print("❌ Failed to run simulation. Exiting.")
        return

    # Plot comparison
    print("\n3️⃣ Creating comparison plots...")
    voltage_mse, capacity_mse = plot_comparison(
        time_exp,
        voltage_exp,
        capacity_exp,
        time_sim,
        voltage_sim,
        capacity_sim_norm,
        capacity_sim_raw,
    )

    # Summary
    total_mse = voltage_mse + capacity_mse
    print("\n📋 === Summary ===")
    print(f"  Voltage MSE: {voltage_mse:.6f}")
    print(f"  Capacity MSE: {capacity_mse:.6f}")
    print(f"  Total MSE: {total_mse:.6f}")
    print(f"  Positive electrode host site ideality factor (0): {c(0):.4f}")

    # Performance assessment
    if total_mse < 0.01:
        print("  🎉 Excellent fit! MSE < 0.01")
    elif total_mse < 0.05:
        print("  ✅ Good fit! MSE < 0.05")
    elif total_mse < 0.1:
        print("  ⚠️  Fair fit. MSE < 0.1")
    else:
        print("  ❌ Poor fit. MSE >= 0.1")

    print("  ✅ Simulation completed successfully!")


if __name__ == "__main__":
    main()
