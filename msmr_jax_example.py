#!/usr/bin/env python3
"""
Example: MSMR (Multi-Species Multi-Reaction) model with JAX solver

This example demonstrates:
1. Setting up an MSMR model with multiple reaction mechanisms
2. Using the JAX solver for automatic differentiation capabilities
3. Running a simple discharge simulation
4. Plotting the results

The MSMR model is more complex than standard models as it can handle:
- Multiple active materials in electrodes
- Multiple reaction pathways
- Different particle size distributions
"""

import matplotlib.pyplot as plt
import numpy as np

import pybamm


def main():
    print("PyBaMM MSMR Model with JAX Solver Example")
    print("=" * 50)

    # Create MSMR model with proper options
    print("1. Creating MSMR model...")
    # MSMR requires specifying the number of reactions
    options = {
        "number of MSMR reactions": (
            "1",
            "1",
        ),  # (negative electrode, positive electrode)
        "particle": "MSMR",
        "open-circuit potential": "MSMR",
    }
    model = pybamm.lithium_ion.MSMR(options=options)
    print(f"   Model: {model.name}")
    print(f"   Model options: {model.options}")

    # Get MSMR parameter set
    print("\n2. Loading MSMR parameter set...")
    try:
        # Use the MSMR example parameter set
        parameter_values = pybamm.ParameterValues("MSMR_Example")
        print("   Successfully loaded MSMR_Example parameter set")
    except Exception as e:
        print(f"   Error loading MSMR parameters: {e}")
        print("   Falling back to default parameters with MSMR modifications...")
        parameter_values = pybamm.ParameterValues("Chen2020")
        # You would typically modify parameters here for MSMR

    # Create JAX solver
    print("\n3. Setting up JAX solver...")
    try:
        jax_solver = pybamm.JaxSolver(rtol=1e-6, atol=1e-8)
        print("   JAX solver created successfully")
        print(f"   Solver: {jax_solver.name}")
    except Exception as e:
        print(f"   JAX solver not available: {e}")
        print("   Falling back to IDAKLU solver...")
        jax_solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-8)

    # Convert model to JAX format for JAX solver
    print("\n4. Converting model to JAX format...")
    model.convert_to_format = "jax"
    # Remove events as JAX solver doesn't support them
    print(f"   Removing {len(model.events)} events for JAX compatibility")
    model.events = []
    print("   Model converted to JAX format")

    # Create simulation
    print("\n5. Creating simulation...")
    sim = pybamm.Simulation(
        model=model, parameter_values=parameter_values, solver=jax_solver
    )
    print("   Simulation created successfully")

    # Define time vector (1 hour discharge at 1C)
    t_eval = np.linspace(0, 3600, 100)  # 1 hour in seconds

    print("\n6. Running simulation...")
    print("   Discharging at 1C for 1 hour...")
    try:
        # Solve the model
        solution = sim.solve(t_eval)
        print("   ✅ Simulation completed successfully!")
        print(f"   Final time: {solution.t[-1]:.1f} s")
        print(f"   Final voltage: {solution['Voltage [V]'].data[-1]:.3f} V")

        # Print some key variables
        print("\n7. Solution summary:")
        print(f"   Initial voltage: {solution['Voltage [V]'].data[0]:.3f} V")
        print(f"   Final voltage: {solution['Voltage [V]'].data[-1]:.3f} V")
        print(
            f"   Voltage drop: {solution['Voltage [V]'].data[0] - solution['Voltage [V]'].data[-1]:.3f} V"
        )

        # Get capacity
        capacity_Ah = solution["Discharge capacity [A.h]"].data[-1]
        print(f"   Discharge capacity: {capacity_Ah:.3f} A.h")

    except Exception as e:
        print(f"   ❌ Simulation failed: {e}")
        return

    # Plot results
    print("\n8. Plotting results...")
    try:
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("MSMR Model Results with JAX Solver", fontsize=14)

        # Plot 1: Voltage vs time
        ax1.plot(solution.t / 3600, solution["Voltage [V]"].data)
        ax1.set_xlabel("Time [h]")
        ax1.set_ylabel("Voltage [V]")
        ax1.set_title("Cell Voltage")
        ax1.grid(True)

        # Plot 2: Current vs time
        ax2.plot(solution.t / 3600, solution["Current [A]"].data)
        ax2.set_xlabel("Time [h]")
        ax2.set_ylabel("Current [A]")
        ax2.set_title("Current")
        ax2.grid(True)

        # Plot 3: Temperature vs time
        try:
            ax3.plot(solution.t / 3600, solution["Cell temperature [K]"].data - 273.15)
            ax3.set_xlabel("Time [h]")
            ax3.set_ylabel("Temperature [°C]")
            ax3.set_title("Cell Temperature")
            ax3.grid(True)
        except KeyError:
            # If temperature is not available
            ax3.plot(solution.t / 3600, solution["Discharge capacity [A.h]"].data)
            ax3.set_xlabel("Time [h]")
            ax3.set_ylabel("Capacity [A.h]")
            ax3.set_title("Discharge Capacity")
            ax3.grid(True)

        # Plot 4: State of charge
        try:
            soc_data = solution["Discharge capacity [A.h]"].data
            initial_capacity = parameter_values.evaluate(
                pybamm.Parameter("Nominal cell capacity [A.h]")
            )
            soc = 1 - soc_data / initial_capacity
            ax4.plot(solution.t / 3600, soc * 100)
            ax4.set_xlabel("Time [h]")
            ax4.set_ylabel("SOC [%]")
            ax4.set_title("State of Charge")
            ax4.grid(True)
        except Exception:
            # Fallback plot
            ax4.plot(solution.t / 3600, solution["Voltage [V]"].data)
            ax4.set_xlabel("Time [h]")
            ax4.set_ylabel("Voltage [V]")
            ax4.set_title("Voltage (duplicate)")
            ax4.grid(True)

        plt.tight_layout()

        # Save the plot
        plt.savefig(
            "/Users/mohitmehta/git_downloads/PyBaMM/msmr_jax_results.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("   📊 Plot saved as 'msmr_jax_results.png'")

        # Show plot
        plt.show()

    except Exception as e:
        print(f"   ⚠️  Plotting failed: {e}")
        # Still try to show basic info
        print("   Using PyBaMM's built-in plotting...")
        try:
            sim.plot()
            plt.savefig(
                "/Users/mohitmehta/git_downloads/PyBaMM/msmr_jax_basic_plot.png"
            )
            print("   📊 Basic plot saved as 'msmr_jax_basic_plot.png'")
        except Exception as plot_e:
            print(f"   Basic plotting also failed: {plot_e}")

    print("\n" + "=" * 50)
    print("MSMR JAX Example Complete!")
    print("Key features demonstrated:")
    print("- MSMR model with multiple reaction mechanisms")
    print("- JAX solver with automatic differentiation")
    print("- 1C discharge simulation")
    print("- Results visualization")


if __name__ == "__main__":
    main()
