"""
Simple Example: Bayesian Optimization with PyBaMM

This script shows a basic example of using BayesianOptimization to fit
PyBaMM parameters to experimental data.
"""

import numpy as np
from bayes_opt import BayesianOptimization

import pybamm

# Set up logging
pybamm.set_logging_level("WARNING")


def create_synthetic_data():
    """Create synthetic experimental data for demonstration."""
    # Simulate voltage vs capacity data
    capacity = np.linspace(0, 1, 20)
    voltage = 4.0 - 0.8 * capacity + 0.1 * np.sin(3 * np.pi * capacity)
    voltage += 0.02 * np.random.normal(0, 1, len(voltage))
    return capacity, voltage


def run_simulation(param1, param2):
    """
    Run a simple PyBaMM simulation with two parameters to optimize.

    Parameters
    ----------
    param1 : float
        First parameter to optimize
    param2 : float
        Second parameter to optimize

    Returns
    -------
    tuple
        (capacity, voltage) arrays
    """
    try:
        # Create a simple SPM model
        model = pybamm.lithium_ion.SPM()

        # Start with default parameters
        param = pybamm.ParameterValues("Chen2020")

        # Update with our optimization parameters
        param.update(
            {
                "Negative electrode diffusivity [m2.s-1]": param1,
                "Positive electrode diffusivity [m2.s-1]": param2,
            },
            check_already_exists=False,
        )

        # Create experiment
        experiment = pybamm.Experiment([("Discharge at C/2 until 3.0 V",)])

        # Run simulation
        sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
        solution = sim.solve(initial_soc=1.0)

        # Extract data
        voltage = solution["Voltage [V]"].entries
        capacity = (
            solution["Discharge capacity [A.h]"].entries
            / solution["Discharge capacity [A.h]"].entries[-1]
        )

        return capacity, voltage

    except Exception as e:
        print(f"Simulation failed: {e}")
        return np.array([0, 1]), np.array([4.0, 3.0])


def objective_function(param1, param2):
    """
    Objective function for Bayesian optimization.
    Returns negative MSE (since BayesianOptimization maximizes).

    Parameters
    ----------
    param1 : float
        First parameter
    param2 : float
        Second parameter

    Returns
    -------
    float
        Negative mean squared error
    """
    # Get experimental data
    capacity_exp, voltage_exp = create_synthetic_data()

    # Run simulation
    capacity_sim, voltage_sim = run_simulation(param1, param2)

    # Calculate MSE
    if len(capacity_sim) > 1:
        voltage_sim_interp = np.interp(capacity_exp, capacity_sim, voltage_sim)
        mse = np.mean((voltage_exp - voltage_sim_interp) ** 2)
        return -mse  # Negative because we want to minimize MSE
    else:
        return -1000  # Penalty for failed simulations


def main():
    """Main optimization function."""
    print("Starting Bayesian optimization for PyBaMM parameters...")

    # Define parameter bounds
    pbounds = {
        "param1": (1e-15, 1e-12),  # Negative electrode diffusivity
        "param2": (1e-15, 1e-12),  # Positive electrode diffusivity
    }

    # Initialize optimizer
    optimizer = BayesianOptimization(
        f=objective_function, pbounds=pbounds, random_state=42, verbose=2
    )

    # Run optimization
    print("Running optimization...")
    optimizer.maximize(
        init_points=3,  # Random initial points
        n_iter=100,  # Optimization iterations
    )

    # Print results
    print("\nOptimization completed!")
    print(f"Best cost: {optimizer.max['target']:.6f}")
    print("Best parameters:")
    for param, value in optimizer.max["params"].items():
        print(f"  {param}: {value:.2e}")

    # Show final fit
    print("\nGenerating final fit plot...")
    capacity_exp, voltage_exp = create_synthetic_data()
    capacity_sim, voltage_sim = run_simulation(
        optimizer.max["params"]["param1"], optimizer.max["params"]["param2"]
    )

    # Simple plot
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.plot(capacity_exp, voltage_exp, "o-", label="Experimental", markersize=4)
        plt.plot(capacity_sim, voltage_sim, "r-", label="Best Fit", linewidth=2)
        plt.xlabel("Capacity [normalized]")
        plt.ylabel("Voltage [V]")
        plt.title("PyBaMM Parameter Optimization Results")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")

    print(f"\nFinal MSE: {-optimizer.max['target']:.6f}")


if __name__ == "__main__":
    main()
