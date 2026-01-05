"""
Example: Multi-Objective Optimization for MSMR Parameter Fitting

This script demonstrates how to perform multi-objective optimization to find
the best C-rate that balances discharge time and MAE (Mean Absolute Error).
It shows how to:
1. Test different C-rate values
2. Calculate both MAE and discharge time for each C-rate
3. Find Pareto-optimal solutions
4. Visualize the trade-off between objectives
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pybamm
from bayes_opt import BayesianOptimization

# Set up logging
pybamm.set_logging_level("WARNING")

# Global variables to store loaded data (loaded once)
EXPERIMENTAL_DATA: tuple = None  # type: ignore
TIME_DATA: np.ndarray = None  # type: ignore
CURRENT_DATA: np.ndarray = None  # type: ignore
VOLTAGE_DATA: np.ndarray = None  # type: ignore
SAMPLING_INFO: dict = None  # type: ignore

def load_experimental_data():
    """
    Load experimental data from test_02.csv file.
    """
    try:
        # Load the CSV file
        df = pd.read_csv('test_02.csv')
        
        # Extract capacity and voltage data
        capacity_experimental = df['Capacity(Ah)'].values
        voltage_experimental = df['Voltage(V)'].values
        
        # Normalize capacity to 0-1 range for comparison with simulation
        capacity_experimental = capacity_experimental / capacity_experimental.max()
        
        print(f"Loaded experimental data: {len(capacity_experimental)} points")
        print(f"Capacity range: {capacity_experimental.min():.6f} to {capacity_experimental.max():.6f}")
        print(f"Voltage range: {voltage_experimental.min():.6f} to {voltage_experimental.max():.6f}")
        
        return capacity_experimental, voltage_experimental
        
    except Exception as e:
        print(f"Error loading experimental data: {e}")
        # Fallback to synthetic data if file loading fails
        print("Falling back to synthetic data...")
        capacity_experimental = np.linspace(0, 1, 50)
        voltage_experimental = 4.2 - 1.2 * capacity_experimental + 0.3 * np.sin(2 * np.pi * capacity_experimental)
        voltage_experimental += 0.02 * np.random.normal(0, 1, len(voltage_experimental))
        return capacity_experimental, voltage_experimental

def load_all_data():
    """
    Load all experimental data once and store in global variables.
    This function should be called once at the beginning.
    """
    global EXPERIMENTAL_DATA, TIME_DATA, CURRENT_DATA, VOLTAGE_DATA, SAMPLING_INFO
    
    try:
        # Load the CSV file
        df = pd.read_csv('test_02.csv')
        
        # Extract all data
        time_data = df['TestTime(s)'].values
        current_data = df['Current(A)'].values
        voltage_data = df['Voltage(V)'].values
        capacity_data = df['Capacity(Ah)'].values
        
        # Normalize capacity to 0-1 range for comparison with simulation
        capacity_data = capacity_data / capacity_data.max()
        
        # Store in global variables
        TIME_DATA = time_data
        CURRENT_DATA = current_data
        VOLTAGE_DATA = voltage_data
        EXPERIMENTAL_DATA = (capacity_data, voltage_data)
        
        # Sample the data once
        SAMPLING_INFO = sample_current_data(time_data, current_data, voltage_data)
        
        print(f"Loaded all experimental data: {len(capacity_data)} points")
        print(f"Capacity range: {capacity_data.min():.6f} to {capacity_data.max():.6f}")
        print(f"Voltage range: {voltage_data.min():.6f} to {voltage_data.max():.6f}")
        print(f"Time range: {time_data[0]:.2f} to {time_data[-1]:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"Error loading experimental data: {e}")
        print("Falling back to synthetic data...")
        
        # Create synthetic data
        time_data = np.linspace(0, 3600, 100)
        current_data = np.ones_like(time_data) * 0.1  # Constant current
        voltage_data = 4.2 - 1.2 * (time_data / time_data[-1]) + 0.3 * np.sin(2 * np.pi * time_data / time_data[-1])
        voltage_data += 0.02 * np.random.normal(0, 1, len(voltage_data))
        capacity_data = time_data / time_data[-1]  # Normalized capacity
        
        # Store in global variables
        TIME_DATA = time_data
        CURRENT_DATA = current_data
        VOLTAGE_DATA = voltage_data
        EXPERIMENTAL_DATA = (capacity_data, voltage_data)
        
        # Sample the data once
        SAMPLING_INFO = sample_current_data(time_data, current_data, voltage_data)
        
        return False

def run_msmr_simulation(params_dict):
    """
    Run MSMR simulation with given parameters using pre-loaded data.
    
    Parameters
    ----------
    params_dict : dict
        Dictionary containing MSMR parameters to optimize
    
    Returns
    -------
    capacity_sim : array
        Simulated capacity values
    voltage_sim : array
        Simulated voltage values
    """
    global TIME_DATA, CURRENT_DATA, VOLTAGE_DATA, SAMPLING_INFO
    
    try:
        # Create MSMR model
        model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
        
        # Start with base parameters
        param = pybamm.ParameterValues("MSMR_Example")
        
        # Convert optimizer parameter names to actual MSMR parameter names
        msmr_params = convert_optimizer_params_to_msmr(params_dict)
        
        # Update with the converted parameters
        param.update(msmr_params, check_already_exists=False)

        experiment = pybamm.Experiment(
    [
        (
            "Discharge at C/7.5 until 3.1 V",
        ),
    ],
    )
        sim = pybamm.Simulation(model, experiment=experiment)
        solution = sim.solve()
        # solution = sim.solution
        voltage_sim = solution["Voltage [V]"].entries
        capacity_sim = solution["Discharge capacity [A.h]"].entries
        time_sim = solution["Time [s]"].entries

        
        # Normalize capacity to 0-1 range
        if len(capacity_sim) > 0 and capacity_sim.max() > 0:
            capacity_sim = capacity_sim / capacity_sim.max()
        else:
            # Return dummy data that spans the experimental range
            capacity_sim = np.linspace(0, 1, len(voltage_sim))
        
        # Check for valid simulation results
        if len(voltage_sim) == 0 or len(capacity_sim) == 0 or len(time_sim) == 0:
            raise ValueError("Empty simulation results")
        
        if np.any(np.isnan(voltage_sim)) or np.any(np.isnan(capacity_sim)) or np.any(np.isnan(time_sim)):
            raise ValueError("NaN values in simulation results")
        
        if np.any(np.isinf(voltage_sim)) or np.any(np.isinf(capacity_sim)) or np.any(np.isinf(time_sim)):
            raise ValueError("Infinite values in simulation results")
        
        return capacity_sim, voltage_sim, time_sim
        
    except Exception as e:
        # Return dummy data if simulation fails
        print(f"Simulation failed: {e}")
        return np.array([0, 1]), np.array([4.2, 3.0]), np.array([0, 1])

def debug_cost_calculation(params_dict):
    """
    Debug function to understand the cost calculation and identify issues.
    """
    global EXPERIMENTAL_DATA
    
    # Get experimental data from global variable
    capacity_exp, voltage_exp = EXPERIMENTAL_DATA
    
    print(f"Experimental data: {len(capacity_exp)} points")
    print(f"Experimental capacity range: {capacity_exp.min():.6f} to {capacity_exp.max():.6f}")
    print(f"Experimental voltage range: {voltage_exp.min():.6f} to {voltage_exp.max():.6f}")
    
    # Run simulation with current parameters
    capacity_sim, voltage_sim, _ = run_msmr_simulation(params_dict)
    
    print(f"Simulation data: {len(capacity_sim)} points")
    print(f"Simulation capacity range: {capacity_sim.min():.6f} to {capacity_sim.max():.6f}")
    print(f"Simulation voltage range: {voltage_sim.min():.6f} to {voltage_sim.max():.6f}")
    
    # Check if interpolation is possible
    if len(capacity_sim) > 1:
        # Check if experimental capacity range is within simulation range
        exp_min_in_sim = capacity_exp.min() >= capacity_sim.min()
        exp_max_in_sim = capacity_exp.max() <= capacity_sim.max()
        
        print(f"Experimental min within simulation range: {exp_min_in_sim}")
        print(f"Experimental max within simulation range: {exp_max_in_sim}")
        
        if exp_min_in_sim and exp_max_in_sim:
            # Interpolate simulation data to match experimental data points
            voltage_sim_interp = np.interp(capacity_exp, capacity_sim, voltage_sim)
            
            # Calculate mean squared error
            mse = np.mean((voltage_exp - voltage_sim_interp) ** 2)
            
            print(f"Interpolated voltage range: {voltage_sim_interp.min():.6f} to {voltage_sim_interp.max():.6f}")
            print(f"Mean squared error: {mse:.6f}")
            
            # Show some sample comparisons
            print("\nSample comparisons (first 10 points):")
            for i in range(min(10, len(capacity_exp))):
                print(f"  Point {i}: Exp={voltage_exp[i]:.4f}V, Sim={voltage_sim_interp[i]:.4f}V, Diff={abs(voltage_exp[i]-voltage_sim_interp[i]):.4f}V")
            
            return mse
        else:
            print("WARNING: Experimental capacity range outside simulation range!")
            print("This will cause interpolation to fail or produce incorrect results.")
            return 1000  # Large penalty
    else:
        print("Simulation failed - insufficient data points")
        return 1000

def calculate_cost(**params_dict):
    """
    Calculate cost function comparing simulation to experimental data.
    Lower cost = better fit.
    
    Parameters
    ----------
    **params_dict : dict
        Dictionary containing parameters to evaluate
    
    Returns
    -------
    float
        Cost value (negative for maximization in BayesianOptimization)
    """
    global EXPERIMENTAL_DATA
    
    # Get experimental data from global variable
    capacity_exp, voltage_exp = EXPERIMENTAL_DATA
    
    # Run simulation with current parameters
    capacity_sim, voltage_sim, _ = run_msmr_simulation(params_dict)
    
    # Interpolate simulation data to match experimental data points
    if len(capacity_sim) > 1:
        # Check if experimental capacity range is within simulation range
        exp_min_in_sim = capacity_exp.min() >= capacity_sim.min()
        exp_max_in_sim = capacity_exp.max() <= capacity_sim.max()
        
        if exp_min_in_sim and exp_max_in_sim:
            voltage_sim_interp = np.interp(capacity_exp, capacity_sim, voltage_sim)
            
            # Calculate mean squared error
            mse = np.mean((voltage_exp - voltage_sim_interp) ** 2)
            
            # Return negative MSE (BayesianOptimization maximizes, so we minimize MSE)
            return -mse
        else:
            # Penalty for out-of-range simulations
            return -1000
    else:
        return -1000  # Large penalty for failed simulations

def sample_current_data(time_data, current_data, voltage_data, max_points=100):
    """
    Sample current data intelligently based on variation.
    
    Parameters
    ----------
    time_data : array
        Time points from experimental data
    current_data : array
        Current values from experimental data
    voltage_data : array
        Voltage values from experimental data
    max_points : int
        Maximum number of points to use for sampling
        
    Returns
    -------
    dict
        Dictionary containing sampling information and data
    """
    # Check current variation
    current_mean = np.mean(current_data)
    current_std = np.std(current_data)
    current_cv = current_std / current_mean  # Coefficient of variation
    
    print(f"Current statistics: mean={current_mean:.6f}A, std={current_std:.6f}A, CV={current_cv:.3f}")
    
    # Always use constant current for simulation
    print("Using constant current for simulation")
    time_sampled = np.array([time_data[0], time_data[-1]])  # Just start and end points
    current_sampled = np.array([current_mean, current_mean])
    voltage_sampled = np.array([voltage_data[0], voltage_data[-1]])
    use_constant_current = True
    
    return {
        'time_sampled': time_sampled,
        'current_sampled': current_sampled,
        'voltage_sampled': voltage_sampled,
        'current_mean': current_mean,
        'current_std': current_std,
        'current_cv': current_cv,
        'use_constant_current': use_constant_current,
        'original_points': len(time_data),
        'sampled_points': len(time_sampled)
    }

def plot_sampled_data():
    """
    Plot the sampled current data to verify sampling is working correctly.
    """
    print("Loading and sampling current data...")
    
    # Load current data from CSV file
    df = pd.read_csv('test_02.csv')
    time_data = df['TestTime(s)'].values
    current_data = df['Current(A)'].values
    voltage_data = df['Voltage(V)'].values
    
    # Sample the data using the new function
    sampling_info = sample_current_data(time_data, current_data, voltage_data)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Original vs Sampled Current
    ax1.plot(time_data, current_data, 'b-', label='Full Experimental Data', linewidth=1, alpha=0.7)
    
    if sampling_info['use_constant_current']:
        ax1.axhline(y=sampling_info['current_mean'], color='r', linestyle='--', linewidth=2, 
                   label=f'Constant Current ({sampling_info["current_mean"]:.4f}A)')
        # Show sample points for reference
        sample_indices = np.linspace(0, len(time_data)-1, 100, dtype=int)
        ax1.plot(time_data[sample_indices], current_data[sample_indices], 'ro', markersize=4, label='Sample Points (100)')
    else:
        ax1.plot(sampling_info['time_sampled'], sampling_info['current_sampled'], 'ro', 
                label=f'Sampled Data ({sampling_info["sampled_points"]} points)', markersize=4)
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Current [A]')
    ax1.set_title('Current Data: Original vs Sampled')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Voltage vs Time
    ax2.plot(time_data, voltage_data, 'g-', label='Full Experimental Voltage', linewidth=1)
    
    # Show sample points for voltage too
    if sampling_info['use_constant_current']:
        sample_indices = np.linspace(0, len(time_data)-1, 100, dtype=int)
    else:
        if len(time_data) > 100:
            sample_indices = np.linspace(0, len(time_data)-1, 100, dtype=int)
        else:
            sample_indices = np.arange(len(time_data))
    
    ax2.plot(time_data[sample_indices], voltage_data[sample_indices], 'go', markersize=4, 
             label=f'Sample Points ({len(sample_indices)})')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Voltage [V]')
    ax2.set_title('Experimental Voltage vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Data Points: {sampling_info["original_points"]}\n'
    stats_text += f'Time Range: {time_data[0]:.1f}s to {time_data[-1]:.1f}s\n'
    stats_text += f'Current Mean: {sampling_info["current_mean"]:.4f}A\n'
    stats_text += f'Current Std: {sampling_info["current_std"]:.4f}A\n'
    stats_text += f'Current CV: {sampling_info["current_cv"]:.3f}\n'
    stats_text += f'Voltage Range: {voltage_data.min():.3f}V to {voltage_data.max():.3f}V'
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return sampling_info

def optimize_msmr_parameters():
    """
    Main function to optimize MSMR parameters using Bayesian optimization.
    """
    print("Starting MSMR parameter optimization...")
    
    # Define the parameter bounds for optimization
    # These are all MSMR parameters for 6 negative electrode reactions and 4 positive electrode reactions
    pbounds = {
        # Negative electrode parameters (reaction 0)
        'neg_ocp_0': (0.1, 0.3),      # Standard potential [V]
        'neg_ideality_0': (0.5, 2.0), # Ideality factor
        'neg_j0_0': (1e-6, 1e-3),     # Exchange current density [A/m²]
        
        # Negative electrode parameters (reaction 1)
        'neg_ocp_1': (0.2, 0.4),
        'neg_ideality_1': (0.5, 2.0),
        'neg_j0_1': (1e-6, 1e-3),
        
        # Negative electrode parameters (reaction 2)
        'neg_ocp_2': (0.3, 0.5),
        'neg_ideality_2': (0.5, 2.0),
        'neg_j0_2': (1e-6, 1e-3),
        
        # Negative electrode parameters (reaction 3)
        'neg_ocp_3': (0.1, 0.4),
        'neg_ideality_3': (0.5, 2.0),
        'neg_j0_3': (1e-6, 1e-3),
        
        # Negative electrode parameters (reaction 4)
        'neg_ocp_4': (0.2, 0.5),
        'neg_ideality_4': (0.5, 2.0),
        'neg_j0_4': (1e-6, 1e-3),
        
        # Negative electrode parameters (reaction 5)
        'neg_ocp_5': (0.1, 0.5),
        'neg_ideality_5': (0.5, 2.0),
        'neg_j0_5': (1e-6, 1e-3),
        
        # Positive electrode parameters (reaction 0)
        'pos_ocp_0': (3.8, 4.0),
        'pos_ideality_0': (0.5, 2.0),
        'pos_j0_0': (1e-6, 1e-3),
        
        # Positive electrode parameters (reaction 1)
        'pos_ocp_1': (3.9, 4.1),
        'pos_ideality_1': (0.5, 2.0),
        'pos_j0_1': (1e-6, 1e-3),
        
        # Positive electrode parameters (reaction 2)
        'pos_ocp_2': (3.8, 4.1),
        'pos_ideality_2': (0.5, 2.0),
        'pos_j0_2': (1e-6, 1e-3),
        
        # Positive electrode parameters (reaction 3)
        'pos_ocp_3': (3.9, 4.2),
        'pos_ideality_3': (0.5, 2.0),
        'pos_j0_3': (1e-6, 1e-3),
    }
    
    # Initialize Bayesian optimization
    optimizer = pybamm.BayesianOptimization(
        f=calculate_cost,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    # Note: Logger removed for compatibility with this version of BayesianOptimization
    
    # Run optimization
    print("Running optimization...")
    optimizer.maximize(
        init_points=10,    # Number of random initial points
        n_iter=20,       # Number of optimization iterations
    )
    
    # Print results
    print("\nOptimization completed!")
    print(f"Best cost: {optimizer.max['target']}")
    print("Best parameters:")
    for param, value in optimizer.max['params'].items():
        print(f"  {param}: {value:.6f}")
    
    return optimizer

def plot_results(optimizer):
    """
    Plot the optimization results and compare best fit to experimental data.
    """
    global EXPERIMENTAL_DATA, TIME_DATA, CURRENT_DATA, VOLTAGE_DATA, SAMPLING_INFO
    
    # Get experimental data
    capacity_exp, voltage_exp = EXPERIMENTAL_DATA
    
    # Load time data for plotting
    time_data = TIME_DATA
    voltage_exp_time = VOLTAGE_DATA
    
    # Run simulation with best parameters
    best_params = convert_optimizer_params_to_msmr(optimizer.max['params'])
    # Get the full solution object for time
    try:
        model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
        param = pybamm.ParameterValues("MSMR_Example")
        param.update(best_params, check_already_exists=False)
        
        # Use experiment-based simulation like the single comparison script
        experiment = pybamm.Experiment([
            ("Discharge at C/9 until 3 V",),
        ])
        sim = pybamm.Simulation(model, experiment=experiment)
        solution = sim.solve()
        voltage_sim = solution["Voltage [V]"].entries
        time_sim = solution["Time [s]"].entries
    except Exception as e:
        print(f"Simulation failed: {e}")
        voltage_sim = np.array([4.2, 3.0])
        time_sim = np.array([time_data[0], time_data[-1]])
    
    # Create plots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Time vs Voltage (Experimental vs Simulated)
    ax1.plot(time_data, voltage_exp_time, 'ko-', label='Experimental', linewidth=2, markersize=4)
    
    # For simulated data, use the actual simulation time points
    if len(voltage_sim) > 0:
        ax1.plot(time_sim, voltage_sim, 'r-', label='Best MSMR Fit', linewidth=2)
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Voltage [V]')
    ax1.set_title('Time vs Voltage: Experimental vs MSMR Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Current vs Time (Input data)
    current_data = CURRENT_DATA
    ax2.plot(time_data, current_data, 'b-', linewidth=2)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Current [A]')
    ax2.set_title('Input Current vs Time')
    ax2.grid(True, alpha=0.3)
    
    # Add current statistics as text
    current_mean = np.mean(current_data)
    current_std = np.std(current_data)
    current_cv = current_std / current_mean
    ax2.text(0.05, 0.95, f'Mean: {current_mean:.4f}A\nStd: {current_std:.4f}A\nCV: {current_cv:.3f}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 3: Optimization convergence
    costs = optimizer.space.target
    ax3.plot(costs, 'b-', linewidth=2)
    ax3.set_xlabel('Optimization Step')
    ax3.set_ylabel('Cost (negative MSE)')
    ax3.set_title('Optimization Convergence')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Parameter evolution
    param_names = list(optimizer.max['params'].keys())
    param_order = optimizer.space._keys
    for idx, param_name in enumerate(param_names):
        param_values = [optimizer.space.params[i][idx] for i in range(len(optimizer.space))]
        ax4.plot(param_values, label=param_name, alpha=0.7)
    
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Parameter Value')
    ax4.set_title('Parameter Evolution')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Cost distribution histogram
    ax5.hist(costs, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.set_xlabel('Cost (Negative MSE)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Cost Distribution')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Best vs worst solutions
    plot_best_vs_worst(optimizer, (capacity_exp, voltage_exp), ax6)
    
    plt.tight_layout()
    plt.show()
    
    # Print final cost
    print(f"\nFinal cost (negative MSE): {optimizer.max['target']:.6f}")
    print(f"Final MSE: {-optimizer.max['target']:.6f}")

def plot_solution_distribution(optimizer, experimental_data, best_params, ax):
    """
    Plot the distribution of all solutions evaluated during optimization.
    
    Parameters
    ----------
    optimizer : BayesianOptimization
        The optimizer object containing all results
    experimental_data : tuple
        (capacity, voltage) experimental data
    best_params : dict
        Best parameters found
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    global EXPERIMENTAL_DATA, TIME_DATA, CURRENT_DATA, VOLTAGE_DATA, SAMPLING_INFO
    
    # Get experimental data from global variable
    capacity_exp, voltage_exp = EXPERIMENTAL_DATA
    
    # Plot experimental data
    ax.plot(capacity_exp, voltage_exp, 'ko-', label='Experimental', linewidth=3, markersize=8)
    
    # Plot all simulated solutions with transparency
    successful_count = 0
    for i in range(len(optimizer.space)):
        try:
            param_set = optimizer.space.params[i]
            msmr_params = convert_optimizer_params_to_msmr(param_set)
            capacity_sim, voltage_sim, _ = run_msmr_simulation(msmr_params)
            alpha = 0.1 if i != len(optimizer.space) - 1 else 0.8  # Make best solution more visible
            color = 'red' if i == len(optimizer.space) - 1 else 'blue'  # Best solution in red
            linewidth = 2 if i == len(optimizer.space) - 1 else 0.5
            ax.plot(capacity_sim, voltage_sim, color=color, alpha=alpha, linewidth=linewidth)
            successful_count += 1
        except:
            # Skip failed simulations
            continue
    
    ax.set_xlabel('Capacity [normalized]')
    ax.set_ylabel('Voltage [V]')
    ax.set_title('Distribution of All Solutions\n(Blue: All solutions, Red: Best solution)')
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    total_sims = len(optimizer.space)
    ax.text(0.05, 0.95, f'Successful simulations: {successful_count}/{total_sims}', 
             transform=ax.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def plot_best_vs_worst(optimizer, experimental_data, ax):
    """
    Plot comparison between best and worst solutions.
    
    Parameters
    ----------
    optimizer : BayesianOptimization
        The optimizer object containing all results
    experimental_data : tuple
        (capacity, voltage) experimental data
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    global EXPERIMENTAL_DATA, TIME_DATA, CURRENT_DATA, VOLTAGE_DATA, SAMPLING_INFO
    
    # Get experimental data from global variable
    capacity_exp, voltage_exp = EXPERIMENTAL_DATA
    costs = optimizer.space.target
    
    # Find best and worst solutions
    best_idx = np.argmax(costs)
    worst_idx = np.argmin(costs)
    
    # Plot experimental data
    ax.plot(capacity_exp, voltage_exp, 'ko-', label='Experimental', linewidth=2)
    
    # Plot best solution
    try:
        param_set = optimizer.space.params[best_idx]
        msmr_params = convert_optimizer_params_to_msmr(param_set)
        capacity_best, voltage_best, _ = run_msmr_simulation(msmr_params)
        ax.plot(capacity_best, voltage_best, 'g-', label=f'Best (MSE: {-costs[best_idx]:.4f})', linewidth=2)
    except:
        pass
    
    # Plot worst solution
    try:
        param_set = optimizer.space.params[worst_idx]
        msmr_params = convert_optimizer_params_to_msmr(param_set)
        capacity_worst, voltage_worst, _ = run_msmr_simulation(msmr_params)
        ax.plot(capacity_worst, voltage_worst, 'r--', label=f'Worst (MSE: {-costs[worst_idx]:.4f})', linewidth=2)
    except:
        pass
    
    ax.set_xlabel('Capacity [normalized]')
    ax.set_ylabel('Voltage [V]')
    ax.set_title('Best vs Worst Solutions')
    ax.legend()
    ax.grid(True, alpha=0.3)

def convert_optimizer_params_to_msmr(optimizer_params):
    """
    Convert optimizer parameter names (e.g., 'neg_ocp_0') to MSMR parameter names
    (e.g., 'Negative electrode host site standard potential (0) [V]').
    Handles all reactions and all parameter types present in optimizer_params.
    """
    msmr_params = {}
    # Define mapping from short names to MSMR parameter name fragments
    param_map = {
        "ocp": "host site standard potential ({i}) [V]",
        "ideality": "host site ideality factor ({i})",
        "j0": "host site reference exchange-current density ({i}) [A.m-2]",
        "alpha": "host site charge transfer coefficient ({i})",
        "theta": "host site occupancy fraction ({i})",
    }
    for domain in ["neg", "pos"]:
        Electrode = "Negative" if domain == "neg" else "Positive"
        for key in optimizer_params:
            if key.startswith(domain + "_"):
                # key format: domain_paramtype_index, e.g., neg_ocp_0
                parts = key.split("_")
                if len(parts) == 3:
                    _, paramtype, idx = parts
                    if paramtype in param_map:
                        msmr_name = f"{Electrode} electrode {param_map[paramtype].format(i=idx)}"
                        msmr_params[msmr_name] = optimizer_params[key]
    return msmr_params

def calculate_mae_mse(exp_data, sim_data, exp_time, sim_time):
    """
    Calculate MAE and MSE between experimental and simulated data.
    
    Parameters
    ----------
    exp_data : array
        Experimental voltage data
    sim_data : array
        Simulated voltage data
    exp_time : array
        Experimental time data
    sim_time : array
        Simulated time data
    
    Returns
    -------
    tuple
        (mae, mse, sim_data_interp) - Mean Absolute Error, Mean Squared Error, and interpolated simulation data
    """
    # Interpolate simulation data to match experimental time points
    if len(sim_data) > 1 and len(sim_time) > 1:
        # Create interpolation function
        sim_interp = np.interp(exp_time, sim_time, sim_data)
        
        # Calculate errors
        mae = np.mean(np.abs(exp_data - sim_interp))
        mse = np.mean((exp_data - sim_interp) ** 2)
        
        return mae, mse, sim_interp
    else:
        return float('inf'), float('inf'), sim_data

def plot_experiment_vs_model():
    """
    Plot experimental data vs MSMR model output (voltage and current vs time)
    using the same style as the sampling plot.
    """
    print("Running MSMR simulation with default parameters...")
    global EXPERIMENTAL_DATA, TIME_DATA, CURRENT_DATA, VOLTAGE_DATA, SAMPLING_INFO
    
    # Use pre-loaded data
    time_data = TIME_DATA
    current_data = CURRENT_DATA
    voltage_data = VOLTAGE_DATA
    sampling_info = SAMPLING_INFO

    # Run MSMR simulation with default parameters
    default_params = {}  # Use empty dict to use base parameters
    # Get the full solution object for time
    try:
        model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
        param = pybamm.ParameterValues("MSMR_Example")
        
        # Use experiment-based simulation like the single comparison script
        experiment = pybamm.Experiment([
            ("Discharge at C/9 until 3 V",),
        ])
        sim = pybamm.Simulation(model, experiment=experiment)
        solution = sim.solve()
        voltage_sim = solution["Voltage [V]"].entries
        sim_time = solution["Time [s]"].entries
    except Exception as e:
        print(f"Simulation failed: {e}")
        voltage_sim = np.array([4.2, 3.0])
        sim_time = np.array([time_data[0], time_data[-1]])

    # Calculate MAE and MSE
    mae, mse, voltage_sim_interp = calculate_mae_mse(voltage_data, voltage_sim, time_data, sim_time)
    
    print(f"\n📊 Error Analysis:")
    print(f"  MAE (Mean Absolute Error): {mae:.6f} V")
    print(f"  MSE (Mean Squared Error): {mse:.6f} V²")
    print(f"  RMSE (Root Mean Square Error): {np.sqrt(mse):.6f} V")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Voltage vs Time
    ax1.plot(time_data, voltage_data, 'g-', label='Experimental Voltage', linewidth=1)
    # Show sample points for voltage
    if sampling_info['use_constant_current']:
        sample_indices = np.linspace(0, len(time_data)-1, 100, dtype=int)
    else:
        if len(time_data) > 100:
            sample_indices = np.linspace(0, len(time_data)-1, 100, dtype=int)
        else:
            sample_indices = np.arange(len(time_data))
    ax1.plot(time_data[sample_indices], voltage_data[sample_indices], 'go', markersize=4, label=f'Experimental Sample Points ({len(sample_indices)})')
    # Plot MSMR model output
    ax1.plot(sim_time, voltage_sim, 'r-', label='MSMR Model', linewidth=2)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Voltage [V]')
    ax1.set_title('Voltage vs Time: Experiment vs MSMR Model')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add error metrics to the plot
    error_text = f'MAE: {mae:.4f} V\nMSE: {mse:.4f} V²\nRMSE: {np.sqrt(mse):.4f} V'
    ax1.text(0.02, 0.98, error_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Plot 2: Current vs Time
    ax2.plot(time_data, current_data, 'b-', label='Experimental Current', linewidth=1, alpha=0.7)
    if sampling_info['use_constant_current']:
        ax2.axhline(y=sampling_info['current_mean'], color='r', linestyle='--', linewidth=2, label=f'Constant Current ({sampling_info["current_mean"]:.4f}A)')
        ax2.plot(time_data[sample_indices], current_data[sample_indices], 'ro', markersize=4, label='Sample Points (100)')
    else:
        ax2.plot(sampling_info['time_sampled'], sampling_info['current_sampled'], 'ro', label=f'Sampled Data ({sampling_info["sampled_points"]} points)', markersize=4)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Current [A]')
    ax2.set_title('Current vs Time: Experiment')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    return mae, mse

def optimize_c_rate():
    """
    Optimize the C-rate value to minimize error between experimental and simulated data.
    """
    print("Optimizing C-rate value to minimize error...")
    global EXPERIMENTAL_DATA, TIME_DATA, CURRENT_DATA, VOLTAGE_DATA, SAMPLING_INFO
    
    # Use pre-loaded data
    time_data = TIME_DATA
    current_data = CURRENT_DATA
    voltage_data = VOLTAGE_DATA
    sampling_info = SAMPLING_INFO
    
    # Define C-rate range to test
    c_rates = np.linspace(1/20, 1/5, 20)  # From C/20 to C/5
    errors = []
    best_c_rate = None
    best_error = float('inf')
    best_mae = None
    best_mse = None
    
    print(f"Testing {len(c_rates)} C-rate values from C/20 to C/5...")
    
    for i, c_rate in enumerate(c_rates):
        try:
            # Create experiment with current C-rate
            experiment = pybamm.Experiment([
                (f"Discharge at C/{1/c_rate:.1f} until 3 V",),
            ])
            
            # Run simulation
            model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
            param = pybamm.ParameterValues("MSMR_Example")
            sim = pybamm.Simulation(model, experiment=experiment)
            solution = sim.solve()
            voltage_sim = solution["Voltage [V]"].entries
            sim_time = solution["Time [s]"].entries
            
            # Calculate errors
            mae, mse, _ = calculate_mae_mse(voltage_data, voltage_sim, time_data, sim_time)
            
            # Use MAE as the optimization metric
            error = mae
            errors.append(error)
            
            print(f"  C/{1/c_rate:.1f}: MAE = {mae:.6f} V, MSE = {mse:.6f} V²")
            
            # Track best result
            if error < best_error:
                best_error = error
                best_c_rate = c_rate
                best_mae = mae
                best_mse = mse
                
        except Exception as e:
            print(f"  C/{1/c_rate:.1f}: Failed - {e}")
            errors.append(float('inf'))
    
    print(f"\n🎯 Optimization Results:")
    print(f"  Best C-rate: C/{1/best_c_rate:.1f}")
    print(f"  Best MAE: {best_mae:.6f} V")
    print(f"  Best MSE: {best_mse:.6f} V²")
    print(f"  Best RMSE: {np.sqrt(best_mse):.6f} V")
    
    # Plot optimization results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Error vs C-rate
    valid_indices = [i for i, err in enumerate(errors) if err != float('inf')]
    valid_c_rates = [c_rates[i] for i in valid_indices]
    valid_errors = [errors[i] for i in valid_indices]
    
    ax1.plot(valid_c_rates, valid_errors, 'bo-', linewidth=2, markersize=6)
    ax1.axvline(x=best_c_rate, color='red', linestyle='--', linewidth=2, label=f'Best: C/{1/best_c_rate:.1f}')
    ax1.set_xlabel('C-rate')
    ax1.set_ylabel('MAE [V]')
    ax1.set_title('Error vs C-rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Best fit comparison
    # Run simulation with best C-rate
    best_experiment = pybamm.Experiment([
        (f"Discharge at C/{1/best_c_rate:.1f} until 3 V",),
    ])
    model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
    param = pybamm.ParameterValues("MSMR_Example")
    sim = pybamm.Simulation(model, experiment=best_experiment)
    solution = sim.solve()
    voltage_sim_best = solution["Voltage [V]"].entries
    sim_time_best = solution["Time [s]"].entries
    
    ax2.plot(time_data, voltage_data, 'g-', label='Experimental', linewidth=1)
    ax2.plot(sim_time_best, voltage_sim_best, 'r-', label=f'MSMR Model (C/{1/best_c_rate:.1f})', linewidth=2)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Voltage [V]')
    ax2.set_title(f'Best Fit: C/{1/best_c_rate:.1f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add error metrics to the plot
    error_text = f'MAE: {best_mae:.4f} V\nMSE: {best_mse:.4f} V²\nRMSE: {np.sqrt(best_mse):.4f} V'
    ax2.text(0.02, 0.98, error_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return best_c_rate, best_mae, best_mse

def multi_objective_optimization():
    """
    Perform multi-objective optimization considering both discharge time error (|T_exp - T_sim|) and MAE.
    """
    print("Performing multi-objective optimization (discharge time error vs MAE)...")
    global EXPERIMENTAL_DATA, TIME_DATA, CURRENT_DATA, VOLTAGE_DATA, SAMPLING_INFO
    
    # Use pre-loaded data
    time_data = TIME_DATA
    current_data = CURRENT_DATA
    voltage_data = VOLTAGE_DATA
    sampling_info = SAMPLING_INFO
    
    # Calculate experimental discharge time (when voltage first drops below 3V)
    exp_discharge_time = time_data[-1]
    for t, v in zip(time_data, voltage_data):
        if v < 3.0:
            exp_discharge_time = t
            break
    print(f"Experimental discharge time: {exp_discharge_time:.1f} s")
    
    # Define C-rate range to test
    c_rates = np.linspace(1/20, 1/5, 100)  # From C/20 to C/5, more points for better resolution
    results = []
    
    print(f"Testing {len(c_rates)} C-rate values from C/20 to C/5...")
    
    for i, c_rate in enumerate(c_rates):
        try:
            # Create experiment with current C-rate
            experiment = pybamm.Experiment([
                (f"Discharge at C/{1/c_rate:.1f} until 3 V",),
            ])
            
            # Run simulation
            model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
            param = pybamm.ParameterValues("MSMR_Example")

            # for domain in ["negative", "positive"]:
            #     Electrode = domain.capitalize()
            #     # Loop over reactions
            #     N = int(param["Number of reactions in " + domain + " electrode"])
            #     for i in range(N):
            #         names = [
            #     f"{Electrode} electrode host site occupancy fraction ({i})",
            #     f"{Electrode} electrode host site standard potential ({i}) [V]",
            #     f"{Electrode} electrode host site ideality factor ({i})",
            #     f"{Electrode} electrode host site charge transfer coefficient ({i})",
            #     f"{Electrode} electrode host site reference exchange-current density ({i}) [A.m-2]",
            # ]
            # for name in names:
            #     print(f"{name} = {param[name]}")            

            sim = pybamm.Simulation(model, experiment=experiment)
            solution = sim.solve()
            voltage_sim = solution["Voltage [V]"].entries
            sim_time = solution["Time [s]"].entries
            
            # Calculate errors
            mae, mse, _ = calculate_mae_mse(voltage_data, voltage_sim, time_data, sim_time)
            
            # Get simulated discharge time (when voltage first drops below 3V)
            sim_discharge_time = sim_time[-1]
            for t, v in zip(sim_time, voltage_sim):
                if v < 3.0:
                    sim_discharge_time = t
                    break
            
            # Calculate discharge time error
            time_error = abs(exp_discharge_time - sim_discharge_time)
            
            results.append({
                'c_rate': c_rate,
                'mae': mae,
                'mse': mse,
                'discharge_time': sim_discharge_time,
                'time_error': time_error,
                'voltage_sim': voltage_sim,
                'sim_time': sim_time
            })
            
            print(f"  C/{1/c_rate:.1f}: MAE = {mae:.6f} V, |ΔT| = {time_error:.1f} s")
                
        except Exception as e:
            print(f"  C/{1/c_rate:.1f}: Failed - {e}")
            results.append({
                'c_rate': c_rate,
                'mae': float('inf'),
                'mse': float('inf'),
                'discharge_time': float('inf'),
                'time_error': float('inf'),
                'voltage_sim': np.array([4.2, 3.0]),
                'sim_time': np.array([0, 1])
            })
    
    # Filter out failed simulations
    valid_results = [r for r in results if r['mae'] != float('inf')]
    
    if not valid_results:
        print("No valid simulations found!")
        return None, None, None
    
    # Normalize objectives for Pareto analysis
    maes = [r['mae'] for r in valid_results]
    time_errors = [r['time_error'] for r in valid_results]
    
    mae_min, mae_max = min(maes), max(maes)
    terr_min, terr_max = min(time_errors), max(time_errors)
    
    # Normalize to 0-1 range (lower is better for both)
    normalized_maes = [(mae - mae_min) / (mae_max - mae_min) for mae in maes]
    normalized_terrs = [(terr - terr_min) / (terr_max - terr_min) for terr in time_errors]
    
    # Calculate combined objective (weighted sum)
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Different weight combinations
    pareto_solutions = []
    
    for weight in weights:
        combined_objectives = [weight * nm + (1 - weight) * nt 
                             for nm, nt in zip(normalized_maes, normalized_terrs)]
        best_idx = np.argmin(combined_objectives)
        pareto_solutions.append(valid_results[best_idx])
    
    # Find Pareto front (non-dominated solutions)
    pareto_front = []
    for i, result in enumerate(valid_results):
        dominated = False
        for j, other in enumerate(valid_results):
            if i != j:
                if (other['mae'] <= result['mae'] and other['time_error'] <= result['time_error'] and
                    (other['mae'] < result['mae'] or other['time_error'] < result['time_error'])):
                    dominated = True
                    break
        if not dominated:
            pareto_front.append(result)
    
    # Sort Pareto front by MAE
    pareto_front.sort(key=lambda x: x['mae'])
    
    print(f"\n🎯 Multi-Objective Optimization Results:")
    print(f"  Total valid solutions: {len(valid_results)}")
    print(f"  Pareto front solutions: {len(pareto_front)}")
    
    # Show Pareto front solutions
    print(f"\n📊 Pareto Front Solutions:")
    for i, solution in enumerate(pareto_front):
        print(f"  {i+1}. C/{1/solution['c_rate']:.1f}: MAE = {solution['mae']:.6f} V, |ΔT| = {solution['time_error']:.1f} s")
    
    # Find best solutions for different criteria
    best_mae_solution = min(valid_results, key=lambda x: x['mae'])
    best_time_solution = min(valid_results, key=lambda x: x['time_error'])
    
    print(f"\n🏆 Best Solutions:")
    print(f"  Best MAE: C/{1/best_mae_solution['c_rate']:.1f} (MAE = {best_mae_solution['mae']:.6f} V, |ΔT| = {best_mae_solution['time_error']:.1f} s)")
    print(f"  Best |ΔT|: C/{1/best_time_solution['c_rate']:.1f} (MAE = {best_time_solution['mae']:.6f} V, |ΔT| = {best_time_solution['time_error']:.1f} s)")
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: MAE vs |ΔT| (Pareto plot)
    maes_plot = [r['mae'] for r in valid_results]
    terrs_plot = [r['time_error'] for r in valid_results]
    c_rates_plot = [r['c_rate'] for r in valid_results]
    
    scatter = ax1.scatter(terrs_plot, maes_plot, c=c_rates_plot, cmap='viridis', s=50, alpha=0.7)
    ax1.scatter([r['time_error'] for r in pareto_front], 
                [r['mae'] for r in pareto_front], 
                c='red', s=100, marker='*', label='Pareto Front', zorder=5)
    ax1.scatter(best_mae_solution['time_error'], best_mae_solution['mae'], 
                c='green', s=200, marker='o', label='Best MAE', zorder=5)
    ax1.scatter(best_time_solution['time_error'], best_time_solution['mae'], 
                c='blue', s=200, marker='s', label='Best |ΔT|', zorder=5)
    
    ax1.set_xlabel('|ΔT| = |T_exp - T_sim| [s]')
    ax1.set_ylabel('MAE [V]')
    ax1.set_title('Multi-Objective Optimization: MAE vs |ΔT|')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='C-rate')
    
    # Plot 2: C-rate vs MAE
    ax2.plot([1/r['c_rate'] for r in valid_results], [r['mae'] for r in valid_results], 'bo-', linewidth=2)
    ax2.axvline(x=1/best_mae_solution['c_rate'], color='green', linestyle='--', linewidth=2, label=f'Best MAE: C/{1/best_mae_solution["c_rate"]:.1f}')
    ax2.set_xlabel('C-rate (1/C)')
    ax2.set_ylabel('MAE [V]')
    ax2.set_title('MAE vs C-rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: C-rate vs |ΔT|
    ax3.plot([1/r['c_rate'] for r in valid_results], [r['time_error'] for r in valid_results], 'ro-', linewidth=2)
    ax3.axvline(x=1/best_time_solution['c_rate'], color='blue', linestyle='--', linewidth=2, label=f'Best |ΔT|: C/{1/best_time_solution["c_rate"]:.1f}')
    ax3.set_xlabel('C-rate (1/C)')
    ax3.set_ylabel('|ΔT| [s]')
    ax3.set_title('|ΔT| vs C-rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Best solutions comparison
    # Show experimental data
    ax4.plot(time_data, voltage_data, 'k-', label='Experimental', linewidth=1, alpha=0.7)
    
    # Show best MAE solution
    ax4.plot(best_mae_solution['sim_time'], best_mae_solution['voltage_sim'], 
             'g-', label=f'Best MAE (C/{1/best_mae_solution["c_rate"]:.1f})', linewidth=2)
    
    # Show best |ΔT| solution
    ax4.plot(best_time_solution['sim_time'], best_time_solution['voltage_sim'], 
             'b--', label=f'Best |ΔT| (C/{1/best_time_solution["c_rate"]:.1f})', linewidth=2)
    
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Voltage [V]')
    ax4.set_title('Best Solutions Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return pareto_front, best_mae_solution, best_time_solution

def calculate_multi_objective_cost(**params_dict):
    """
    Calculate multi-objective cost function comparing simulation to experimental data.
    Returns both MAE and discharge time error.
    
    Parameters
    ----------
    **params_dict : dict
        Dictionary containing parameters to evaluate
    
    Returns
    -------
    tuple
        (mae, time_error) - Mean Absolute Error and discharge time error
    """
    global EXPERIMENTAL_DATA, TIME_DATA, CURRENT_DATA, VOLTAGE_DATA, SAMPLING_INFO
    
    # Get experimental data from global variable
    capacity_exp, voltage_exp = EXPERIMENTAL_DATA
    time_data = TIME_DATA
    voltage_data = VOLTAGE_DATA
    
    # Calculate experimental discharge time
    exp_discharge_time = time_data[-1]
    for t, v in zip(time_data, voltage_data):
        if v < 3.0:
            exp_discharge_time = t
            break
    
    # Run simulation with current parameters
    capacity_sim, voltage_sim, time_sim = run_msmr_simulation(params_dict)
    
    # Calculate MAE
    if len(capacity_sim) > 1:
        # Check if experimental capacity range is within simulation range
        exp_min_in_sim = capacity_exp.min() >= capacity_sim.min()
        exp_max_in_sim = capacity_exp.max() <= capacity_sim.max()
        
        if exp_min_in_sim and exp_max_in_sim:
            voltage_sim_interp = np.interp(capacity_exp, capacity_sim, voltage_sim)
            mae = np.mean(np.abs(voltage_exp - voltage_sim_interp))
        else:
            mae = float('inf')
    else:
        mae = float('inf')
    
    # Calculate discharge time error
    if len(time_sim) > 0:
        sim_discharge_time = time_sim[-1]
        for t, v in zip(time_sim, voltage_sim):
            if v < 3.0:
                sim_discharge_time = t
                break
        time_error = abs(exp_discharge_time - sim_discharge_time)
    else:
        time_error = float('inf')
    
    return mae, time_error

if __name__ == "__main__":
    # Load all experimental data once
    load_all_data()
    # Perform C-rate optimization
    print("Running C-rate optimization...")
    pareto_front, best_mae, best_time = multi_objective_optimization()
    if pareto_front:
        print(f"\nC-rate optimization complete!")
        print(f"Found {len(pareto_front)} Pareto-optimal solutions")
        print(f"Best MAE solution: C/{1/best_mae['c_rate']:.1f}")
        print(f"Best Time solution: C/{1/best_time['c_rate']:.1f}")
    else:
        print(f"\nC-rate optimization failed!")
