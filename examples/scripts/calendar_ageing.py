import numpy as np
import pybamm as pb
import concurrent.futures
import time
import csv
import matplotlib.pyplot as plt
from pathlib import Path

pb.set_logging_level("WARNING")

def run_simulation(args):
    days, initial_soc = args
    try:
        # Create a new DFN model with reaction limited SEI for calendar ageing
        model = pb.lithium_ion.DFN({"SEI": "reaction limited"})
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
        voltage_before = voltage[0]
        voltage_after = voltage[-1]
        delta_v = voltage_before - voltage_after
        
        return initial_soc, days, voltage_before, voltage_after, delta_v
    except Exception as e:
        return initial_soc, days, None, None, str(e)

if __name__ == "__main__":
    socs_to_test = [0.3, 0.9]
    day_range = list(range(2, 31))
    
    # Create an array of argument tuples for parallel execution
    simulation_args = []
    for soc in socs_to_test:
        for days in day_range:
            simulation_args.append((days, soc))
    
    start_time = time.time()
    
    print(f"Starting simulations for SOCs {socs_to_test} across {len(day_range)} durations...")
    
    # Process all combinations in parallel
    results_dict = {soc: {"days": [], "delta_v": []} for soc in socs_to_test}
    csv_rows = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map ensures the results are returned in the exact order of simulation_args
        results = list(executor.map(run_simulation, simulation_args))
        
    print(f"\nTotal simulation execution time: {time.time() - start_time:.2f} seconds\n")
    
    # Process results locally and populate data structures for CSV and Plotting
    for res in results:
        soc, days, v_before, v_after, d_v = res
        if v_before is None:
            print(f"Error for SOC {soc:.0%} at {days} days: {d_v}")
        else:
            results_dict[soc]["days"].append(days)
            results_dict[soc]["delta_v"].append(d_v)
            csv_rows.append([soc, days, v_before, v_after, d_v])

    # 1. Write to CSV
    csv_filename = Path(__file__).parent / "calendar_ageing_results.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Initial SOC", "Days", "Voltage Before (V)", "Voltage After (V)", "Delta V (V)"])
        writer.writerows(csv_rows)
    print(f"Results successfully saved to: {csv_filename}")
    
    # 2. Plotting the results
    plt.figure(figsize=(10, 6))
    
    for soc in socs_to_test:
        # Need to put delta_v into millivolts for better readability
        delta_v_mv = [dv * 1000 for dv in results_dict[soc]["delta_v"]]
        plt.plot(results_dict[soc]["days"], delta_v_mv, marker='o', label=f"{soc * 100:.0f}% SOC")
        
    plt.title("Voltage Loss ($\Delta$V) vs Rest Duration for Calendar Ageing")
    plt.xlabel("Rest Duration (Days)")
    plt.ylabel("Voltage Loss (mV)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plot_filename = Path(__file__).parent / "calendar_ageing_plot.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {plot_filename}")

