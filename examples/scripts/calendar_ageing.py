import numpy as np
import pybamm as pb
import concurrent.futures
import time

pb.set_logging_level("WARNING")

def run_simulation(days):
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
        
        # Solve with 30% initial SOC
        sol = sim.solve(t_eval=t_eval, solver=solver, initial_soc=0.3)
        
        # Extract voltage data
        voltage = sol["Voltage [V]"].entries
        voltage_before = voltage[0]
        voltage_after = voltage[-1]
        delta_v = voltage_before - voltage_after
        
        return days, voltage_before, voltage_after, delta_v
    except Exception as e:
        return days, None, None, str(e)

if __name__ == "__main__":
    # From 2 to 30 days
    day_range = list(range(2, 31))
    
    start_time = time.time()
    
    print(f"{'Days':<5} | {'V_before (V)':<15} | {'V_after (V)':<15} | {'Delta V (V)':<15}")
    print("-" * 57)
    
    # Optimize for parallelism using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map ensures the results are returned in the exact order of day_range
        results = list(executor.map(run_simulation, day_range))
        
    # Print results summary
    for res in results:
        if res[1] is None:
            print(f"{res[0]:<5} | Error: {res[3]}")
        else:
            days, v_before, v_after, d_v = res
            print(f"{days:<5} | {v_before:<15.6f} | {v_after:<15.6f} | {d_v:<15.6f}")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
