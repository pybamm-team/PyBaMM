import sys
import json
import os
import pybamm
import matplotlib.pyplot as plt
import numpy as np
import time

# ... (rest of configuration)

CYCLES_TO_RUN = 50

def run_investigation(config_name, var_pts, tol):
    print(f"\n--- Running Config: {config_name} ---")
    print(f"  Mesh: {var_pts}")
    print(f"  Tol: {tol}")
    
    # "No Plating" Variant Parameters
    options = {
        "SEI": "solvent-diffusion limited",
        "SEI porosity change": "true",
        "lithium plating": "none",
        "lithium plating porosity change": "false",
        "particle mechanics": ("swelling and cracking", "swelling only"),
        "SEI on cracks": "true",
        "loss of active material": "none",
    }
    
    parameter_values = pybamm.ParameterValues("OKane2022")
    solver = pybamm.IDAKLUSolver(atol=tol, rtol=tol)
    
    # 1. Cycle to build up gradients
    CYCLES_PER_CHUNK = 5
    NUM_CHUNKS = CYCLES_TO_RUN // CYCLES_PER_CHUNK
    
    experiment_chunk = pybamm.Experiment(
        [
            (
                "Discharge at C/9 until 3.2 V",
                "Rest for 15 minutes",
                "Charge at C/7 until 4.1 V",
                "Hold at 4.1 V until C/37",
                "Rest for 15 minutes",
                "Discharge at C/4 for 5s",
            )
        ] * CYCLES_PER_CHUNK
    )
    
    # helper for mesh creation to avoid duplication
    def get_submesh(model_obj):
        s_mesh = model_obj.default_submesh_types.copy()
        s_mesh["negative particle"] = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params={"side": "right"})
        s_mesh["positive particle"] = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params={"side": "right"})
        return s_mesh

    print(f"  Cycling for {CYCLES_TO_RUN} cycles (in {NUM_CHUNKS} chunks of {CYCLES_PER_CHUNK})...")
    t0 = time.time()
    
    starting_solution = None
    
    for chunk_idx in range(NUM_CHUNKS):
        print(f"    Chunk {chunk_idx + 1}/{NUM_CHUNKS}...")
        
        # Create fresh model/sim for each chunk
        model = pybamm.lithium_ion.DFN(options)
        submesh = get_submesh(model)
        
        sim = pybamm.Simulation(
            model, 
            experiment=experiment_chunk,
            parameter_values=parameter_values, 
            solver=solver,
            var_pts=var_pts,
            submesh_types=submesh
        )
        
        if starting_solution:
            sim.model.set_initial_conditions_from(starting_solution)
        
        try:
            if chunk_idx == 0:
                sim.solve(initial_soc=0.3)
            else:
                sim.solve()
        except pybamm.SolverError as e:
            print(f"Solver Error in chunk {chunk_idx}: {e}")
            return None
            
        starting_solution = sim.solution
        
        # Explicitly clear old sim to free memory
        sim = None
        # gc.collect() - handled by python mostly, but can add import gc if needed (it is imported?)
        # import gc inside function if global missing
        import gc
        gc.collect()

    print(f"  Cycling done in {time.time()-t0:.2f}s")
    
    # Rest
    # ... (same rest logic)
    # Rest
    print("  Running Rest Step...")
    rest_model = pybamm.lithium_ion.DFN(options)
    side = "right"
    rest_submesh = rest_model.default_submesh_types.copy()
    rest_submesh["negative particle"] = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params={"side": side})
    rest_submesh["positive particle"] = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params={"side": side})
    
    rest_sim = pybamm.Simulation(
        rest_model,
        experiment=pybamm.Experiment(["Rest for 96 hours"]),
        parameter_values=parameter_values,
        solver=solver,
        var_pts=var_pts,
        submesh_types=rest_submesh
    )
    rest_sim.model.set_initial_conditions_from(starting_solution)
    rest_sim.solve()
    
    # Extract Data for JSON
    sol = rest_sim.solution
    t = sol["Time [h]"].entries
    t = t - t[0]
    v = sol["Terminal voltage [V]"].entries
    cn_surf = sol["X-averaged negative particle surface concentration [mol.m-3]"].entries
    cp_surf = sol["X-averaged positive particle surface concentration [mol.m-3]"].entries
    # Extract Spatial Data (r nodes)
    # Note: rest_sim.mesh gives the mesh. 
    # For DFN, "negative particle" submesh nodes are what we want.
    # The nodes are scaled 0 to 1.
    r_n = rest_sim.mesh["negative particle"].nodes
    r_p = rest_sim.mesh["positive particle"].nodes
    
    # Extract full concentration profiles (r, t)
    # cn_avg from previous lines corresponds to "X-averaged ..."
    # We want the actual distribution along r.
    cn_dist = sol["X-averaged negative particle concentration [mol.m-3]"].entries
    cp_dist = sol["X-averaged positive particle concentration [mol.m-3]"].entries
    
    # Get Temporal Snapshots (Indices)
    # t is in hours. We want: 0, ~10s, ~1m, ~10m, ~1h, End
    target_times = [0, 0.0028, 0.0167, 0.167, 1.0, t[-1]]
    indices = []
    for target in target_times:
        # Find closest index
        idx = (np.abs(t - target)).argmin()
        if idx not in indices:
            indices.append(int(idx))
    
    # Extract profiles at these indices
    cn_profiles = [cn_dist[:, i].tolist() for i in indices]
    profile_times = [t[i] for i in indices]

    # Get Start (t=0) and End (t=-1) profiles
    cn_start = cn_dist[:, 0]
    cn_end = cn_dist[:, -1]
    cp_start = cp_dist[:, 0]
    cp_end = cp_dist[:, -1]
    
    # Restore compat for existing plots
    cn_avg = cn_dist
    cp_avg = cp_dist
    
    return {
        "time": t.tolist(), 
        "voltage": v.tolist(), 
        "config": config_name,
        "cn_surf": cn_surf.tolist(),
        "cp_surf": cp_surf.tolist(),
        "cn_avg": cn_avg.tolist(),
        "cp_avg": cp_avg.tolist(),
        "spatial": {
            "r_n": r_n.tolist(),
            "r_p": r_p.tolist(),
            "cn_start": cn_start.tolist(),
            "cn_end": cn_end.tolist(),
            "cp_start": cp_start.tolist(),
            "cp_end": cp_end.tolist(),
            "cn_profiles": cn_profiles,
            "times": profile_times
        }
    }

if __name__ == "__main__":
    mode = sys.argv[1] # "run" or "plot"
    
    if mode == "run":
        config_type = sys.argv[2] # "standard" or "refined"
        
        if config_type == "standard":
            pts = {"x_n": 10, "x_s": 10, "x_p": 10, "r_n": 20, "r_p": 20}
            tol = 1e-8
        elif config_type == "refined":
            # Increased mesh size as per user request (Chunking enabled)
            pts = {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 50, "r_p": 50}
            tol = 1e-9
            
        data = run_investigation(config_type, pts, tol)
        if data:
            with open(f"instability_data_{config_type}.json", "w") as f:
                json.dump(data, f)
            print(f"Saved instability_data_{config_type}.json")
            
    elif mode == "plot":
        # Plot 1: Voltage
        plt.figure(figsize=(10, 6))
        for cfg in ["standard", "refined"]:
            fname = f"instability_data_{cfg}.json"
            if os.path.exists(fname):
                with open(fname, "r") as f:
                    d = json.load(f)
                plt.plot(d["time"], d["voltage"], label=f"{d['config']}", alpha=0.8)
                print(f"Config {cfg}: Time {d['time'][0]} to {d['time'][-1]}")
        plt.xlabel("Rest Time [h]")
        plt.ylabel("Voltage [V]")
        plt.title("Voltage Relaxation Instability Investigation (96h Rest)")
        plt.legend()
        plt.grid(True)
        plt.savefig("voltage_instability_investigation.png")
        print("Plot saved to voltage_instability_investigation.png")
        
        # Plot 2: Concentration (Negative)
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        for cfg in ["standard", "refined"]:
            fname = f"instability_data_{cfg}.json"
            if os.path.exists(fname):
                with open(fname, "r") as f:
                    d = json.load(f)
                # cn_avg is likely (r, t) or just (t,). If it failed with (20, 155), it's (r, t) for 20 nodes.
                # Let's take the mean over the particle radius for a quick "Average" visualization
                cn_surf = np.array(d["cn_surf"])
                cn_avg = np.array(d["cn_avg"])
                if cn_avg.ndim > 1: cn_avg = np.mean(cn_avg, axis=0)
                
                plt.plot(d["time"], cn_surf, label=f"{d['config']} (Surf)", linestyle="-")
                plt.plot(d["time"], cn_avg, label=f"{d['config']} (Vol-Avg)", linestyle="--")
        plt.xlabel("Rest Time [h]")
        plt.ylabel("Conecntration [mol.m-3]")
        plt.title("Negative Electrode Concentration during Rest")
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Concentration (Positive)
        plt.subplot(2, 1, 2)
        for cfg in ["standard", "refined"]:
            fname = f"instability_data_{cfg}.json"
            if os.path.exists(fname):
                with open(fname, "r") as f:
                    d = json.load(f)
                
                cp_surf = np.array(d["cp_surf"])
                cp_avg = np.array(d["cp_avg"])
                if cp_avg.ndim > 1: cp_avg = np.mean(cp_avg, axis=0)

                plt.plot(d["time"], cp_surf, label=f"{d['config']} (Surf)", linestyle="-")
                plt.plot(d["time"], cp_avg, label=f"{d['config']} (Vol-Avg)", linestyle="--")
        plt.xlabel("Rest Time [h]")
        plt.ylabel("Conecntration [mol.m-3]")
        plt.title("Positive Electrode Concentration during Rest")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("concentration_during_rest.png")
        print("Plot saved to concentration_during_rest.png")
        
        # Plot 3: Spatial Distribution
        plt.figure(figsize=(14, 6))
        
        for cfg in ["standard", "refined"]:
            fname = f"instability_data_{cfg}.json"
            if os.path.exists(fname):
                with open(fname, "r") as f:
                    d = json.load(f)
                
                if "spatial" in d:
                    s = d["spatial"]
                    r_n = s["r_n"]
                    r_p = s["r_p"]
                    
                    # Anode
                    plt.subplot(1, 2, 1)
                    plt.plot(r_n, s["cn_start"], label=f"{d['config']} (Start)", linestyle='--', alpha=0.7)
                    plt.plot(r_n, s["cn_end"], label=f"{d['config']} (End)", linestyle='-', alpha=0.9)
                    
                    # Cathode
                    plt.subplot(1, 2, 2)
                    plt.plot(r_p, s["cp_start"], label=f"{d['config']} (Start)", linestyle='--', alpha=0.7)
                    plt.plot(r_p, s["cp_end"], label=f"{d['config']} (End)", linestyle='-', alpha=0.9)
        
        plt.subplot(1, 2, 1)
        plt.xlabel("Normalized Radius (r/Rp)")
        plt.ylabel("Concentration [mol.m-3]")
        plt.title("Negative Electrode Particle Distribution")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.xlabel("Normalized Radius (r/Rp)")
        plt.ylabel("Concentration [mol.m-3]")
        plt.title("Positive Electrode Particle Distribution")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("concentration_spatial_distribution.png")
        print("Plot saved to concentration_spatial_distribution.png")
        
        # Plot 4: Anode Focused Spatial Distribution (Evolution)
        plt.figure(figsize=(12, 8))
        R_n = 5.86e-6  # Negative particle radius for OKane2022
        
        # Prefer refined, fallback to standard
        target_cfg = "refined"
        fname = f"instability_data_{target_cfg}.json"
        if not os.path.exists(fname):
            target_cfg = "standard"
            fname = f"instability_data_{target_cfg}.json"
            
        if os.path.exists(fname):
            with open(fname, "r") as f:
                d = json.load(f)
            
            if "spatial" in d:
                s = d["spatial"]
                r_n = np.array(s["r_n"]) / R_n
                
                # Check if we have profiles list (new format) or just start/end (old format)
                if "cn_profiles" in s:
                    profiles = s["cn_profiles"]
                    times = s["times"]
                    
                    # Colormap
                    import matplotlib.cm as cm
                    colors = cm.viridis(np.linspace(0, 1, len(profiles)))
                    
                    for i, (prof, t_val) in enumerate(zip(profiles, times)):
                        label_t = f"{t_val*3600:.1f}s" if t_val < 0.1 else f"{t_val:.2f}h"
                        if i == 0: label_t = "Start (0s)"
                        if i == len(profiles)-1: label_t = f"End ({t_val:.1f}h)"
                        
                        plt.plot(r_n, prof, label=label_t, color=colors[i], linewidth=2 if i in [0, len(profiles)-1] else 1.5)
                else:
                    # Fallback for old data format
                    plt.plot(r_n, s["cn_start"], label="Start", linestyle='--')
                    plt.plot(r_n, s["cn_end"], label="End", linestyle='-')

        plt.xlabel("Normalized Radius ($r/R_n$)", fontsize=14)
        plt.ylabel("Concentration [mol.m$^{-3}$]", fontsize=14)
        plt.title(f"Anode Particle Relaxation ({target_cfg.capitalize()} Mesh)", fontsize=16)
        plt.legend(title="Rest Time", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("anode_spatial_distribution.png")
        print("Plot saved to anode_spatial_distribution.png")
