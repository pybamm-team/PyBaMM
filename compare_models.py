import gc

import matplotlib.pyplot as plt

import pybamm

# Define generic SEI parameters (taken from Mohtat2020 default to apply to Chen)
sei_params = {
    "SEI kinetic rate constant [m.s-1]": 1e-15,  # Tuned value we used earlier
    "SEI open-circuit potential [V]": 0.4,
    "SEI resistivity [Ohm.m]": 200000.0,
    "Outer SEI partial molar volume [m3.mol-1]": 9.585e-05,
    "Ratio of inner and outer SEI partial molar volumes": 1.0,
    "Inner SEI partial molar volume [m3.mol-1]": 9.585e-05,
    "SEI reaction exchange current density [A.m-2]": 1.5e-07,
    "Initial outer SEI thickness [m]": 5e-09,
    "Initial inner SEI thickness [m]": 0.0,
}

# Experiment
experiment = pybamm.Experiment(
    [
        (
            "Discharge at C/9 until 3.2 V",
            "Rest for 15 minutes",
            "Charge at C/7 until 4.1 V",
            "Hold at 4.1 V until C/37",
            "Rest for 15 minutes",
            "Discharge at C/4 for 5s",
            "Rest for 15 minutes",
        )
    ]
    * 50  # 50 cycles for speed
)

# Mesh settings from isolate_mechanism.py
var_pts = {"x_n": 10, "x_s": 10, "x_p": 10, "r_n": 20, "r_p": 20}

# Pre-load Chen Kinetics for injection
chen_params_raw = pybamm.ParameterValues("Chen2020")
chen_j0 = chen_params_raw["Negative electrode exchange-current density [A.m-2]"]

# Models to compare
mechanisms = ["reaction limited", "solvent-diffusion limited"]
# param_sets = ["Mohtat2020", "Chen2020", "OKane2022"]
param_sets = ["Chen2020", "OKane2022", "OKane2022 (Chen Diffusivity)"]

results = []

for mech in mechanisms:
    for p_name in param_sets:
        print(f"Running {p_name} with {mech}...")

        # Handle Parameter Loading
        base_name = "OKane2022" if "OKane" in p_name else p_name
        params = pybamm.ParameterValues(base_name)

        # Inject SEI Parameters (Baseline)
        params.update(sei_params, check_already_exists=False)

        # Inject Chen Diffusivity if requested
        if "(Chen Diffusivity)" in p_name:
            # Chen's constant value (hardcoded or extracted)
            # From previous diff: 3.3e-14
            new_diff = 3.3e-14
            params.update(
                {"Negative particle diffusivity [m2.s-1]": new_diff},
                check_already_exists=False,
            )
            print(
                f"  -> INJECTED Diffusivity: {params['Negative particle diffusivity [m2.s-1]']}"
            )
        else:
            print(
                f"  -> Default Diffusivity: {params['Negative particle diffusivity [m2.s-1]']}"
            )

        # Inject Chen Kinetics/OCP (Removing this for now to isolate Diffusivity)
        if "(Chen Kinetics)" in p_name:
            pass

        # Create Model
        options = {"SEI": mech, "SEI porosity change": "true"}
        model = pybamm.lithium_ion.DFN(options)

        # Apply submesh types (Exponential for particles)
        submesh_types = model.default_submesh_types.copy()
        submesh_types["negative particle"] = pybamm.MeshGenerator(
            pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}
        )
        submesh_types["positive particle"] = pybamm.MeshGenerator(
            pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}
        )

        solver = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8)

        sim = pybamm.Simulation(
            model,
            experiment=experiment,
            parameter_values=params,
            solver=solver,
            var_pts=var_pts,
            submesh_types=submesh_types,
        )

        try:
            sim.solve()

            # Extract Data
            sol = sim.solution
            t = sol["Time [h]"].entries
            # cycles = sol["Cycle number"].entries # Not reliable

            # Extract SEI
            # Check variable names
            sei_var = "X-averaged total SEI thickness [m]"
            if sei_var not in sol.all_models[0].variables:
                # Fallback
                sei_var = "X-averaged negative SEI thickness [m]"

            sei_thickness = sol[sei_var].entries

            results.append(
                {
                    "ParameterSet": p_name,
                    "Mechanism": mech,
                    "Time": t,
                    "SEI Thickness": sei_thickness,
                }
            )

            del sol, t, sei_thickness

        except Exception as e:
            print(f"Failed {p_name} {mech}: {e}")

        # Cleanup memory
        del sim, model, params
        gc.collect()

# Plotting
plt.figure(figsize=(10, 8))

colors = {
    "Mohtat2020": "blue",
    "Chen2020": "red",
    "OKane2022": "green",
    "OKane2022 (Chen Diffusivity)": "purple",
}
linestyles = {"reaction limited": "-", "solvent-diffusion limited": "--"}

for res in results:
    p_name = res["ParameterSet"]
    mech = res["Mechanism"]
    t = res["Time"]
    sei = res["SEI Thickness"]

    label = f"{p_name} - {mech}"
    plt.plot(t, sei, color=colors[p_name], linestyle=linestyles[mech], label=label)

plt.xlabel("Time [h]")
plt.ylabel("SEI Thickness [m]")
plt.title("SEI Growth: Mohtat vs Chen (Identical Kinetics)")
plt.legend()
plt.grid(True)
plt.savefig("compare_mohtat_chen.png")

# Analyze total growth
print("--- Summary (End of Experiment) ---")
print(f"{'Setup':<40} | {'Initial [m]':<15} | {'Final [m]':<15}")
for res in results:
    p_name = res["ParameterSet"]
    mech = res["Mechanism"]
    initial_sei = res["SEI Thickness"][0]
    final_sei = res["SEI Thickness"][-1]
    print(f"{p_name + ' ' + mech:<40} | {initial_sei:.4e}     | {final_sei:.4e}")
