import matplotlib.pyplot as plt
import numpy as np

import pybamm

# Load models
models = {
    "Lumped": pybamm.lithium_ion.SPM(options={"thermal": "lumped"}),
    "3D": pybamm.lithium_ion.Basic3DThermalSPM(
        options={"cell geometry": "pouch", "dimensionality": 3}
    ),
}

# Load base parameters
parameter_values = pybamm.ParameterValues("Marquis2019")

# Define heat transfer coefficients to test
h_values = [0.1, 1, 10, 100]  # W/m2.K

# Set up experiment
experiment = pybamm.Experiment(
    [
        ("Discharge at 3C until 2.8V", "Rest for 10 minutes"),
    ]
)

# Set spatial points
var_pts = {
    "x_n": 20,
    "x_s": 20,
    "x_p": 20,
    "r_n": 30,
    "r_p": 30,
    "x": None,
    "y": None,
    "z": None,
}

# Dictionary to store solutions for each h value
all_solutions = {}

# Loop over different h values
for h in h_values:
    print(f"\nSolving models for h = {h} W/m2.K...")

    # Update parameter values for this h
    h_params = parameter_values.copy()
    h_params.update(
        {
            # Used in lumped model
            "Total heat transfer coefficient [W.m-2.K-1]": h,
            # Used in 3D model
            "Left face heat transfer coefficient [W.m-2.K-1]": h,
            "Right face heat transfer coefficient [W.m-2.K-1]": h,
            "Front face heat transfer coefficient [W.m-2.K-1]": h,
            "Back face heat transfer coefficient [W.m-2.K-1]": h,
            "Bottom face heat transfer coefficient [W.m-2.K-1]": h,
            "Top face heat transfer coefficient [W.m-2.K-1]": h,
        },
        check_already_exists=False,
    )

    solutions = {}
    for model_name, model in models.items():
        print(f"Solving {model_name} model...")
        sim = pybamm.Simulation(
            model, parameter_values=h_params, var_pts=var_pts, experiment=experiment
        )
        solutions[model_name] = sim.solve()
        print(f"{model_name} model solved.")

    all_solutions[h] = solutions

# Plot results for all h values on same axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
# Define color map for h values
colors = plt.cm.viridis(np.linspace(0, 1, len(h_values)))
styles = {
    "3D": {"linestyle": "-", "label": "3D Thermal Model"},
    "Lumped": {"linestyle": "--", "label": "Lumped Thermal Model"},
}

for i, h in enumerate(h_values):
    solutions = all_solutions[h]
    for name, sol in solutions.items():
        style = styles[name].copy()
        style["color"] = colors[i]
        style["label"] = f"{style['label']} (h={h})"

        ax1.plot(sol["Time [s]"].data, sol["Voltage [V]"].data, **style)
        ax2.plot(
            sol["Time [s]"].data,
            sol["Volume-averaged cell temperature [K]"].data,
            **style,
        )

ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Voltage [V]")
ax1.set_title("Voltage Comparison")
ax1.grid(True)

ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Temperature [K]")
ax2.set_title("Temperature Comparison")
ax2.grid(True)
ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
fig.suptitle("Model Comparison: 3D Thermal vs. Lumped Thermal", fontsize=16)
plt.tight_layout()
plt.show()
