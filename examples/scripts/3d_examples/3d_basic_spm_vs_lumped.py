import matplotlib.pyplot as plt
import numpy as np

import pybamm

# Load models
models = {
    "Lumped": pybamm.lithium_ion.SPM(options={"thermal": "lumped"}),
    "3D": pybamm.lithium_ion.BasicSPM_with_3DThermal(
        options={"cell geometry": "box", "dimensionality": 3}
    ),
}

# Load parameters
parameter_values = pybamm.ParameterValues("Marquis2019")
h = 1
parameter_values.update(
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

# Set up and solve simulations
experiment = pybamm.Experiment(
    [
        ("Discharge at 5C until 2.8V", "Rest for 10 minutes"),
    ]
)
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

solutions = {}
for model_name, model in models.items():
    print(f"Solving {model_name} model...")
    sim = pybamm.Simulation(
        model, parameter_values=parameter_values, var_pts=var_pts, experiment=experiment
    )
    solutions[model_name] = sim.solve()
    print(f"{model_name} model solved.")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
styles = {
    "3D": {"color": "b", "linestyle": "-", "label": "3D Thermal Model"},
    "Lumped": {"color": "r", "linestyle": "--", "label": "Lumped Thermal Model"},
}
for name, sol in solutions.items():
    ax1.plot(sol["Time [s]"].data, sol["Voltage [V]"].data, **styles[name])
    ax2.plot(
        sol["Time [s]"].data,
        sol["Volume-averaged cell temperature [K]"].data,
        **styles[name],
    )

ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Voltage [V]")
ax1.set_title("Voltage Comparison")
ax1.legend()
ax1.grid(True)

ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Temperature [K]")
ax2.set_title("Temperature Comparison")
ax2.legend()
ax2.grid(True)

fig.suptitle("Model Comparison: 3D Thermal vs. Lumped Thermal", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
