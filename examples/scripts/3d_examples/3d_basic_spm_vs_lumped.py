import matplotlib.pyplot as plt
import numpy as np

import pybamm

model_3d = pybamm.lithium_ion.BasicSPM_with_3DThermal(
    options={"cell geometry": "box", "dimensionality": 3}
)

params_3d = pybamm.ParameterValues("Chen2020")

t_ramp = np.array([0, 5, 3600])  # Changed 0.1 to 5
I_ramp = np.array([0, 5, 5])
current_func = pybamm.Interpolant(t_ramp, I_ramp, pybamm.t)
h = 100
params_3d.update(
    {
        "Current function [A]": current_func,
        "Ambient temperature [K]": 298.15,
        "Initial temperature [K]": 298.15,
        "Left face heat transfer coefficient [W.m-2.K-1]": h,
        "Right face heat transfer coefficient [W.m-2.K-1]": h,
        "Front face heat transfer coefficient [W.m-2.K-1]": h,
        "Back face heat transfer coefficient [W.m-2.K-1]": h,
        "Bottom face heat transfer coefficient [W.m-2.K-1]": h,
        "Top face heat transfer coefficient [W.m-2.K-1]": h,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1e5,
        "Separator thermal conductivity [W.m-1.K-1]": 1e5,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 1e5,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 1e5,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 1e5,
    },
    check_already_exists=False,
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
sim_3d = pybamm.Simulation(model_3d, parameter_values=params_3d, var_pts=var_pts)
sol_3d = sim_3d.solve([0, 3600])
print("3D model solved.")

print("\nSetting up lumped thermal model...")
model_lumped = pybamm.lithium_ion.SPM(options={"thermal": "lumped"})

params_lumped = pybamm.ParameterValues("Chen2020")

params_lumped.update(
    {
        "Current function [A]": current_func,
        "Ambient temperature [K]": 298.15,
        "Initial temperature [K]": 298.15,
        "Total heat transfer coefficient [W.m-2.K-1]": h,
    },
    check_already_exists=False,
)

sim_lumped = pybamm.Simulation(model_lumped, parameter_values=params_lumped)
sol_lumped = sim_lumped.solve([0, 3600])
print("Lumped model solved.")

print("\nPlotting comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(
    sol_3d["Time [s]"].data, sol_3d["Voltage [V]"].data, "b-", label="3D Thermal Model"
)
ax1.plot(
    sol_lumped["Time [s]"].data,
    sol_lumped["Voltage [V]"].data,
    "r--",
    label="Lumped Thermal Model",
)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Voltage [V]")
ax1.set_title("Voltage Comparison")
ax1.legend()
ax1.grid(True)

ax2.plot(
    sol_3d["Time [s]"].data,
    sol_3d["Volume-averaged cell temperature [K]"].data,
    "b-",
    label="3D Thermal Model (Avg)",
)
ax2.plot(
    sol_lumped["Time [s]"].data,
    sol_lumped["Volume-averaged cell temperature [K]"].data,
    "r--",
    label="Lumped Thermal Model",
)
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Temperature [K]")
ax2.set_title("Temperature Comparison")
ax2.legend()
ax2.grid(True)

fig.suptitle("Model Comparison: 3D Thermal (High λ) vs. Lumped Thermal", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
