import numpy as np

import pybamm

model_3d = pybamm.lithium_ion.BasicSPM_with_3DThermal(
    options={"cell geometry": "box", "dimensionality": 3}
)

params_3d = pybamm.ParameterValues("Chen2020")

t_ramp = np.array([0, 5, 3600])
I_ramp = np.array([0, 5, 5])
current_func = pybamm.Interpolant(t_ramp, I_ramp, pybamm.t)
h = 100
params_3d.update(
    {
        "Current function [A]": current_func / 10,
        "Ambient temperature [K]": 298.15,
        "Initial temperature [K]": 298.15,
        "Left face heat transfer coefficient [W.m-2.K-1]": h,
        "Right face heat transfer coefficient [W.m-2.K-1]": h,
        "Front face heat transfer coefficient [W.m-2.K-1]": h,
        "Back face heat transfer coefficient [W.m-2.K-1]": h,
        "Bottom face heat transfer coefficient [W.m-2.K-1]": h,
        "Top face heat transfer coefficient [W.m-2.K-1]": h,
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
temp_arr = sol_3d["Cell temperature [K]"].entries
print(np.mean(temp_arr), np.max(temp_arr), np.min(temp_arr))
print("3D model solved.")

sol_3d.plot(
    [
        "Voltage [V]",
        "Volume-averaged cell temperature [K]",
        "Current [A]",
    ]
)

pybamm.plot_cross_section(
    sol_3d,
    variable="Cell temperature [K]",
    t=3600,
    plane="xy",
    position=0.5,
    show_plot=True,
)
