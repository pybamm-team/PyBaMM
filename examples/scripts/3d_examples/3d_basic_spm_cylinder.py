import pybamm

pybamm.set_logging_level("INFO")

model_options = {"cell geometry": "cylindrical", "dimensionality": 3}
model = pybamm.lithium_ion.BasicSPM_with_3DThermal(options=model_options)

params = pybamm.ParameterValues("NCA_Kim2011")

nominal_capacity_Ah = params["Nominal cell capacity [A.h]"]
c_rate_current = nominal_capacity_Ah  

params.update({
    "Current function [A]": c_rate_current/10, 
    "Inner cell radius [m]": 0.005, 
    "Outer cell radius [m]": 0.018, 
    "Ambient temperature [K]": 298.15,
    "Initial temperature [K]": 298.15,
    "Inner radius heat transfer coefficient [W.m-2.K-1]": 100, 
    "Outer radius heat transfer coefficient [W.m-2.K-1]": 100, 
    "Bottom face heat transfer coefficient [W.m-2.K-1]": 100,    
    "Top face heat transfer coefficient [W.m-2.K-1]": 100,       
    
}, check_already_exists=False)

var_pts = {
    "x_n": 20,
    "x_s": 20,
    "x_p": 20,
    "r_n": 30,
    "r_p": 30,
    "r_macro": None,
    "y": None,
    "z": None
}
sim = pybamm.Simulation(model, parameter_values=params, var_pts=var_pts)

solution = sim.solve([0, 3600])

sim.plot([
    "Voltage [V]", 
    "Volume-averaged cell temperature [K]",
    "Current [A]",
])

print("\n" + "="*50)
print("CYLINDRICAL CELL THERMAL ANALYSIS")
print("="*50)

time = solution.t
temp_avg = solution["Volume-averaged cell temperature [K]"].data

print(f"Initial temperature: {temp_avg[0]:.2f} K ({temp_avg[0]-273.15:.2f} °C)")
print(f"Final temperature: {temp_avg[-1]:.2f} K ({temp_avg[-1]-273.15:.2f} °C)")
print(f"Temperature rise: {temp_avg[-1] - temp_avg[0]:.2f} K")
print(f"Maximum temperature: {temp_avg.max():.2f} K ({temp_avg.max()-273.15:.2f} °C)")

pybamm.plot_cross_section(
    solution,
    variable="Cell temperature [K]",
    t=time[-1],
    plane="rz",
    position=0.5,
    ax=None,
    show_plot=True,
    levels=20,
    cmap="inferno",
)