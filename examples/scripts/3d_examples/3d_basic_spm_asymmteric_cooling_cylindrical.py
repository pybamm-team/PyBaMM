import pybamm

print("Setting up asymmetric radial cooling simulation for a cylinder...")
model_3d = pybamm.lithium_ion.BasicSPM_with_3DThermal(
    options={"cell geometry": "cylindrical", "dimensionality": 3}
)

parameter_values = pybamm.ParameterValues("NCA_Kim2011")

# Define our cooling scenario
h_cooling = 20  # W.m-2.K-1 -> Cooling on the outside surface
h_insulation = 0.1  # W.m-2.K-1 -> Insulation on other surfaces

parameter_values.update(
    {
        "Inner cell radius [m]": 0.005,
        "Outer cell radius [m]": 0.018,
        # Apply cooling to the outer radial surface
        "Outer radius heat transfer coefficient [W.m-2.K-1]": h_cooling,
        # Insulate the inner radius and the flat top/bottom faces
        "Inner radius heat transfer coefficient [W.m-2.K-1]": h_insulation,
        "Bottom face heat transfer coefficient [W.m-2.K-1]": h_insulation,
        "Top face heat transfer coefficient [W.m-2.K-1]": h_insulation,
    },
    check_already_exists=False,
)

# Use a high discharge rate to generate significant heat
experiment = pybamm.Experiment([("Discharge at 4C until 2.5V")])

var_pts = {
    "x_n": 20,
    "x_s": 20,
    "x_p": 20,
    "r_n": 30,
    "r_p": 30,
    "r_macro": 20,
    "z": 20,
}

sim = pybamm.Simulation(
    model_3d,
    parameter_values=parameter_values,
    var_pts=var_pts,
    experiment=experiment,
)
print("Solving... (this may take a minute)")
solution = sim.solve()
print("Solve complete.")

print("Generating plots...")
final_time = solution.t[-1]

# Plot the overall voltage and average temperature
solution.plot(["Voltage [V]", "Volume-averaged cell temperature [K]"])

# Create detailed heatmaps at the final timestep
print(f"\n--- Displaying heatmaps at t={final_time:.0f}s ---")

# Plot the r-z plane to show the radial gradient (hot core, cool edge)
pybamm.plot_3d_cross_section(solution, t=final_time, plane="rz")

# Plot the polar xy-plane to show the temperature at a mid-height slice
pybamm.plot_3d_cross_section(solution, t=final_time, plane="xy", position=0.5)

# Plot a 3D heatmap of the temperature distribution
pybamm.plot_3d_heatmap(solution, t=final_time, marker_size=5)
