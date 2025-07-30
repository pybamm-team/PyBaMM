import pybamm

print("Setting up asymmetric cooling simulation...")
model_3d = pybamm.lithium_ion.Basic3DThermalSPM(
    options={"cell geometry": "pouch", "dimensionality": 3}
)

parameter_values = pybamm.ParameterValues("Ecker2015")

# Define our cooling scenario
h_cooling = 20  # W.m-2.K-1 -> A cooling plate on one side
h_insulation = 0.1  # W.m-2.K-1 -> Good insulation on other sides

parameter_values.update(
    {
        # Apply strong cooling to the RIGHT face of the cell
        "Right face heat transfer coefficient [W.m-2.K-1]": h_cooling,
        # Insulate all other faces
        "Left face heat transfer coefficient [W.m-2.K-1]": h_insulation,
        "Front face heat transfer coefficient [W.m-2.K-1]": h_insulation,
        "Back face heat transfer coefficient [W.m-2.K-1]": h_insulation,
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
    "x": None,
    "y": None,
    "z": None,
}

submesh_types = model_3d.default_submesh_types
submesh_types["cell"] = pybamm.ScikitFemGenerator3D("pouch", h="0.01")  # very fine mesh

sim = pybamm.Simulation(
    model_3d,
    parameter_values=parameter_values,
    var_pts=var_pts,
    experiment=experiment,
    submesh_types=submesh_types,
)
print("Solving... (this may take a minute)")
solution = sim.solve()
print("Solve complete.")

print("Generating plots...")
final_time = solution.t[-1]

# First, plot the overall voltage and average temperature
solution.plot(["Voltage [V]", "Volume-averaged cell temperature [K]"])

# Now, create detailed heatmaps at the final timestep
print(f"\n--- Displaying heatmaps at t={final_time:.0f}s ---")

# Plot a slice through the center, showing the gradient from left (hot) to right (cool)
pybamm.plot_3d_cross_section(
    solution,
    plane="xz",
    variable="Cell temperature [K]",
    t=None,  # Last time stamp
    position=0.5,
    show_mesh=True,
    mesh_color="white",
    mesh_alpha=0.4,
    mesh_linewidth=0.7,
)

# Plot a slice showing the temperature distribution across the face of the cell
pybamm.plot_3d_cross_section(
    solution,
    plane="yz",
    position=0.5,
    t=None,
    variable="Cell temperature [K]",
    show_mesh=True,
    mesh_color="white",
    mesh_alpha=0.4,
    mesh_linewidth=0.7,
)

# Plot a 3D heatmap of the temperature distribution
pybamm.plot_3d_heatmap(
    solution, t=final_time, marker_size=5, variable="Cell temperature [K]"
)
