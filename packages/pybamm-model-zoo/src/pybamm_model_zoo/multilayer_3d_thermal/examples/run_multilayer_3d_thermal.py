"""Cool a pouch stack through one face, and follow the current through it.

A cold plate on the left face and insulation everywhere else leave the stack
coolest next to the plate. Kinetics and diffusion are faster where it is warm,
so for most of the discharge the warm zones carry more of the current; they run
down sooner for it, and near the end hand the current back to the cool zones.
"""

import numpy as np

import pybamm
import pybamm_model_zoo as zoo

# 24 unit cells, lumped into 6 zones of 4.
model = zoo.load("MultiLayer3DThermalSPM")(num_physical_layers=24, num_subdivisions=6)
parameter_values = model.apply_stack_scaling(model.default_parameter_values)
parameter_values.update(
    {
        # An adhesive or air gap between layers, which lets a gradient build.
        model.CONTACT_RESISTANCE_PARAM: 1e-2,
        "Left face heat transfer coefficient [W.m-2.K-1]": 50.0,
        **{
            f"{face} face heat transfer coefficient [W.m-2.K-1]": 0.1
            for face in ("Right", "Front", "Back", "Bottom", "Top")
        },
    }
)
simulation = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    experiment=pybamm.Experiment(["Discharge at 3C until 2.8 V"]),
)
solution = simulation.solve()
end = solution.t[-1]
middle = end / 2

print(
    f"{model.num_subdivisions} zones of {model.layers_per_zone} unit cells, "
    f"3C to 2.8 V in {end:.0f} s; zone 0 is on the cold plate"
)
print("                        current fraction")
print(f"  zone   T end [K]   t = {middle:4.0f} s   t = {end:4.0f} s")
for i in range(model.num_subdivisions):
    temperature = solution[f"Layer {i} average temperature [K]"](end)
    fraction = solution[f"Layer {i} current fraction"]
    print(
        f"  {i:4d}   {temperature:9.3f}   {fraction(middle):10.5f}   "
        f"{fraction(end):10.5f}"
    )
print(f"temperature spread {solution['Temperature spread [K]'](end):.3f} K")

# Each zone's field is 3D, so it can be read anywhere inside the zone: here along
# the stack's centreline, through its thickness.
width = parameter_values["Electrode width [m]"]
height = parameter_values["Electrode height [m]"]
print("\n  x [mm]   T [K] on the centreline")
for i in range(model.num_subdivisions):
    temperature = solution[f"Layer {i} temperature [K]"]
    nodes = temperature.mesh.nodes
    x = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), 3)
    values = temperature(
        t=end, x=x, y=np.full_like(x, width / 2), z=np.full_like(x, height / 2)
    )
    for position, value in zip(x, np.ravel(values), strict=True):
        print(f"  {position * 1e3:6.3f}   {value:7.3f}")
