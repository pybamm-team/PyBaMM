"""Compare SPM, SPMe, and DFN zones in the same asymmetrically cooled stack.

The zones' electrochemistry sets how much heat each generates and how strongly
its current responds to temperature, so the three fidelities predict different
through-stack gradients and current splits for the same stack and cooling.
"""

import pybamm
from pybamm_model_zoo.multilayer_3d_thermal import (
    MultiLayer3DThermalDFN,
    MultiLayer3DThermalSPM,
    MultiLayer3DThermalSPMe,
)

NUM_PHYSICAL_LAYERS = 12
NUM_SUBDIVISIONS = 4

solutions = {}
for model_class in (
    MultiLayer3DThermalSPM,
    MultiLayer3DThermalSPMe,
    MultiLayer3DThermalDFN,
):
    model = model_class(NUM_PHYSICAL_LAYERS, NUM_SUBDIVISIONS)
    parameter_values = model.apply_stack_scaling(model.default_parameter_values)
    parameter_values.update(
        {
            model.CONTACT_RESISTANCE_PARAM: 5e-3,
            "Left face heat transfer coefficient [W.m-2.K-1]": 50.0,
            **{
                f"{face} face heat transfer coefficient [W.m-2.K-1]": 0.5
                for face in ("Right", "Front", "Back", "Bottom", "Top")
            },
        }
    )
    simulation = pybamm.Simulation(
        model,
        parameter_values=parameter_values,
        experiment=pybamm.Experiment(["Discharge at 3C until 2.8 V"]),
    )
    solutions[model_class.__name__.removeprefix("MultiLayer3DThermal")] = (
        simulation.solve()
    )

print(f"{NUM_PHYSICAL_LAYERS} unit cells in {NUM_SUBDIVISIONS} zones, 3C to 2.8 V")
print("  model   end [s]   T spread [K]   T max [K]")
for name, solution in solutions.items():
    end = solution.t[-1]
    print(
        f"  {name:5s}   {end:7.0f}   {solution['Temperature spread [K]'](end):12.3f}"
        f"   {solution['Maximum layer-averaged temperature [K]'](end):9.3f}"
    )

print("\nCurrent fraction half way through, zone 0 on the cold plate")
print("  zone " + "".join(f"{name:>10s}" for name in solutions))
for i in range(NUM_SUBDIVISIONS):
    fractions = "".join(
        f"{solution[f'Layer {i} current fraction'](solution.t[-1] / 2):10.5f}"
        for solution in solutions.values()
    )
    print(f"  {i:4d} {fractions}")
