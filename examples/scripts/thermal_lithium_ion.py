#
# Compares the full and lumped thermal models for a single layer Li-ion cell
#
from __future__ import annotations

import pybamm

pybamm.set_logging_level("INFO")

# load models
# for the full model we use the "x-full" thermal submodel, which means that we solve
# the thermal model in the x-direction for a single-layer pouch cell
# for the lumped model we use the "arbitrary" cell geometry, which means that we can
# specify the surface area for cooling and total heat transfer coefficient
full_thermal_model = pybamm.lithium_ion.SPMe(
    {"thermal": "x-full"}, name="full thermal model"
)
lumped_thermal_model = pybamm.lithium_ion.SPMe(
    {"cell geometry": "arbitrary", "thermal": "lumped"}, name="lumped thermal model"
)
models = [full_thermal_model, lumped_thermal_model]

# load parameter values, we will use the Marquis2019 parameter set
parameter_values = pybamm.ParameterValues("Marquis2019")
# for the "full" model we use a heat transfer coefficient of 5 W.m-2.K-1 on the large
# surfaces of the pouch and zero heat transfer coefficient on the tabs and edges
full_params = parameter_values.copy()
full_params.update(
    {
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]"
        "": 5,
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]"
        "": 5,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 0,
        "Edge heat transfer coefficient [W.m-2.K-1]": 0,
    }
)
# for the lumped model we set the "Total heat transfer coefficient [W.m-2.K-1]"
# parameter as well as the "Cell cooling surface area [m2]" and "Cell volume [m3]
# parameters. Since the "full" model only accounts for cooling from the large surfaces
# of the pouch, we set the "Surface area for cooling [m2]" parameter to the area of the
# large surfaces of the pouch, and the total heat transfer coefficient to 5 W.m-2.K-1
A = parameter_values["Electrode width [m]"] * parameter_values["Electrode height [m]"]
contributing_layers = [
    "Negative current collector",
    "Negative electrode",
    "Separator",
    "Positive electrode",
    "Positive current collector",
]
total_thickness = sum(
    [parameter_values[layer + " thickness [m]"] for layer in contributing_layers]
)
electrode_volume = (
    total_thickness
    * parameter_values["Electrode height [m]"]
    * parameter_values["Electrode width [m]"]
)
lumped_params = parameter_values.copy()
lumped_params.update(
    {
        "Total heat transfer coefficient [W.m-2.K-1]": 5,
        "Cell cooling surface area [m2]": 2 * A,
        "Cell volume [m3]": electrode_volume,
    }
)

# loop over the models and solve
params = [full_params, lumped_params]
sols = []
for model, param in zip(models, params):
    sim = pybamm.Simulation(model, parameter_values=param)
    sim.solve([0, 3600])
    sols.append(sim.solution)

# plot
output_variables = [
    "Voltage [V]",
    "X-averaged cell temperature [K]",
    "Cell temperature [K]",
]
pybamm.dynamic_plot(sols, output_variables)
