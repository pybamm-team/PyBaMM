#
# Compares the full and lumped thermal models for a single layer Li-ion cell
#
import pybamm

pybamm.set_logging_level("INFO")

# load models
full_thermal_model = pybamm.lithium_ion.SPMe(
    {"thermal": "x-full"}, name="full thermal model"
)
lumped_thermal_model = pybamm.lithium_ion.SPMe(
    {"thermal": "lumped"}, name="lumped thermal model"
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
# parameter as well as the "Cell cooling surface area [m2]" parameter. Since the "full"
# model only accounts for cooling from the large surfaces of the pouch, we set the
# "Surface area for cooling" parameter to the area of the large surfaces of the pouch,
# and the total heat transfer coefficient to 5 W.m-2.K-1
A = parameter_values["Electrode width [m]"] * parameter_values["Electrode height [m]"]
lumped_params = parameter_values.copy()
lumped_params.update(
    {
        "Total heat transfer coefficient [W.m-2.K-1]": 5,
        "Cell cooling surface area [m2]": 2 * A,
    }
)
params = [full_params, lumped_params]
# loop over the models and solve
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
