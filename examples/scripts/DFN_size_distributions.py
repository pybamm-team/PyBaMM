#
# Compare the DFN with particle size distributions (MP-DFN)
# to the DFN with a single particle size, and the MPM
#
import pybamm

pybamm.set_logging_level("INFO")

# load models
models = [
    pybamm.lithium_ion.DFN(options={"particle size": "distribution"}, name="MP-DFN"),
    pybamm.lithium_ion.MPM(name="MPM"),
    pybamm.lithium_ion.DFN(name="DFN"),
]

# parameter set
params = pybamm.ParameterValues("Marquis2019")

# add distribution params (lognormals) with custom standard deviations
params = pybamm.get_size_distribution_parameters(params, sd_n=0.2, sd_p=0.4)

# experiment
experiment = pybamm.Experiment(
    [
        "Discharge at 1C for 3450 seconds",
        "Rest for 3600 seconds",
    ]
)

# solve
sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=params, experiment=experiment)
    sims.append(sim)

for sim in sims:
    sim.solve()

# plot MP-DFN variables
output_variables = [
    "Negative particle surface concentration distribution [mol.m-3]",
    "Positive particle surface concentration distribution [mol.m-3]",
    "Current [A]",
    "X-averaged negative area-weighted particle-size distribution [m-1]",
    "X-averaged positive area-weighted particle-size distribution [m-1]",
    "Voltage [V]",
]
sims[0].plot(output_variables)

# compare models (size-averaged concentrations)
output_variables = [
    "Negative particle surface concentration [mol.m-3]",
    "Positive particle surface concentration [mol.m-3]",
    "Current [A]",
    "Voltage [V]",
]
pybamm.dynamic_plot(sims, output_variables=output_variables)
