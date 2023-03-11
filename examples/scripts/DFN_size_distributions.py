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

# discharge and relaxation: define current function
t_cutoff = 3450  # [s]
t_rest = 3600  # [s]
I_typ = params["Nominal cell capacity [A.h]"]  # cell capacity gives current for 1C


def current(t):
    return I_typ * pybamm.EqualHeaviside(t, t_cutoff)


params.update({"Current function [A]": current})
t_eval = [0, t_cutoff + t_rest]

# solve
solver = pybamm.CasadiSolver(mode="fast")
sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=params, solver=solver)
    sims.append(sim)

for sim in sims:
    sim.solve(t_eval=t_eval)

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
