#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")
parameter_values = pybamm.ParameterValues("Mohtat2020")
param = pybamm.LithiumIonParameters()

Vmin = 2.8
Vmax = 4.2
Cn = parameter_values.evaluate(param.C_n_init)
Cp = parameter_values.evaluate(param.C_p_init)
nLi = parameter_values.evaluate(param.n_Li_init)

print(Cn)
print(Cp)
print(nLi)
# load models
models = [
    pybamm.lithium_ion.SPM(),
    pybamm.lithium_ion.SPMe(),
    pybamm.lithium_ion.DFN(),
    pybamm.lithium_ion.NewmanTobias(),
]

# create and run simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model)
    sim.solve([0, 3600])
    sims.append(sim)

# plot
pybamm.dynamic_plot(sims)
