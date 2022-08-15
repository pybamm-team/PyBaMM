#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")

# load models
models = [
    # pybamm.lithium_ion.SPM({"particle phases": ("2", "1")}),
    # pybamm.lithium_ion.SPMe({"particle phases": ("2", "1")}),
    pybamm.lithium_ion.DFN({"particle phases": ("2", "1")}),
]

parameter_set = pybamm.parameter_sets.Chen2020
parameter_values = pybamm.ParameterValues(parameter_set)

for parameter in [
    "Negative electrode OCP [V]",
    "Negative electrode OCP entropic change [V.K-1]",
    "Maximum concentration in negative electrode [mol.m-3]",
    "Initial concentration in negative electrode [mol.m-3]",
    "Negative particle radius [m]",
    "Negative electrode diffusivity [m2.s-1]",
    "Negative electrode exchange-current density [A.m-2]",
    "Negative electrode active material volume fraction",
    "Negative electrode electrons in reaction",
]:
    parameter_values.update(
        {
            f"Primary: {parameter}": parameter_values[parameter],
            f"Secondary: {parameter}": parameter_values[parameter],
        },
        check_already_exists=False,
    )
    del parameter_values[parameter]

# create and run simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sim.solve([0, 3600])
    sims.append(sim)

# plot
pybamm.dynamic_plot(sims, ["Terminal voltage [V]"])
