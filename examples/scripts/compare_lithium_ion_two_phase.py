#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")

# load models
models = [
    # pybamm.lithium_ion.SPM({"particle phases": ("2", "1")}),
    pybamm.lithium_ion.SPMe({"particle phases": ("2", "1")}),
    # pybamm.lithium_ion.DFN({"particle phases": ("2", "1")}),
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
parameter_values.update(
    {
        "Primary: Negative electrode active material volume fraction": parameter_values[
            "Negative electrode active material volume fraction"
        ]
        * 0.1,
        "Secondary: Negative electrode active material volume "
        "fraction": parameter_values[
            "Negative electrode active material volume fraction"
        ]
        * 0.9,
    },
    check_already_exists=False,
)
del parameter_values["Negative electrode active material volume fraction"]

# print(parameter_values.evaluate(models[0].param.n.prim.a_R))
# print(parameter_values.evaluate(models[0].param.n.sec.a_R))

# create and run simulations
sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sim.solve([0, 3600])
    sims.append(sim)

# plot
pybamm.dynamic_plot(
    sims,
    [
        "Terminal voltage [V]",
        [
            "X-averaged negative electrode primary active material volume fraction",
            "X-averaged negative electrode secondary active material volume fraction",
            "X-averaged negative electrode active material volume fraction",
        ],
        [
            "Average negative primary particle concentration",
            "Average negative secondary particle concentration",
        ],
        [
            "X-averaged negative electrode primary interfacial current density",
            "X-averaged negative electrode secondary interfacial current density",
            "X-averaged negative electrode interfacial current density",
        ],
        [
            "Negative electrode primary interfacial current density",
            "Negative electrode secondary interfacial current density",
            "Negative electrode interfacial current density",
        ],
        [
            "X-averaged negative electrode primary volumetric interfacial current density",
            "X-averaged negative electrode secondary volumetric interfacial current density",
            "X-averaged negative electrode volumetric interfacial current density",
        ],
        "Current [A]",
    ],
)
