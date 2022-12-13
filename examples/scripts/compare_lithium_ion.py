#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")
options = {"interface utilisation": "constant"}
# load models
models = [
    pybamm.lithium_ion.SPM(),
    # pybamm.lithium_ion.SPM(options),
    # pybamm.lithium_ion.SPMe(options),
    # pybamm.lithium_ion.DFN(options),
    # pybamm.lithium_ion.NewmanTobias(),
]


parameter_values = pybamm.ParameterValues("Marquis2019")
parameter_values.update(
    {
        "Initial negative electrode interface utilisation": 0.9,
        "Initial positive electrode interface utilisation": 0.8,
    },
    check_already_exists=False,
)
sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sim.solver.atol = 1e-8
    sim.solver.rtol = 1e-8
    sol = sim.solve([0, 3600])
    sims.append(sim)
    # print(sol.y[:, -1])
    print(sim.built_model.concatenated_rhs.evaluate(y=sol.y[:, -1].full()))

# plot
pybamm.dynamic_plot(
    sims,
    [
        "Total lithium in particles [mol]",
        # "X-averaged negative particle concentration [mol.m-3]",
        # "X-averaged positive particle concentration [mol.m-3]",
        # "Negative electrode filling fraction",
        # "Positive electrode filling fraction",
        # "Negative electrode interfacial current density [A.m-2]",
        # "Current collector current density [A.m-2]",
        # "Electrolyte current density [A.m-2]",
    ],
)
