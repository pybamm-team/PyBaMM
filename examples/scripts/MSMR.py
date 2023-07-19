import pybamm

pybamm.set_logging_level("DEBUG")

model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
param = model.param
for i in range(6):
    xj = model.variables[f"Average x_n_{i}"]
    Xj = model.param.n.prim.X_j(i)
    model.variables[f"Xj - xj n_{i}"] = Xj - xj
for i in range(4):
    xj = model.variables[f"Average x_p_{i}"]
    Xj = model.param.p.prim.X_j(i)
    model.variables[f"Xj - xj p_{i}"] = Xj - xj
experiment = pybamm.Experiment(
    [
        (
            "Discharge at 1C until 3V",
            # "Rest for 1 hour",
            # "Charge at C/2 until 4.1 V",
            # "Hold at 4.1 V until 10 mA",
            # "Rest for 1 hour",
        ),
    ]
)
sim = pybamm.Simulation(model, experiment=experiment)
sim.solve(initial_soc=0.9)
xns = [
    f"Average x_n_{i}" for i in range(6)
]  # negative electrode reactions: x_n_0, x_n_1, ..., x_n_5
Xxns = [f"Xj - xj n_{i}" for i in range(6)]
xps = [
    f"Average x_p_{i}" for i in range(4)
]  # positive electrode reactions: x_p_0, x_p_1, ..., x_p_3
Xxps = [f"Xj - xj p_{i}" for i in range(4)]
sim.plot(
    [
        xns,
        Xxns,
        xps,
        Xxps,
        "Current [A]",
        "Negative electrode interfacial current density [A.m-2]",
        "Positive electrode interfacial current density [A.m-2]",
        "Voltage [V]",
    ]
)
