#
# Compare lithium-ion battery models
#
import pybamm

pybamm.set_logging_level("INFO")

# load models
models = [
    pybamm.lithium_ion.SPMe(
        {"thermal": "lumped", "surface temperature": "ambient"},
        name="ambient surface temperature",
    ),
    pybamm.lithium_ion.SPMe(
        {"thermal": "lumped", "surface temperature": "lumped"},
        name="lumped surface temperature",
    ),
]

experiment = pybamm.Experiment(
    [
        "Discharge at 1C until 2.5V",
        "Rest for 1 hour",
    ]
)

parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update(
    {
        "Casing heat capacity [J.K-1]": 30,
        "Environment thermal resistance [K.W-1]": 10,
    },
    check_already_exists=False,
)

# create and run simulations
sols = []
for model in models:
    model.variables["Bulk temperature [°C]"] = (
        model.variables["Volume-averaged cell temperature [K]"] - 273.15
    )
    model.variables["Surface temperature [°C]"] = (
        model.variables["Surface temperature [K]"] - 273.15
    )
    sim = pybamm.Simulation(
        model, parameter_values=parameter_values, experiment=experiment
    )
    sol = sim.solve([0, 3600])
    sols.append(sol)

# plot
pybamm.dynamic_plot(
    sols,
    [
        "Voltage [V]",
        "Bulk temperature [°C]",
        "Surface temperature [°C]",
        "Surface total cooling [W]",
        "Environment total cooling [W]",
    ],
)
