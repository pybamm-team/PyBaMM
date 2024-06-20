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

parameter_values = pybamm.ParameterValues("ORegan2022")
parameter_values.update(
    {
        "External volumetric heat capacity [J.m-3.K-1]": 3e6,
        "External volumetric thermal resistance [K.W-1.m-3]": 5e-4,
    },
    check_already_exists=False,
)
# create and run simulations
sols = []
for model in models:
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
        "Volume-averaged cell temperature [K]",
        "Surface temperature [K]",
        "Lumped total cooling [W.m-3]",
        "External total cooling [W.m-3]",
    ],
)
