import pybamm

model1 = pybamm.equivalent_circuit.Thevenin(
    options={"diffusion element": "false"}
)  # Model1: ECM Model definition
model2 = pybamm.equivalent_circuit.Thevenin(
    options={"diffusion element": "true"}
)  # Model2: ECMD Model definition

parameter_values1 = model1.default_parameter_values
parameter_values2 = model2.default_parameter_values
parameter_values2.update(
    {"Diffusion time constant [s]": 580}, check_already_exists=False
)

experiment = pybamm.Experiment(
    [
        (
            "Discharge at C/10 for 10 hours or until 3.3 V",
            "Rest for 30 minutes",
            "Rest for 2 hours",
            "Charge at 100 A until 4.1 V",
            "Hold at 4.1 V until 5 A",
            "Rest for 30 minutes",
            "Rest for 1 hour",
        ),
    ]
)

models = [
    model1,
    model2,
]

sims = []
for model, parameter_values in zip(
    [model1, model2], [parameter_values1, parameter_values2]
):
    sim = pybamm.Simulation(
        model, experiment=experiment, parameter_values=parameter_values
    )
    sim.solve()  # Solve for specific time range (0 to 3600 seconds)
    sims.append(sim)

labels = [
    "Thevenin (No Diffusion)",
    "Thevenin (Diffusion, τ=580s)",  # Adjust labels as needed
]
pybamm.dynamic_plot(sims, time_unit="seconds", labels=labels)

default_params = model1.default_parameter_values
nominal_capacity = default_params["Nominal cell capacity [A.h]"]
print(f"Nominal Cell Capacity (Ah): {nominal_capacity}")

# Second plot to compare SoC and Surface SoC
output_variables = ["SoC", "Surface SoC"]

sim_diffusion = sims[1]  # Assuming model2 is at index 1 in sims list
sim_diffusion.plot(output_variables=output_variables)
