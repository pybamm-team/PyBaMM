from __future__ import annotations

import pybamm

pybamm.set_logging_level("INFO")

# Use the MSMR model, with 6 negative electrode reactions and 4 positive electrode
# reactions
msmr_model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})

# We can also use a SPM with MSMR thermodynamics, transport and kinetics by changing
# model options. Note we need to se the "surface form" to "algebraic" or "differential"
# to use the MSMR, since we cannot explicitly invert the kinetics
spm_msmr_model = pybamm.lithium_ion.SPM(
    {
        "number of MSMR reactions": ("6", "4"),
        "open-circuit potential": "MSMR",
        "particle": "MSMR",
        "intercalation kinetics": "MSMR",
        "surface form": "algebraic",
    },
    name="Single Particle MSMR",
)

# Load in the example MSMR parameter set
parameter_values = pybamm.ParameterValues("MSMR_Example")

# Define an experiment
experiment = pybamm.Experiment(
    [
        (
            "Discharge at 1C for 1 hour or until 3 V",
            "Rest for 1 hour",
            "Charge at C/3 until 4.2 V",
            "Hold at 4.2 V until 10 mA",
            "Rest for 1 hour",
        ),
    ]
)

# Loop over the models, creating and solving a simulation
sols = []
for model in [msmr_model, spm_msmr_model]:
    sim = pybamm.Simulation(
        model, parameter_values=parameter_values, experiment=experiment
    )
    sol = sim.solve(initial_soc=0.9)
    sols.append(sol)

# Plot the fractional occupancy x_j of the individual MSMR reactions, along with some
# other variables of interest
xns = [
    f"Average x_n_{i}" for i in range(6)
]  # negative electrode reactions: x_n_0, x_n_1, ..., x_n_5
xps = [
    f"Average x_p_{i}" for i in range(4)
]  # positive electrode reactions: x_p_0, x_p_1, ..., x_p_3
pybamm.dynamic_plot(
    sols,
    [
        xns,
        xps,
        "Current [A]",
        "Negative electrode interfacial current density [A.m-2]",
        "Positive electrode interfacial current density [A.m-2]",
        "Negative particle surface concentration [mol.m-3]",
        "Positive particle surface concentration [mol.m-3]",
        "Voltage [V]",
    ],
)
