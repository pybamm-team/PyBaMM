import pybamm
import numpy as np

pybamm.set_logging_level("INFO")
model = pybamm.lithium_sulfur.MarinescuEtAl2016()

# Update current and ICs to correspond to initial 2.4V as in ref [2]
params = model.default_parameter_values
params.update(
    {
        "Current function [A]": 1.7,
        "Initial Condition for S8 ion [g]": 2.6730,
        "Initial Condition for S4 ion [g]": 0.0128,
        "Initial Condition for S2 ion [g]": 4.3321e-6,
        "Initial Condition for S ion [g]": 1.6321e-6,
        "Initial Condition for Precipitated Sulfur [g]": 2.7e-06,
        "Initial Condition for Terminal Voltage [V]": 2.4,
        "Shuttle rate coefficient during charge [s-1]": 0.0002,
        "Shuttle rate coefficient during discharge [s-1]": 0.0002,
    }
)

# Set up and solve simulation
sim = pybamm.Simulation(
    model,
    parameter_values=params,
    solver=pybamm.CasadiSolver(
        atol=1e-6, rtol=1e-3, extra_options_setup={"max_step_size": 0.1}
    ),
)
sim.solve(np.linspace(0, 6680, 668))

# Plot results
sim.plot(
    [
        "S8 [g]",
        "S4 [g]",
        "S2 [g]",
        "S [g]",
        "Precipitated Sulfur [g]",
        "High plateau current [A]",
        "Low plateau current [A]",
        "High plateau over-potential [V]",
        "Low plateau over-potential [V]",
        "High plateau potential [V]",
        "Low plateau potential [V]",
        "Terminal voltage [V]",
    ],
    time_unit="seconds",
)
