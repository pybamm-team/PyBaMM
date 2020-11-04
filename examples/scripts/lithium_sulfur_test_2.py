import pybamm
import numpy as np
import matplotlib.pyplot as plt

model = pybamm.lithium_sulfur.MarinescuEtAl2016()

# Update current and ICs for second model to correspond to initial 2.4V as in ref [2]
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
        # "Shuttle rate coefficient during discharge [s-1]": 1e-10,
        "Shuttle rate coefficient during discharge [s-1]": 0.0002,
    }
)

# Set up simulation
sim = pybamm.Simulation(
    model, parameter_values=params, solver=pybamm.ScikitsDaeSolver(atol=1e-6, rtol=1e-3)
)

# set up figure
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
ax[0].set_xlim([0, 3.5])
ax[0].set_ylim([2.1, 2.4])
ax[0].set_xlabel("Discharge capacity [A.h]")
ax[0].set_ylabel("Terminal voltage [V]")
ax[1].set_xlim([0.85, 1.15])
ax[1].set_ylim([0, 0.15])
ax[1].set_xlabel("Discharge capacity [A.h]")
ax[1].set_ylabel("Precipitated sulfur [g]")


# step solution
tstep = 4000  # going to fix npts = tpts, but not sure this is necessary
tmin = 10  # minimum step 10s

while tstep > tmin:
    try:
        solution = sim.step(tstep, npts=int(tstep))
        DC = solution["Discharge capacity [A.h]"].entries
        V = solution["Terminal voltage [V]"].entries
        ax[0].plot(DC[-1], V[-1], "ro")
    except pybamm.SolverError:
        tstep = tstep / 2

# extract variables and plot
DC = solution["Discharge capacity [A.h]"].entries
V = solution["Terminal voltage [V]"].entries
S = solution["Precipitated Sulfur [g]"].entries

print("Final time {}s".format(solution.t[-1]))
print("Final discharge capacity {} Ah".format(DC[-1]))

ax[0].plot(DC, V)
ax[1].plot(DC, S)

# plot results for updated parameters
sim.plot(
    [
        ["S8 [g]", "S4 [g]", "S2 [g]", "S [g]", "Precipitated Sulfur [g]"],
        ["High plateau current [A]", "Low plateau current [A]"],
        ["High plateau over-potential [V]", "Low plateau over-potential [V]"],
        [
            "High plateau potential [V]",
            "Low plateau potential [V]",
            "Terminal voltage [V]",
        ],
    ],
    time_unit="seconds",
)
