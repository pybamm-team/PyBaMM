
import pandas as pd
import pybamm
import numpy as np
import os

pybamm.set_logging_level("INFO")

# Import drive cycle from file
data_loader = pybamm.DataLoader()
drive_cycle_path = data_loader.get_data('US06.csv')
drive_cycle_current = pd.read_csv(
    drive_cycle_path, comment="#", header=None
).to_numpy()

# Map Drive Cycle
def map_drive_cycle(x, min_op_value, max_op_value):
    min_ip_value = x[:, 1].min()
    max_ip_value = x[:, 1].max()
    x[:, 1] = (x[:, 1] - min_ip_value) / (max_ip_value - min_ip_value) * (
        max_op_value - min_op_value
    ) + min_op_value
    return x

# Map current drive cycle to voltage and power
drive_cycle_power = map_drive_cycle(drive_cycle_current.copy(), 1.5, 3.5)

# Define the experiment
# 1. Calendar Ageing: Rest for 2 years
# 2. Drive Cycle: Charge, Hold, Rest, Drive (current), Rest, Drive (power), Rest
experiment = pybamm.Experiment(
    [
        "Rest for 52560 hours",
        (
            "Charge at 1 A until 4.0 V",
            "Hold at 4.0 V until 50 mA",
            "Rest for 30 minutes",
            pybamm.step.current(drive_cycle_current),
            "Rest for 30 minutes",
            pybamm.step.power(drive_cycle_power),
            "Rest for 30 minutes",
        ),
    ],
    period="1 hour",  # Period for reporting results (optional but good for long sims)
)

# Use the DFN model
model = pybamm.lithium_ion.DFN({"SEI": "reaction limited"})

# Reduce aging rate
param = pybamm.ParameterValues("Chen2020")
param["SEI kinetic rate constant [m.s-1]"] = 1e-13

# Create the simulation
solver = pybamm.CasadiSolver(mode="safe")
sim = pybamm.Simulation(model, experiment=experiment, solver=solver, parameter_values=param)

# Solve the simulation
sim.solve()

# Show all plots
# Show all plots
import matplotlib.pyplot as plt

# 1. Electrical Variables
plt.figure()
plot_electrical = sim.plot(
    [
        "Voltage [V]",
        "Current [A]",
        "Terminal power [W]",
    ],
    show_plot=True
)
plot_electrical.fig.savefig("plot_electrical.png")

# 2. Particle Variables
plt.figure()
plot_particle = sim.plot(
    [
        "Negative particle surface concentration",
        "X-averaged negative particle surface concentration",
        "Electrolyte concentration [mol.m-3]",
    ],
    show_plot=False
)
plot_particle.fig.savefig("plot_particle.png")

# 3. Ageing Variables
plt.figure()
plot_ageing = sim.plot(
    [
        "Negative SEI concentration [mol.m-3]",
        "X-averaged negative SEI concentration [mol.m-3]",
        "X-averaged negative SEI thickness [m]",
        "Loss of lithium inventory [%]",
        ["Total lithium lost [mol]", "Loss of lithium to negative SEI [mol]"],
    ],
    show_plot=False
)
plot_ageing.fig.savefig("plot_ageing.png")

print("Plots saved: plot_electrical.png, plot_particle.png, plot_ageing.png")
