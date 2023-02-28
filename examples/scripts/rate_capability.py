import pybamm
import numpy as np
import matplotlib.pyplot as plt

pybamm.set_logging_level("INFO")

# load model
model = pybamm.lithium_ion.DFN()

# solve model
C_rates = np.linspace(0.05, 5, 20)
capacities = np.zeros_like(C_rates)
currents = np.zeros_like(C_rates)
voltage_av = np.zeros_like(C_rates)

for i, C_rate in enumerate(C_rates):
    experiment = pybamm.Experiment(
        ["Discharge at {:.4f}C until 3.2V".format(C_rate)],
        period="{:.4f} seconds".format(10 / C_rate),
    )
    sim = pybamm.Simulation(model, experiment=experiment, solver=pybamm.CasadiSolver())
    sim.solve()

    time = sim.solution["Time [s]"].entries
    capacity = sim.solution["Discharge capacity [A.h]"]
    current = sim.solution["Current [A]"]
    voltage = sim.solution["Voltage [V]"]

    capacities[i] = capacity(time[-1])
    currents[i] = current(time[-1])
    voltage_av[i] = np.mean(voltage(time))

plt.figure(1)
plt.scatter(C_rates, capacities)
plt.xlabel("C-rate")
plt.ylabel("Capacity [Ah]")

plt.figure(2)
plt.scatter(currents * voltage_av, capacities * voltage_av)
plt.xlabel("Power [W]")
plt.ylabel("Energy [Wh]")

plt.show()
