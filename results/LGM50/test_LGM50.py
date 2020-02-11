import pybamm
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8})

pybamm.set_logging_level("INFO")
experiment = pybamm.Experiment(
    [
        "Discharge at 1C until 2.5 V",
        "Rest for 2 hours",
    ],
    period="10 seconds",
)
filename = "1C"
model = pybamm.lithium_ion.DFN()
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)

cspmax = 50483 * 1.25  #1.25
csnmax = 29583 * 1.13  #1.13

param["Initial concentration in negative electrode [mol.m-3]"] = 0.90 * csnmax
param["Initial concentration in positive electrode [mol.m-3]"] = 0.26 * cspmax
param["Maximum concentration in negative electrode [mol.m-3]"] = csnmax
param["Maximum concentration in positive electrode [mol.m-3]"] = cspmax
param["Negative electrode Bruggeman coefficient (electrolyte)"] = 1.5
param["Positive electrode Bruggeman coefficient (electrolyte)"] = 1.5
param["Separator Bruggeman coefficient (electrolyte)"] = 1.5
param["Positive electrode diffusivity [m2.s-1]"] = 4E-15
param["Negative electrode diffusivity [m2.s-1]"] = 3.3E-14  # 1C
# param["Negative electrode diffusivity [m2.s-1]"] = 6.3E-14  # 1.5C
# param["Negative electrode diffusivity [m2.s-1]"] = 1.3E-14  # C/2

sim = pybamm.Simulation(
    model,
    parameter_values=param,
    experiment=experiment,
    solver=pybamm.CasadiSolver()
)
sim.solve()

# Show all plots
sim.plot()

# Compare with experiments
tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge)

data_experiments = pd.read_csv(
    "~/LGM50/data/data" + filename + "_rest.csv"
).to_numpy()

voltage = sim.solution["Terminal voltage [V]"]
time = sim.solution["Time [h]"]
Ueq = sim.solution["X-averaged battery open circuit voltage [V]"]

interpolated_voltage = interpolate.PchipInterpolator(
    time(sim.solution.t),
    voltage(sim.solution.t),
    extrapolate=True
)

error = np.absolute(
    interpolated_voltage(data_experiments[60:-1, 0] / 3600) - data_experiments[60:-1, 1]
)

rmse = np.sqrt(np.mean(np.square(error)))

print("RMSE: ", rmse)
print("Peak error: ", np.max(error))

fig10, axes10 = plt.subplots(1, 2, num=10, figsize=(6, 2.5))

axes10[0].fill_between(
    data_experiments[:, 0] / 3600,
    data_experiments[:, 1] + data_experiments[:, 2],
    data_experiments[:, 1] - data_experiments[:, 2],
    color="#808080",
    label="experiments"
)
axes10[0].plot(
    np.array([data_experiments[0, 0] / 3600, 0]),
    Ueq(sim.solution.t[0]) * np.ones(2),
    color="C1"
)
axes10[0].plot(time(sim.solution.t), voltage(sim.solution.t), color="C1", label="model")
axes10[0].set_xlabel("t [h]")
axes10[0].set_ylabel("Voltage [V]")
axes10[0].legend()

axes10[1].plot(data_experiments[60:-1, 0] / 3600, error)
axes10[1].set_xlabel("t [h]")
axes10[1].set_ylabel("Voltage error [V]")

plt.tight_layout()

plt.savefig(
    "/home/ferranbrosa/LGM50/figures/fig" + filename + ".png",
    dpi=300
)

plt.figure(num=1, figsize=(6, 4))
plt.fill_between(
    data_experiments[:, 0] / 3600,
    data_experiments[:, 1] + data_experiments[:, 2],
    data_experiments[:, 1] - data_experiments[:, 2],
    color="#808080",
    label="experiments"
)
plt.plot(
    np.array([data_experiments[0, 0] / 3600, 0]),
    Ueq(sim.solution.t[0]) * np.ones(2),
    color="C1"
)
plt.plot(time(sim.solution.t), voltage(sim.solution.t), color="C1", label="model")
plt.xlabel("t [h]")
plt.ylabel("Voltage [V]")
plt.legend()

plt.tight_layout()

plt.show()
