import matplotlib.pyplot as plt
import pybamm

options = {"sei": "solvent-diffusion limited"}
model = pybamm.lithium_ion.DFN(options=options)

params = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
params.update({"Reference temperature [K]": 278.15})
params.update({"Ambient temperature [K]": 278.15})
params.update({"Initial temperature [K]": 278.15})
params.update({"Upper voltage cut-off [V]": 4.21})

periods = [10, 60]
markers = ["x", "o"]
for period, marker in zip(periods, markers):
    exp = pybamm.Experiment(
        ["Hold at 4.2V until C/100", "Rest for 4 hours (3 minute period)"]
        + [f"Discharge at 1C until 2.5V ({period} second period)"]
    )
    sim = pybamm.Simulation(model=model, parameter_values=params, experiment=exp)
    sim.solve()
    time = sim.solution.t
    voltage = sim.solution["Terminal voltage [V]"].entries
    plt.plot(
        time, voltage, linestyle="-", marker=marker, label=f"({period} second period"
    )
plt.legend()
plt.show()
