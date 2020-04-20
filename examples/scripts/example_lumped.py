import pybamm
import numpy as np
import matplotlib.pyplot as plt


pybamm.set_logging_level("INFO")

C_rate = 5

options = {
    "thermal": "x-lumped",
    # "current collector": "potential pair",
    # "dimensionality": 2,
}
dfn_1D = pybamm.lithium_ion.DFN(options=options)

options = {
    "thermal": "x-lumped",
    "current collector": "potential pair",
    "dimensionality": 1,
}
dfn_1p1D = pybamm.lithium_ion.DFN(options=options)

options = {
    "thermal": "x-lumped",
    "current collector": "potential pair",
    "dimensionality": 2,
}
dfn_2p1D = pybamm.lithium_ion.DFN(options=options)

models = {"DFN 1D": dfn_1D, "DFN 1+1D": dfn_1p1D, "DFN 2+1D": dfn_2p1D}

solutions = {}
other_vars = {}

for model_name, model in models.items():

    var = pybamm.standard_spatial_vars
    var_pts = {
        var.x_n: 3,
        var.x_s: 3,
        var.x_p: 3,
        var.r_n: 3,
        var.r_p: 3,
        var.y: 5,
        var.z: 5,
    }

    # var_pts = None

    chemistry = pybamm.parameter_sets.NCA_Kim2011
    parameter_values = pybamm.ParameterValues(chemistry=chemistry)

    parameter_values.update(
        {
            "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 0,
            "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 0,
            "Negative tab heat transfer coefficient [W.m-2.K-1]": 0,
            "Positive tab heat transfer coefficient [W.m-2.K-1]": 0,
            "Edge heat transfer coefficient [W.m-2.K-1]": 500,
            "Negative current collector thermal conductivity [W.m-1.K-1]": 267.467
            * 100000,
            "Positive current collector thermal conductivity [W.m-1.K-1]": 158.079
            * 100000,
            "Negative current collector conductivity [S.m-1]": 1e10,
            "Positive current collector conductivity [S.m-1]": 1e10,
        }
    )

    solver = pybamm.CasadiSolver(mode="fast")
    sim = pybamm.Simulation(
        model,
        var_pts=var_pts,
        solver=solver,
        parameter_values=parameter_values,
        C_rate=C_rate,
    )
    t_eval = np.linspace(0, 3500 / 6, 100)
    sim.solve(t_eval=t_eval)

    solutions[model_name] = sim.solution

    av = sim.solution["Volume-averaged cell temperature [K]"].entries

    if model_name == "DFN 2+1D":
        cell_temp = sim.solution["X-averaged cell temperature [K]"].entries
        max_temp = np.max(np.max(cell_temp, axis=0), axis=0)
        min_temp = np.min(np.min(cell_temp, axis=0), axis=0)

    elif model_name == "DFN 1+1D":
        cell_temp = sim.solution["X-averaged cell temperature [K]"].entries
        max_temp = np.max(cell_temp, axis=0)
        min_temp = np.min(cell_temp, axis=0)

    elif model_name == "DFN 1D":
        max_temp = sim.solution["Volume-averaged cell temperature [K]"].entries
        min_temp = sim.solution["Volume-averaged cell temperature [K]"].entries

    other_vars[model_name] = {
        "Time [s]": sim.solution["Time [s]"].entries,
        "Max temperature [K]": max_temp,
        "Min temperature [K]": min_temp,
        "Volume-averaged cell temperature [K]": av,
    }

# sim.plot(["X-averaged cell temperature [K]"])

plot = pybamm.QuickPlot(
    list(solutions.values()), output_variables=["Volume-averaged cell temperature [K]"]
)
plot.dynamic_plot()

fig, ax = plt.subplots(1, 3)

for i, model_name in enumerate(list(models.keys())):
    ax[0].plot(
        other_vars[model_name]["Time [s]"],
        other_vars[model_name]["Max temperature [K]"],
        label=model_name,
    )
    ax[1].plot(
        other_vars[model_name]["Time [s]"],
        other_vars[model_name]["Min temperature [K]"],
        label=model_name,
    )
    ax[2].plot(
        other_vars[model_name]["Time [s]"],
        other_vars[model_name]["Volume-averaged cell temperature [K]"],
        label=model_name,
    )


ax[2].legend()

plt.show()
