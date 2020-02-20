import pybamm
import numpy as np
import matplotlib.pyplot as plt

pybamm.set_logging_level("INFO")

# load model
options = {"thermal": "isothermal"}

models = [
    pybamm.lithium_ion.SPM(options),
    pybamm.lithium_ion.SPMe(options),
    pybamm.lithium_ion.DFN(options),
]
solvers = [
    pybamm.ScikitsOdeSolver(),
    pybamm.ScikitsOdeSolver(),
    pybamm.ScikitsDaeSolver()
]

labels = ["SPM", "SPMe", "DFN"]

# load parameter values and process model and geometry
param = models[0].default_parameter_values
for model in models:
    param.process_model(model)
    model.events.pop("Zero electrolyte concentration cut-off")

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 10, var.r_p: 10}

# discretise model
discs = [None] * len(models)
for i, model in enumerate(models):
    # create geometry
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    discs[i] = disc


# solve model
t_eval = np.linspace(0, 1.2, 2)
C_rates = np.linspace(0.01, 50, 100)
capacities = np.zeros((len(models), C_rates.size))
times = np.zeros_like(capacities)

for i, C_rate in enumerate(C_rates):
    for j, model in enumerate(models):
        param.update(
                {
                    "C-rate": C_rate,
                    "Current function": "[constant]",
                }
            )
        param.update_model(model, discs[j])

        # solver = pybamm.ScikitsDaeSolver()
        solver = solvers[j]
        solver.rtol = 1e-6
        solver.atol = 1e-6
        solution = solver.solve(model, t_eval)
        current = model.variables["Current [A]"].evaluate(solution.t)
        tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge).evaluate()
        time =  solution.t_event * tau

        capacities[j, i] = time * current
        times[j, i] = time

plt.figure(1)
for i, model in enumerate(models):
    plt.plot(
        C_rates, capacities[i, :],
        label=labels[i],
        color="C{}".format(i)
    )
plt.xlabel('C-rate')
plt.ylabel('Capacity [C]')
plt.legend()

plt.figure(2)
for i, model in enumerate(models):
    plt.plot(
        C_rates, times[i, :],
        label=labels[i],
        color="C{}".format(i)
    )
plt.xlabel('C-rate')
plt.ylabel('Discharge time [s]')
plt.legend()

plt.show()
