#
# Simulate user-defined current profile
#
import pybamm
import numpy as np


# create user-defined current class
class current:
    def __init__(self, amplitude=1, frequency=1):
        self.amplitude = amplitude
        self.frequency = frequency

    def __call__(self, t):
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t)


# choose frequencies
frequencies = [0.1, 1]

# load models (need to create new instances of model, not copies)
pybamm.set_logging_level("INFO")
models = [None] * len(frequencies)
for i in range(len(frequencies)):
    models[i] = pybamm.lithium_ion.SPM()

# load parameter values and process models
param = models[0].default_parameter_values
for i, frequency in enumerate(frequencies):
    param.update({"Current function": current(frequency=frequency)})
    param.process_model(models[i])

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}

# discretise models
for model in models:
    # create geometry
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

# Example: simulate for 30 seconds
simulation_time = 30  # end time in seconds
tau = param.process_symbol(
    pybamm.standard_parameters_lithium_ion.tau_discharge
).evaluate(0)

# loop over frequencies
solutions = [None] * len(frequencies)
labels = [None] * len(frequencies)
for i, frequency in enumerate(frequencies):
    param["Current function"] = current(frequency=frequency)
    param.update_model(model, disc)

    # need enough timesteps to resolve output
    npts = 20 * simulation_time * frequency
    t_eval = np.linspace(0, simulation_time / tau, npts)
    solutions[i] = model.default_solver.solve(model, t_eval)
    labels[i] = "Frequency: {} Hz".format(frequency)

# plot
output_variables = ["Current [A]", "Terminal voltage [V]"]
plot = pybamm.QuickPlot(models, mesh, solutions, output_variables, labels)
plot.dynamic_plot()
