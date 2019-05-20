import pybamm
import numpy as np

# load model
model = pybamm.lithium_ion.DFN()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values

V_local = pybamm.Parameter("Local potential difference")
param.update({"Local potential difference": 3.0})  # add parameter value for local V
model.boundary_conditions[model.variables["Positive electrode potential"]][
    "right"
] = (V_local, "Dirichlet")

param.process_model(model)
param.process_geometry(geometry)


# set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
solver = model.default_solver
t_eval = np.linspace(0, 2, 100)
solver.solve(model, t_eval)

# plot
plot = pybamm.QuickPlot(model, mesh, solver)
plot.dynamic_plot()
