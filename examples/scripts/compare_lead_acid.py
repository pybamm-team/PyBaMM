import pybamm
import numpy as np

# load models
leading_order_model = pybamm.lead_acid.LOQS()
composite_model = pybamm.lead_acid.Composite()

# create geometry
geometry = composite_model.default_geometry

# load parameter values and process models and geometry
param = composite_model.default_parameter_values
param.process_model(leading_order_model)
param.process_model(composite_model)
param.process_geometry(geometry)

# set mesh
mesh = pybamm.Mesh(
    geometry, composite_model.default_submesh_types, composite_model.default_var_pts
)

# discretise models
disc_loqs = pybamm.Discretisation(mesh, leading_order_model.default_spatial_methods)
disc_loqs.process_model(leading_order_model)
disc_comp = pybamm.Discretisation(mesh, composite_model.default_spatial_methods)
disc_comp.process_model(composite_model)

# solve model
solver_loqs = composite_model.default_solver
solver_comp = composite_model.default_solver
t_eval = np.linspace(0, 1, 100)
solver_loqs.solve(leading_order_model, t_eval)
solver_comp.solve(composite_model, t_eval)

# plot
models = [leading_order_model, composite_model]
solvers = [solver_loqs, solver_comp]
plot = pybamm.QuickPlot(models, param, mesh, solvers)
plot.dynamic_plot()
