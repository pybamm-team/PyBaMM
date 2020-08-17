import pybamm

c = pybamm.Variable("c", domain=["negative electrode"])
model = pybamm.BaseModel()
# x = pybamm.SpatialVariableEdge("x", domain=["negative electrode"])
# v = 0.5 + 0.5 * x
v = pybamm.PrimaryBroadcastToEdges(1, ["negative electrode"])
model.rhs = {c: -pybamm.div(-pybamm.downwind(c) * v) + 2}
model.initial_conditions = {c: 0}
model.boundary_conditions = {c: {"left": (0, "Neumann"), "right": (0, "Dirichlet")}}
# model.boundary_conditions = {c: {"left": (0, "Dirichlet"), "right": (0, "Neumann")}}
model.variables = {"c": c}

spm = pybamm.lithium_ion.SPM()

var = pybamm.standard_spatial_vars

param = spm.default_parameter_values
geometry = {
    "negative electrode": {var.x_n: {"min": 0, "max": 1}},
    "current collector": {var.z: {"position": 1}},
}
submesh_types = spm.default_submesh_types
var_pts = spm.default_var_pts
spatial_methods = spm.default_spatial_methods

# load parameter values and process model and geometry
param.process_model(model)
param.process_geometry(geometry)

# set mesh
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

t_eval = [0, 100]
solution = pybamm.CasadiSolver().solve(model, t_eval)

# plot
plot = pybamm.QuickPlot(solution, ["c"])
plot.dynamic_plot()
