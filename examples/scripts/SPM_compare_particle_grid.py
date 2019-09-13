import pybamm
import numpy as np
import matplotlib.pyplot as plt

pybamm.set_logging_level("INFO")

# load model
model = pybamm.lithium_ion.SPM()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)

# set mesh
submesh_types = {
    "negative electrode": pybamm.Uniform1DSubMesh,
    "separator": pybamm.Uniform1DSubMesh,
    "positive electrode": pybamm.Uniform1DSubMesh,
    "negative particle": pybamm.Chebyshev1DSubMesh,
    "positive particle": pybamm.Chebyshev1DSubMesh,
    "current collector": pybamm.SubMesh0D,
}
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 9, var.r_p: 9}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 0.2, 100)
solution = model.default_solver.solve(model, t_eval)

# plot
r_n = mesh["negative particle"][0].edges
c_n = pybamm.ProcessedVariable(
    model.variables["X-average negative particle concentration [mol.m-3]"],
    solution.t,
    solution.y,
    mesh=mesh,
)
r_p = mesh["positive particle"][0].edges
c_p = pybamm.ProcessedVariable(
    model.variables["X-average positive particle concentration [mol.m-3]"],
    solution.t,
    solution.y,
    mesh=mesh,
)
import ipdb; ipdb.set_trace()
fig, ax = plt.subplots(figsize=(15, 8))
plt.tight_layout()
plt.subplot(121)
plt.plot(
    r_n,
    np.zeros_like(r_n),
    "ro",
    mesh["negative particle"][0].nodes,
    c_n(t=0.1, r=mesh["negative particle"][0].nodes),
    "b-",
)
plt.subplot(122)
plt.plot(
    r_p,
    np.zeros_like(r_p),
    "ro",
    mesh["positive particle"][0].nodes,
    c_p(t=0.1, r=mesh["positive particle"][0].nodes),
    "b-",
)
plt.show()
