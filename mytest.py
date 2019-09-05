import pybamm
import numpy as np

model = pybamm.BaseModel()

x = pybamm.SpatialVariable("x", domain="SEI layer", coord_sys="cartesian")
c = pybamm.Variable("Solvent concentration", domain="SEI layer")

dcdt = pybamm.inner(x, pybamm.grad(c))

model.rhs = {c: dcdt}
model.boundary_conditions = {c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}}
model.initial_conditions = {c: pybamm.Scalar(1)}
model.variables = {"Solvent concentration": c}

geometry = {
    "SEI layer": {"primary": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}
}


submesh_types = {"SEI layer": pybamm.Uniform1DSubMesh}
var_pts = {x: 10}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

spatial_methods = {"SEI layer": pybamm.FiniteVolume}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

import ipdb

ipdb.set_trace()

solver = pybamm.ScipySolver()
t = np.linspace(0, 1, 100)
solution = solver.solve(model, t)
