import pybamm


model = pybamm.BaseModel()
# r = pybamm.SpatialVariable("r", domain="domain", coord_sys=coord_sys)
# u = pybamm.Variable("u", domain="domain")
# del_u = pybamm.div(pybamm.grad(u))
# model.boundary_conditions = {
#     u: {
#         "left": (pybamm.Scalar(0), "Dirichlet"),
#         "right": (pybamm.Scalar(1), "Dirichlet"),
#     }
# }
# model.algebraic = {u: del_u}
# model.initial_conditions = {u: pybamm.Scalar(0)}
# model.variables = {"u": u, "r": r}
# geometry = {"domain": {r: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)}}}
# submesh_types = {"domain": pybamm.Uniform1DSubMesh}
# var_pts = {r: 500}
# mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
# spatial_methods = {"domain": pybamm.FiniteVolume()}
# disc = pybamm.Discretisation(mesh, spatial_methods)
# disc.process_model(model)
# solver = pybamm.CasadiAlgebraicSolver()
