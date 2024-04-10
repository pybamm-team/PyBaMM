import pybamm


model = pybamm.BaseModel()
x = pybamm.SpatialVariable("x", domain="domain", coord_sys="cartesian")
u = pybamm.Variable("u", domain="domain")
u_an = x + (pybamm.t - x) * ((x - pybamm.t) > 0)
v = pybamm.PrimaryBroadcastToEdges(1, ["domain"])
rhs = -pybamm.div(pybamm.upwind(u) * v) + 1
# rhs = - pybamm.div(u * v) + 1
model.boundary_conditions = {
    u: {
        "left": (pybamm.Scalar(0), "Dirichlet"),
    }
}
model.rhs = {u: rhs}
model.initial_conditions = {u: pybamm.Scalar(0)}
model.variables = {"u": u, "x": x, "analytical": u_an}
geometry = {"domain": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}
submesh_types = {"domain": pybamm.Uniform1DSubMesh}
N = 5
var_pts = {x: N}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
spatial_methods = {"domain": pybamm.FiniteVolume()}
disc = pybamm.Discretisation(mesh, spatial_methods)
model_disc = disc.process_model(model, inplace=False)

# Fix the entries of the discretised system (uncomment to get original behaviour)
# for i in range(N - 1):
#     model_disc.rhs[u].children[1].children[0].entries[i + 1, i] = 0
#     model_disc.rhs[u].children[1].children[0].entries[i + 1, i + 1] = -N
#     model_disc.rhs[u].children[1].children[0].entries[i + 1, i + 2] = N
# model_disc.rhs[u].children[1].children[0].entries[0, 0] = - 2 * N
# model_disc.rhs[u].children[1].children[0].entries[0, 1] = 2 * N


print(model_disc.rhs[u].children[1].children[0].entries.toarray())
solver = pybamm.CasadiSolver()
solution = solver.solve(model_disc, [0, 10])

plot = pybamm.QuickPlot(solution, [["u", "analytical"]])
plot.dynamic_plot()
