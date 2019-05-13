import pybamm

model = pybamm.BaseModel()
whole_cell = ["negative electrode", "separator", "positive electrode"]
c = pybamm.Variable("c", domain=whole_cell)
d = pybamm.Variable("d", domain=whole_cell)
model.rhs = {c: 5 * pybamm.div(pybamm.grad(d)) - 1, d: -c}
model.initial_conditions = {c: 1, d: 2}
model.boundary_conditions = {
    c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
    d: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
}
model._variables = {
    "something": None,
    "something else": c,
    "another thing": None,
}

print(model._variables)
model.check_well_posedness()
print(model._variables)
