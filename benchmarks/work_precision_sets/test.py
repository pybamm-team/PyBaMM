import pybamm
import numpy as np
import matplotlib.pyplot as plt

model = pybamm.BaseModel()
x = pybamm.Variable("x")

dxdt = x
model.initial_conditions = {x: pybamm.Scalar(1)}
model.rhs = {x: dxdt}
disc = pybamm.Discretisation()  # use the default discretisation
disc.process_model(model)
solver = pybamm.ScipySolver()
t = np.linspace(0, 1, 20)
solution = solver.solve(model, t)
print(solution.condition_number())