import pybamm
import matplotlib.pyplot as plt

x = pybamm.linspace(-1, 1, 100000)
rhs = pybamm.softplus(0, x, 100)
plt.plot(x.evaluate(), rhs.evaluate(), x.evaluate(), (x > 0).evaluate())
plt.show()


pybamm.set_logging_level("DEBUG")
disc = pybamm.Discretisation()
solver = pybamm.CasadiSolver("fast with events")
# solver = pybamm.CasadiSolver("safe")
# solver = pybamm.ScipySolver()

model1 = pybamm.BaseModel()
x = pybamm.Variable("x")
rhs = -pybamm.sqrt(x) * pybamm.sigmoid(0.5, x, 30)
model1.rhs = {x: rhs}
model1.initial_conditions = {x: 1}
model1.variables = {"x": x, "rhs": rhs}
model1.events = [pybamm.Event("x=0", x > 0)]
disc.process_model(model1)
sol1 = solver.solve(model1, t_eval=[0, 1.999])

model2 = pybamm.BaseModel()
x = pybamm.Variable("x")
rhs = -pybamm.sqrt(x)
model2.rhs = {x: rhs}
model2.initial_conditions = {x: 1}
model2.variables = {"x": x, "rhs": rhs}
disc.process_model(model2)
sol2 = pybamm.ScipySolver().solve(model2, t_eval=[0, 1.999])

pybamm.dynamic_plot([sol1, sol2], ["x", "rhs"])

# from casadi import *

# x = MX.sym("x", 1)
# # Two states

# # Expression for ODE right-hand side
# sigmoid = (1 + tanh(10 * (x - 0.01))) / 2
# k = 100
# kx = k * x
# smooth_abs = x * (exp(kx) - exp(-kx)) / (exp(kx) + exp(-kx))
# rhs = -sqrt(smooth_abs) * sigmoid

# ode = {}  # ODE declaration
# ode["x"] = x  # states
# ode["ode"] = rhs  # right-hand side

# # Construct a Function that integrates over 4s
# F = integrator("F", "cvodes", ode, {"tf": 10, "regularity_check": False})

# # Start from x=1
# res = F(x0=1)
# print(res["xf"])
