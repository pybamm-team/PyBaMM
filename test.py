import numpy as np
import matplotlib.pyplot as plt
import pybamm

# pts = np.linspace(1.0, 8.0, 100)

# v1 = 1.0
# v2 = 3.0


# def h_exact(x):
#     return (x <= 5) * v1 + (x > 5) * v2


# def h_smooth(x, k):
#     return (v2 - v1) * np.tanh(k * (x - 5)) / 2 + (v1 + v2) / 2


# plt.plot(pts, [h_exact(x) for x in pts], lw=3)
# plt.plot(pts, [h_smooth(x, 1) for x in pts], ":")
# plt.plot(pts, [h_smooth(x, 10) for x in pts], ":")
# plt.plot(pts, [h_smooth(x, 100) for x in pts], ":")
# plt.show()


# v1 = 1
# v2 = 2
# switch = 1


# def rhs_exact(x):
#     return (x <= switch) * v1 + (x > switch) * v2


# def rhs_smooth(x, k):
#     return (v2 - v1) * pybamm.tanh(k * (x - switch)) / 2 + (v1 + v2) / 2


# def rhs_exact(x):
#     return (x <= switch) * v1 + (x > switch) * v2


# def rhs_smooth(x, k):
#     return smooth_heaviside(x, switch, k) * v1 + smooth_heaviside(switch, x, k) * v2
#     # return (v2 - v1) * pybamm.tanh(k * (x - switch)) / 2 + (v1 + v2) / 2


def smooth_maximum(left, right, k):
    return (left * pybamm.exp(k * left) + right * pybamm.exp(k * right)) / (
        pybamm.exp(k * left) + pybamm.exp(k * right)
    )


def rhs_exact(x):
    return pybamm.maximum(x, 1)


def rhs_smooth(x, k):
    return pybamm.smooth_maximum(x, 1, k)


pts = pybamm.linspace(0, 2, 100)

plt.plot(pts.evaluate(), rhs_exact(pts).evaluate())
plt.plot(pts.evaluate(), rhs_smooth(pts, 1).evaluate(), ":")
plt.plot(pts.evaluate(), rhs_smooth(pts, 10).evaluate(), ":")
plt.plot(pts.evaluate(), rhs_smooth(pts, 100).evaluate(), ":")
plt.show()


x = pybamm.Variable("x")
y = pybamm.Variable("y")

model_exact = pybamm.BaseModel(name="exact")
model_exact.rhs = {x: rhs_exact(x)}
model_exact.initial_conditions = {x: 0.5}
model_exact.variables = {"x": x, "rhs": rhs_exact(x)}

model_smooth = pybamm.BaseModel(name="smooth")
k = pybamm.InputParameter("k")
model_smooth.rhs = {x: rhs_smooth(x, k)}
model_smooth.initial_conditions = {x: 0.5}
model_smooth.variables = {"x": x, "rhs": rhs_smooth(x, k)}

pybamm.set_logging_level("INFO")
# solver = pybamm.ScikitsDaeSolver()
solver = pybamm.CasadiSolver(extra_options_setup={"print_stats": True})
print("Exact-------------------------")
sols = [solver.solve(model_exact, [0, 2])]
for k in [1, 10, 50]:
    print(f"Smooth, k={k}-------------------------")
    # solver = pybamm.ScikitsDaeSolver()
    solver = pybamm.CasadiSolver(extra_options_setup={"print_stats": True})
    sol = solver.solve(model_smooth, [0, 2], inputs={"k": k})
    sols.append(sol)

pybamm.dynamic_plot(sols, ["x", "rhs"])