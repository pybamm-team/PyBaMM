#!/usr/bin/env python#!/usr/bin/env python
# coding: utf-8

import pybamm
import tests
import numpy as np
import os
import matplotlib.pyplot as plt
from pprint import pprint

os.chdir(pybamm.__path__[0] + "/..")

param_dict = {"a": 1, "b": 2, "c": 3}
parameter_values = pybamm.ParameterValues(param_dict)
print("parameter values are {}".format(parameter_values))

f = open("input/param_file.csv", "w+")
f.write(
    """
Name [units],Value
a, 4
b, 5
c, 6
"""
)
f.close()

parameter_values = pybamm.ParameterValues("input/param_file.csv")
print("parameter values are {}".format(parameter_values))

f = open("input/squared.py", "w+")
f.write(
    """
def squared(x):
    return x ** 2
"""
)
f.close()

parameter_values = pybamm.ParameterValues(
    "input/param_file.csv", {"my function": "squared.py"}
)
print("parameter values are {}".format(parameter_values))

a = pybamm.Parameter("a")
b = pybamm.Parameter("b")
c = pybamm.Parameter("c")
func = pybamm.FunctionParameter("my function", a)

expr = a + b * c
try:
    expr.evaluate()
except NotImplementedError as e:
    print(e)

expr_eval = parameter_values.process_symbol(expr)
print("{} = {}".format(expr_eval, expr_eval.evaluate()))

func_eval = parameter_values.process_symbol(func)
print("{} = {}".format(func_eval, func_eval.evaluate()))

new_parameter_values = pybamm.ParameterValues({"a": 2})

expr_eval_update = new_parameter_values.update_scalars(expr_eval)
print("{} = {}".format(expr_eval_update, expr_eval_update.evaluate()))

func_eval_update = new_parameter_values.update_scalars(func_eval)
print("{} = {}".format(func_eval_update, func_eval_update.evaluate()))

# Create model
model = pybamm.BaseModel()
u = pybamm.Variable("u")
a = pybamm.Parameter("a")
b = pybamm.Parameter("b")
model.rhs = {u: -a * u}
model.initial_conditions = {u: b}
model.variables = {"u": u}

# Set parameters ############################################
parameter_values = pybamm.ParameterValues({"a": 3, "b": 2})
parameter_values.process_model(model)
#############################################################

# Discretise using default discretisation
disc = pybamm.Discretisation()
disc.process_model(model)

# Solve
t_eval = np.linspace(0, 2, 30)
ode_solver = pybamm.ScikitsOdeSolver()
ode_solver.solve(model, t_eval)

# Post-process, so that u1 can be called at any time t (using interpolation)
t_sol1, y_sol1 = ode_solver.t, ode_solver.y
u1 = pybamm.ProcessedVariable(model.variables["u"], t_sol1, y_sol1)

# Update parameters and solve again ###############################
new_parameter_values = pybamm.ParameterValues({"a": 4, "b": -1})
new_parameter_values.update_model(model, disc)  # no need to re-discretise
ode_solver.solve(model, t_eval)
t_sol2, y_sol2 = ode_solver.t, ode_solver.y
u2 = pybamm.ProcessedVariable(model.variables["u"], t_sol2, y_sol2)
###################################################################

# Plot
t_fine = np.linspace(0, t_eval[-1], 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
ax1.plot(t_fine, 2 * np.exp(-3 * t_fine), t_sol1, u1(t_sol1), "o")
ax1.set_xlabel("t")
ax1.legend([" * exp(-3 * t)", "u1"], loc="best")
ax1.set_title("a = 3, b = 2")

ax2.plot(t_fine, -np.exp(-4 * t_fine), t_sol2, u2(t_sol2), "o")
ax2.set_xlabel("t")
ax2.legend(["-exp(-4 * t)", "u2"], loc="best")
ax2.set_title("a = 4, b = -1")


plt.tight_layout()
plt.show()

model.rhs

parameters = {"a": a, "b": b, "a + b": a + b, "a * b": a * b}
param_eval = pybamm.print_parameters(parameters, parameter_values)
for name, (value, C_dependence) in param_eval.items():
    print("{}: {}".format(name, value))
