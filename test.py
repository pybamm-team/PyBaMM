import pybamm
import tests

import numpy as np

t = pybamm.t
y = pybamm.StateVector(slice(0, 3))
var = t * y
disc = tests.get_2p1d_discretisation_for_testing()
var.mesh = disc.mesh["current collector"]
t_sol = np.array([1])
y_sol = np.linspace(0, 5)[:, np.newaxis]
pybamm.ProcessedVariable(var, pybamm.Solution(t_sol, y_sol))
