import pybamm
import tests

import numpy as np

var = pybamm.Variable("var", domain=["current collector"])
y = pybamm.SpatialVariable("y", domain=["current collector"])
z = pybamm.SpatialVariable("z", domain=["current collector"])

disc = tests.get_2p1d_discretisation_for_testing()
disc.set_variable_slices([var])
y_sol = disc.process_symbol(y).entries[:, 0]
z_sol = disc.process_symbol(z).entries[:, 0]
var_sol = disc.process_symbol(var)
t_sol = np.array([0])
u_sol = np.ones(var_sol.shape[0])[:, np.newaxis]

processed_var = pybamm.ProcessedVariable(var_sol, t_sol, u_sol, mesh=disc.mesh)
np.testing.assert_array_equal(
    processed_var.entries,
    np.reshape(u_sol, [len(y_sol), len(z_sol)]),
)
