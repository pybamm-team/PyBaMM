#
# Test for the evaluate-to-Julia functions
#
import pybamm

from tests import (
    get_mesh_for_testing,
    get_1p1d_mesh_for_testing,
    get_discretisation_for_testing,
    get_1p1d_discretisation_for_testing,
)
import unittest
import numpy as np
import scipy.sparse
from collections import OrderedDict

from julia import Main


model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, solver=pybamm.CasadiSolver(mode="fast"))
sim.solve([0, 3600])
param = model.default_parameter_values
timescale = param.evaluate(model.timescale)
sol = sim.solve(np.linspace(0, 0.15 * timescale, 100))
print(sol.y[:, -1])
print(sol.integration_time)
expr = pybamm.NumpyConcatenation(
    sim.built_model.concatenated_rhs, sim.built_model.concatenated_algebraic
).simplify()
# expr = sim.built_model.concatenated_rhs.simplify()

evaluator_str = pybamm.get_julia_function(expr)
n_rhs = sim.built_model.concatenated_rhs.size
n_alg = sim.built_model.concatenated_algebraic.size
# np.set_printoptions(
#     threshold=max(
#         np.get_printoptions()["threshold"],
#         n_rhs + n_alg,
#     )
# )
with open("tmp.txt", "w") as f:
    f.write(evaluator_str + "\n\n")
    f.write(f"u0 = {np.array2string(sol.model.y0, separator=',')}\n")
    f.write(f"du0 = zeros({n_rhs + n_alg})\n")
    f.write(f"differential_vars=[ones({n_rhs});zeros({n_alg})]\n")

# expr2 = sim.built_model.variables["Terminal voltage [V]"]
# evaluator_str2 = pybamm.get_julia_function(expr2)
# with open("tmp2.txt", "w") as f:
#     f.write(evaluator_str2 + "\n\n")

Main.eval(evaluator_str)
Main.dy = np.zeros(n_rhs + n_alg)
Main.y = sol.model.y0
Main.eval("f(dy,y,0,0)")
print(Main.dy)
expr.evaluate(y=sol.model.y0)
# print(expr.evaluate(y=Main.y))
# # test something with a heaviside
# a = pybamm.Vector([1, 2])
# expr = a <= pybamm.StateVector(slice(0, 2))
# evaluator_str = pybamm.get_julia_function(expr)
# evaluator = Main.eval(evaluator_str)
# for t, y in zip(t_tests, y_tests):
#     result = evaluator(t, y, None)
#     # note 1D arrays are flattened in Julia
#     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).flatten())

# expr = a > pybamm.StateVector(slice(0, 2))
# evaluator_str = pybamm.get_julia_function(expr)
# evaluator = Main.eval(evaluator_str)
# for t, y in zip(t_tests, y_tests):
#     result = evaluator(t, y, None)
#     # note 1D arrays are flattened in Julia
#     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).flatten())

# # # test something with a minimum or maximum
# # a = pybamm.Vector([1, 2])
# # expr = pybamm.minimum(a, pybamm.StateVector(slice(0, 2)))
# # evaluator_str = pybamm.get_julia_function(expr)
# # evaluator = Main.eval(evaluator_str)
# # for t, y in zip(t_tests, y_tests):
# #     result = evaluator(t,y,None)
# #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

# # expr = pybamm.maximum(a, pybamm.StateVector(slice(0, 2)))
# # evaluator_str = pybamm.get_julia_function(expr)
# # evaluator = Main.eval(evaluator_str)
# # for t, y in zip(t_tests, y_tests):
# #     result = evaluator(t,y,None)
# #     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

# # test something with an index
# expr = pybamm.Index(A @ pybamm.StateVector(slice(0, 2)), 0)
# evaluator_str = pybamm.get_julia_function(expr)
# evaluator = Main.eval(evaluator_str)
# for t, y in zip(t_tests, y_tests):
#     result = evaluator(t, y, None)
#     self.assertEqual(result, expr.evaluate(t=t, y=y))

# # test something with a sparse matrix multiplication
# A = pybamm.Matrix([[1, 2], [3, 4]])
# B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
# C = pybamm.Matrix(scipy.sparse.coo_matrix(np.array([[1, 0], [0, 4]])))
# expr = A @ B @ C @ pybamm.StateVector(slice(0, 2))
# evaluator_str = pybamm.get_julia_function(expr)
# evaluator = Main.eval(evaluator_str)
# for t, y in zip(t_tests, y_tests):
#     result = evaluator(t, y, None)
#     # note 1D arrays are flattened in Julia
#     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).flatten())

# expr = B @ pybamm.StateVector(slice(0, 2))
# evaluator_str = pybamm.get_julia_function(expr)
# evaluator = Main.eval(evaluator_str)
# for t, y in zip(t_tests, y_tests):
#     result = evaluator(t, y, None)
#     # note 1D arrays are flattened in Julia
#     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).flatten())

# # test numpy concatenation
# a = pybamm.StateVector(slice(0, 1))
# b = pybamm.StateVector(slice(1, 2))
# c = pybamm.StateVector(slice(2, 3))

# y_tests = [np.array([[2], [3], [4]]), np.array([[1], [3], [2]])]
# t_tests = [1, 2]

# expr = pybamm.NumpyConcatenation(a, b)
# evaluator_str = pybamm.get_julia_function(expr)
# evaluator = Main.eval(evaluator_str)
# for t, y in zip(t_tests, y_tests):
#     result = evaluator(t, y, None)
#     # note 1D arrays are flattened in Julia
#     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).flatten())

# expr = pybamm.NumpyConcatenation(a, c)
# evaluator_str = pybamm.get_julia_function(expr)
# evaluator = Main.eval(evaluator_str)
# for t, y in zip(t_tests, y_tests):
#     result = evaluator(t, y, None)
#     # note 1D arrays are flattened in Julia
#     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).flatten())

# # test sparse stack
# A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
# B = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[2, 0], [5, 0]])))
# a = pybamm.StateVector(slice(0, 1))
# expr = pybamm.SparseStack(A, a * B)
# evaluator_str = pybamm.get_julia_function(expr)
# evaluator = Main.eval(evaluator_str)
# for t, y in zip(t_tests, y_tests):
#     result = evaluator(t, y, None).toarray()
#     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).toarray())

# # test Inner
# expr = pybamm.Inner(a, b)
# evaluator_str = pybamm.get_julia_function(expr)
# evaluator = Main.eval(evaluator_str)
# for t, y in zip(t_tests, y_tests):
#     result = evaluator(t,y,None)
#     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))

# v = pybamm.StateVector(slice(0, 2))
# A = pybamm.Matrix(scipy.sparse.csr_matrix(np.array([[1, 0], [0, 4]])))
# expr = pybamm.Inner(A, v)
# evaluator_str = pybamm.get_julia_function(expr)
# evaluator = Main.eval(evaluator_str)
# for t, y in zip(t_tests, y_tests):
#     result = evaluator(t,y,None).toarray()
#     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y).toarray())

# y_tests = [np.array([[2], [3], [4], [5]]), np.array([[1], [3], [2], [1]])]
# t_tests = [1, 2]
# a = pybamm.StateVector(slice(0, 1), slice(3, 4))
# b = pybamm.StateVector(slice(1, 3))
# expr = a * b
# evaluator_str = pybamm.get_julia_function(expr)
# evaluator = Main.eval(evaluator_str)
# for t, y in zip(t_tests, y_tests):
#     result = evaluator(t,y,None)
#     np.testing.assert_allclose(result, expr.evaluate(t=t, y=y))
