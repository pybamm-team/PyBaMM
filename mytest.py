import pybamm

import numpy as np
import unittest
from scipy.sparse.coo import coo_matrix

y = pybamm.StateVector(slice(0, 4))
u = pybamm.StateVector(slice(0, 2))
v = pybamm.StateVector(slice(2, 4))

y0 = np.ones(4)

import ipdb; ipdb.set_trace()
u.domain = ["current collector"]
func = pybamm.Outer(u, pybamm.Scalar(4))
jacobian = np.array([[4,0,0,0], [0,4,0,0]])
dfunc_dy = func.jac(y).evaluate(y=y0)

func = pybamm.Outer(u, pybamm.Vector(np.array([1,2,3])))
jacobian = np.array([[1,0,0,0], [2,0,0,0], [3,0,0,0], [0,1,0,0], [0,2,0,0], [0,3,0,0]])
dfunc_dy = func.jac(y).evaluate(y=y0)
