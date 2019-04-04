import pybamm
from tests import get_mesh_for_testing
import numpy as np

whole_cell = ["negative electrode", "separator", "positive electrode"]

# create discretisation
mesh = get_mesh_for_testing()
spatial_methods = {"macroscale": pybamm.FiniteVolume}
disc = pybamm.Discretisation(mesh, spatial_methods)

combined_submesh = mesh.combine_submeshes(*whole_cell)

# Discretise some equations where averaging is needed
var = pybamm.Variable("var", domain=whole_cell)
disc.set_variable_slices([var])
y = pybamm.StateVector(slice(0, combined_submesh[0].npts))
y_test = np.ones_like(combined_submesh[0].nodes)

flux = var * pybamm.grad(var)
eqn = pybamm.div(flux)
disc._bcs = {flux.id: {"left": pybamm.Scalar(1), "right": pybamm.Scalar(2)}}
eqn_disc = disc.process_symbol(eqn)
import ipdb

ipdb.set_trace()
eqn_jac = eqn_disc.jac(y)
eqn_jac.evaluate(y=y_test)
