#
# Test for the operator class
#
import pybamm
from tests import get_2p1d_mesh_for_testing

import numpy as np
import unittest


class TestFiniteElement(unittest.TestCase):
    def test(self):
        # get mesh
        mesh = get_2p1d_mesh_for_testing()
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "current collector": pybamm.FiniteElementFenics,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # discretise some equations
        var = pybamm.Variable("var", domain="current collector")
        y = pybamm.SpatialVariable("y", ["current collector"])
        z = pybamm.SpatialVariable("z", ["current collector"])
        disc.set_variable_slices([var])
        y_test = np.ones(mesh["current collector"][0].npts)
        const_source = pybamm.Broadcast(1, "current collector")
        disc.bcs = {var.id: {"left": (pybamm.Scalar(0), "Neumann"), "right": (pybamm.Scalar(0), "Neumann")}}

        for eqn in [
            pybamm.laplacian(var),
            pybamm.source(const_source),
            pybamm.laplacian(var) - pybamm.source(const_source),
            pybamm.Integral(var, [y, z]) - 1,
        ]:
            eqn_disc = disc.process_symbol(eqn)
            eqn_disc.evaluate(None, y_test)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
