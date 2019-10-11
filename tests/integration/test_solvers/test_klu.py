import pybamm
import numpy as np
import sys
import unittest


@unittest.skipIf(pybamm.have_klu(), "klu solver is not installed")
class TestKLUSolver(unittest.TestCase):
    def test_on_spme(self):
        model = pybamm.lithium_ion.SPMe()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        t_eval = np.linspace(0, 0.2, 100)
        solution = pybamm.KLU().solve(model, t_eval)
        np.testing.assert_array_less(1, solution.t.size)


if __name__ == "__main__":
    print("Add -v for more debug output")

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()

