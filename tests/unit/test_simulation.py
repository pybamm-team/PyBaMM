import pybamm
import unittest


class TestSimulation(unittest.TestCase):
    def test_basic_ops(self):

        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        self.assertEqual(model.__class__, sim._model_class)
        self.assertEqual(model.options, sim._model_options)

        # check that the model is unprocessed
        self.assertEqual(sim._status, "Unprocessed")
        self.assertEqual(sim._mesh, None)
        self.assertEqual(sim._disc, None)
        for val in list(sim.model.rhs.values()):
            self.assertTrue(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertFalse(val.has_symbol_of_classes(pybamm.Matrix))

        sim.parameterize()
        self.assertEqual(sim._status, "Parameterized")
        self.assertEqual(sim._mesh, None)
        self.assertEqual(sim._disc, None)
        for val in list(sim.model.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertFalse(val.has_symbol_of_classes(pybamm.Matrix))

        sim.build()
        self.assertEqual(sim._status, "Built")
        self.assertFalse(sim._mesh is None)
        self.assertFalse(sim._disc is None)
        for val in list(sim.model.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertTrue(val.has_symbol_of_classes(pybamm.Matrix))

        sim.parameterize()
        self.assertEqual(sim._status, "Parameterized")
        self.assertEqual(sim._mesh, None)
        self.assertEqual(sim._disc, None)
        for val in list(sim.model.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertFalse(val.has_symbol_of_classes(pybamm.Matrix))

        sim.build()
        sim.reset()
        self.assertEqual(sim._status, "Unprocessed")
        self.assertEqual(sim._mesh, None)
        self.assertEqual(sim._disc, None)
        for val in list(sim.model.rhs.values()):
            self.assertTrue(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertFalse(val.has_symbol_of_classes(pybamm.Matrix))

    def test_solve(self):

        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sim.solve()
        self.assertFalse(sim._solution is None)
        self.assertEqual(sim._status, "Solved")
        for val in list(sim.model.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertTrue(val.has_symbol_of_classes(pybamm.Matrix))

        sim.reset()
        self.assertEqual(sim._status, "Unprocessed")
        for val in list(sim.model.rhs.values()):
            self.assertTrue(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertFalse(val.has_symbol_of_classes(pybamm.Matrix))

        self.assertEqual(sim._solution, None)
        # check can now re-parameterize model

    def test_reuse_commands(self):

        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())

        sim.parameterize()
        sim.parameterize()

        sim.build()
        sim.build()

        sim.solve()
        sim.solve()

        sim.build()
        sim.solve()
        sim.parameterize()

    def test_specs(self):
        # test can rebuild after setting specs
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sim.build()

        model_options = {"thermal": "lumped"}
        sim.specs(model_options=model_options)
        sim.build()
        self.assertEqual(sim.model.options["thermal"], "lumped")

        params = sim.parameter_values
        # normally is 0.0001
        params.update({"Negative electrode thickness [m]": 0.0002})
        sim.specs(parameter_values=params)

        self.assertEqual(
            sim.parameter_values["Negative electrode thickness [m]"], 0.0002
        )
        sim.build()

        geometry = sim.unprocessed_geometry
        custom_geometry = {}
        x_n = pybamm.standard_spatial_vars.x_n
        custom_geometry["negative electrode"] = {
            "primary": {
                x_n: {"min": pybamm.Scalar(0), "max": pybamm.geometric_parameters.l_n}
            }
        }
        geometry.update(custom_geometry)
        sim.specs(geometry=geometry)
        sim.build()

        var_pts = sim.var_pts
        var_pts[pybamm.standard_spatial_vars.x_n] = 5
        sim.specs(var_pts=var_pts)
        sim.build()

        spatial_methods = sim.spatial_methods
        # nothing to change this to at the moment but just reload in
        sim.specs(spatial_methods=spatial_methods)
        sim.build()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()

