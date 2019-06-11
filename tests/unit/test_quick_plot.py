import pybamm
import unittest
import numpy as np


class TestQuickPlot(unittest.TestCase):
    def test_simple_ode_model(self):
        model = pybamm.SimpleODEModel()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        solver = model.default_solver
        t_eval = np.linspace(0, 2, 100)
        solution = solver.solve(model, t_eval)
        quick_plot = pybamm.QuickPlot(model, mesh, solution)
        quick_plot.plot(0)

        # update the axis
        new_axis = [0, 0.5, 0, 1]
        quick_plot.axis.update({"a": new_axis})
        self.assertEqual(quick_plot.axis["a"], new_axis)

        # and now reset them
        quick_plot.reset_axis()
        self.assertNotEqual(quick_plot.axis["a"], new_axis)

        # check dynamic plot loads
        quick_plot.dynamic_plot(testing=True)

        quick_plot.update(0.01)

        # Test with different output variables
        quick_plot = pybamm.QuickPlot(
            model, mesh, solution, ["b broadcasted", "c broadcasted"]
        )
        self.assertEqual(len(quick_plot.axis), 2)
        quick_plot.plot(0)

        # update the axis
        new_axis = [0, 0.5, 0, 1]
        var = "c broadcasted"
        quick_plot.axis.update({var: new_axis})
        self.assertEqual(quick_plot.axis[var], new_axis)

        # and now reset them
        quick_plot.reset_axis()
        self.assertNotEqual(quick_plot.axis[var], new_axis)

        # check dynamic plot loads
        quick_plot.dynamic_plot(testing=True)

        quick_plot.update(0.01)

    def test_failure(self):
        with self.assertRaisesRegex(TypeError, "'models' must be"):
            pybamm.QuickPlot(1, None, None)
        with self.assertRaisesRegex(TypeError, "'solutions' must be"):
            pybamm.QuickPlot(pybamm.BaseModel(), None, 1)
        with self.assertRaisesRegex(ValueError, "must provide the same"):
            pybamm.QuickPlot(
                pybamm.BaseModel(),
                None,
                [pybamm.Solution(0, 0, ""), pybamm.Solution(0, 0, "")],
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
