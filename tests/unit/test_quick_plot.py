import pybamm
import unittest
import numpy as np


class TestQuickPlot(unittest.TestCase):
    """
    Tests that QuickPlot is created correctly
    """

    def test_plot_creation(self):
        model = pybamm.lithium_ion.SPMe()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        solver = model.default_solver
        t_eval = np.linspace(0, 2, 100)
        solver.solve(model, t_eval)

        quick_plot = pybamm.QuickPlot(model, param, mesh, solver)
        quick_plot.plot(0)

        # update the axis
        new_axis = [0, 0.5, 0, 1]
        quick_plot.axis.update({"Electrolyte concentration": new_axis})
        self.assertEqual(quick_plot.axis["Electrolyte concentration"], new_axis)

        # and now reset them
        quick_plot.reset_axis()
        self.assertNotEqual(quick_plot.axis["Electrolyte concentration"], new_axis)

        # check dynamic plot loads
        quick_plot.dynamic_plot(testing=True)

        quick_plot.update(0.01)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
