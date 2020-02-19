import pybamm
import unittest
import numpy as np


class TestQuickPlot(unittest.TestCase):
    """
    Tests that QuickPlot is created correctly
    """

    def test_plot_lithium_ion(self):
        spm = pybamm.lithium_ion.SPM()
        spme = pybamm.lithium_ion.SPMe()
        geometry = spm.default_geometry
        param = spm.default_parameter_values
        param.process_model(spm)
        param.process_model(spme)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, spme.default_submesh_types, spme.default_var_pts)
        disc_spm = pybamm.Discretisation(mesh, spm.default_spatial_methods)
        disc_spme = pybamm.Discretisation(mesh, spme.default_spatial_methods)
        disc_spm.process_model(spm)
        disc_spme.process_model(spme)
        t_eval = np.linspace(0, 3600, 100)
        solution_spm = spm.default_solver.solve(spm, t_eval)
        solution_spme = spme.default_solver.solve(spme, t_eval)
        quick_plot = pybamm.QuickPlot([solution_spm, solution_spme])
        quick_plot.plot(0)

        # update the axis
        new_axis = [0, 0.5, 0, 1]
        quick_plot.axis.update({("Electrolyte concentration",): new_axis})
        self.assertEqual(quick_plot.axis[("Electrolyte concentration",)], new_axis)

        # and now reset them
        quick_plot.reset_axis()
        self.assertNotEqual(quick_plot.axis[("Electrolyte concentration",)], new_axis)

        # check dynamic plot loads
        quick_plot.dynamic_plot(testing=True)

        quick_plot.update(0.01)

        # Test with different output variables
        output_vars = [
            "Negative particle surface concentration",
            "Electrolyte concentration",
            "Positive particle surface concentration",
        ]
        quick_plot = pybamm.QuickPlot(solution_spm, output_vars)
        self.assertEqual(len(quick_plot.axis), 3)
        quick_plot.plot(0)

        # update the axis
        new_axis = [0, 0.5, 0, 1]
        quick_plot.axis.update({("Electrolyte concentration",): new_axis})
        self.assertEqual(quick_plot.axis[("Electrolyte concentration",)], new_axis)

        # and now reset them
        quick_plot.reset_axis()
        self.assertNotEqual(quick_plot.axis[("Electrolyte concentration",)], new_axis)

        # check dynamic plot loads
        quick_plot.dynamic_plot(testing=True)

        quick_plot.update(0.01)

    def test_plot_lead_acid(self):
        loqs = pybamm.lead_acid.LOQS()
        geometry = loqs.default_geometry
        param = loqs.default_parameter_values
        param.process_model(loqs)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, loqs.default_submesh_types, loqs.default_var_pts)
        disc_loqs = pybamm.Discretisation(mesh, loqs.default_spatial_methods)
        disc_loqs.process_model(loqs)
        t_eval = np.linspace(0, 3600, 100)
        solution_loqs = loqs.default_solver.solve(loqs, t_eval)

        pybamm.QuickPlot(solution_loqs)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
