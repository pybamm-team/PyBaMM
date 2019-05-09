import pybamm
import unittest
import numpy as np


class TestQuickPlot(unittest.TestCase):
    """
    Tests that QuickPlot is created correctly
    """

    def test_plot_creation(self):
        spm = pybamm.lithium_ion.SPM()
        spme = pybamm.lithium_ion.SPMe()
        geometry = spme.default_geometry
        param = spme.default_parameter_values
        param.process_model(spm)
        param.process_model(spme)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, spme.default_submesh_types, spme.default_var_pts)
        disc = pybamm.Discretisation(mesh, spme.default_spatial_methods)
        disc.process_model(spm)
        disc.process_model(spme)
        solver_spm = spm.default_solver
        solver_spme = spme.default_solver
        t_eval = np.linspace(0, 2, 100)
        solver_spm.solve(spm, t_eval)
        solver_spme.solve(spme, t_eval)

        quick_plot = pybamm.QuickPlot(
            [spm, spme], param, mesh, [solver_spm, solver_spme]
        )
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
