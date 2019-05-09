import pybamm
import unittest
import numpy as np


class TestQuickPlot(unittest.TestCase):
    """
    Tests that QuickPlot is created correctly
    """

    def setUp(self):
        # Set up lithium ion
        spm = pybamm.lithium_ion.SPM()
        spme = pybamm.lithium_ion.SPMe()
        geometry = spm.default_geometry
        param_li = spm.default_parameter_values
        param_li.process_model(spm)
        param_li.process_model(spme)
        param_li.process_geometry(geometry)
        self.mesh_li = pybamm.Mesh(
            geometry, spme.default_submesh_types, spme.default_var_pts
        )
        disc_spm = pybamm.Discretisation(self.mesh_li, spm.default_spatial_methods)
        disc_spme = pybamm.Discretisation(self.mesh_li, spme.default_spatial_methods)
        disc_spm.process_model(spm)
        disc_spme.process_model(spme)
        solver_spm = spm.default_solver
        solver_spme = spme.default_solver
        t_eval = np.linspace(0, 2, 100)
        solver_spm.solve(spm, t_eval)
        solver_spme.solve(spme, t_eval)
        self.models_li = [spm, spme]
        self.solvers_li = [solver_spm, solver_spme]

        # Set up lead-acid
        self.loqs = pybamm.lead_acid.LOQS()
        geometry = self.loqs.default_geometry
        param_la = self.loqs.default_parameter_values
        param_la.process_model(self.loqs)
        param_la.process_geometry(geometry)
        self.mesh_la = pybamm.Mesh(
            geometry, self.loqs.default_submesh_types, self.loqs.default_var_pts
        )
        disc_loqs = pybamm.Discretisation(
            self.mesh_la, self.loqs.default_spatial_methods
        )
        disc_loqs.process_model(self.loqs)
        self.solver_loqs = self.loqs.default_solver
        self.solver_loqs.solve(self.loqs, t_eval)

    def tearDown(self):
        del self.models_li
        del self.mesh_la
        del self.solvers_li
        del self.loqs
        del self.mesh_li
        del self.solver_loqs

    def test_plot_creation(self):
        quick_plot = pybamm.QuickPlot(self.models_li, self.mesh_li, self.solvers_li)
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

    def test_plot_creation_lead_acid(self):
        pybamm.QuickPlot(self.loqs, self.mesh_la, self.solver_loqs)

    def test_plot_different_variables(self):
        output_vars = [
            "Negative particle surface concentration",
            "Electrolyte concentration",
            "Positive particle surface concentration",
        ]
        quick_plot = pybamm.QuickPlot(
            self.models_li, self.mesh_li, self.solvers_li, output_vars
        )
        self.assertEqual(len(quick_plot.axis), 3)

    def test_failure(self):
        with self.assertRaisesRegex(TypeError, "'models' must be"):
            pybamm.QuickPlot(1, None, None)
        with self.assertRaisesRegex(TypeError, "'solvers' must be"):
            pybamm.QuickPlot(self.models_li, None, 1)
        with self.assertRaisesRegex(ValueError, "must provide the same"):
            pybamm.QuickPlot(self.models_li, None, self.solvers_li[0])


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
