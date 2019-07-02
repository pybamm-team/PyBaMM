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
        quick_plot.axis.update({("a",): new_axis})
        self.assertEqual(quick_plot.axis[("a",)], new_axis)

        # and now reset them
        quick_plot.reset_axis()
        self.assertNotEqual(quick_plot.axis[("a",)], new_axis)

        # check dynamic plot loads
        quick_plot.dynamic_plot(testing=True)

        quick_plot.update(0.01)

        # Test with different output variables
        quick_plot = pybamm.QuickPlot(model, mesh, solution, ["b broadcasted"])
        self.assertEqual(len(quick_plot.axis), 1)
        quick_plot.plot(0)

        quick_plot = pybamm.QuickPlot(
            model,
            mesh,
            solution,
            [["a", "a"], ["b broadcasted", "b broadcasted"], "c broadcasted"],
        )
        self.assertEqual(len(quick_plot.axis), 3)
        quick_plot.plot(0)

        # update the axis
        new_axis = [0, 0.5, 0, 1]
        var_key = ("c broadcasted",)
        quick_plot.axis.update({var_key: new_axis})
        self.assertEqual(quick_plot.axis[var_key], new_axis)

        # and now reset them
        quick_plot.reset_axis()
        self.assertNotEqual(quick_plot.axis[var_key], new_axis)

        # check dynamic plot loads
        quick_plot.dynamic_plot(testing=True)

        quick_plot.update(0.01)

        # Test longer name
        model.variables["Variable with a very long name"] = model.variables["a"]
        quick_plot = pybamm.QuickPlot(model, mesh, solution)
        quick_plot.plot(0)

        # Test errors
        with self.assertRaisesRegex(ValueError, "mismatching variable domains"):
            pybamm.QuickPlot(model, mesh, solution, [["a", "b broadcasted"]])
        model.variables["3D variable"] = disc.process_symbol(
            pybamm.Broadcast(1, ["negative particle"])
        )
        with self.assertRaisesRegex(NotImplementedError, "cannot plot 3D variables"):
            pybamm.QuickPlot(model, mesh, solution, ["3D variable"])

    def test_loqs_spm_base(self):
        t_eval = np.linspace(0, 0.01, 2)

        # SPM
        options = {"thermal": None, "Voltage": "on"}
        for model in [pybamm.lithium_ion.SPM(options), pybamm.old_lead_acid.OldLOQS()]:
            geometry = model.default_geometry
            param = model.default_parameter_values
            param.process_model(model)
            param.process_geometry(geometry)
            mesh = pybamm.Mesh(
                geometry, model.default_submesh_types, model.default_var_pts
            )
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            disc.process_model(model)
            solver = model.default_solver
            solution = solver.solve(model, t_eval)
            pybamm.QuickPlot(model, mesh, solution)

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
