import pybamm
import numpy as np
import unittest


class TestSimulation(unittest.TestCase):
    def test_basic_ops(self):

        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        self.assertEqual(model.__class__, sim._model_class)
        self.assertEqual(model.options, sim._model_options)

        # check that the model is unprocessed
        self.assertEqual(sim._mesh, None)
        self.assertEqual(sim._disc, None)
        for val in list(sim.model.rhs.values()):
            self.assertTrue(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertFalse(val.has_symbol_of_classes(pybamm.Matrix))

        sim.set_parameters()
        self.assertEqual(sim._mesh, None)
        self.assertEqual(sim._disc, None)
        for val in list(sim.model_with_set_params.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertFalse(val.has_symbol_of_classes(pybamm.Matrix))

        sim.build()
        self.assertFalse(sim._mesh is None)
        self.assertFalse(sim._disc is None)
        for val in list(sim.built_model.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertTrue(val.has_symbol_of_classes(pybamm.Matrix))

        sim.reset()
        sim.set_parameters()
        self.assertEqual(sim._mesh, None)
        self.assertEqual(sim._disc, None)
        self.assertEqual(sim.built_model, None)

        for val in list(sim.model_with_set_params.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertFalse(val.has_symbol_of_classes(pybamm.Matrix))

        sim.build()
        sim.reset()
        self.assertEqual(sim._mesh, None)
        self.assertEqual(sim._disc, None)
        self.assertEqual(sim.model_with_set_params, None)
        self.assertEqual(sim.built_model, None)
        for val in list(sim.model.rhs.values()):
            self.assertTrue(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertFalse(val.has_symbol_of_classes(pybamm.Matrix))

    def test_solve(self):

        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sim.solve()
        self.assertFalse(sim._solution is None)
        for val in list(sim.built_model.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertTrue(val.has_symbol_of_classes(pybamm.Matrix))

        sim.reset()
        self.assertEqual(sim.model_with_set_params, None)
        self.assertEqual(sim.built_model, None)
        for val in list(sim.model.rhs.values()):
            self.assertTrue(val.has_symbol_of_classes(pybamm.Parameter))
            self.assertFalse(val.has_symbol_of_classes(pybamm.Matrix))

        self.assertEqual(sim._solution, None)

    def test_reuse_commands(self):

        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())

        sim.set_parameters()
        sim.set_parameters()

        sim.build()
        sim.build()

        sim.solve()
        sim.solve()

        sim.build()
        sim.solve()
        sim.set_parameters()

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

    def test_set_crate(self):
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM(), C_rate=2)
        self.assertEqual(sim.parameter_values["C-rate"], 2)
        sim.specs(C_rate=3)
        self.assertEqual(sim.parameter_values["C-rate"], 3)

    def test_set_defaults(self):
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())

        model_options = {"thermal": "x-full"}
        submesh_types = {
            "Negative particle": pybamm.MeshGenerator(pybamm.Exponential1DSubMesh)
        }
        solver = pybamm.BaseSolver()
        quick_plot_vars = ["Negative particle surface concentration"]
        sim.specs(
            model_options=model_options,
            submesh_types=submesh_types,
            solver=solver,
            quick_plot_vars=quick_plot_vars,
        )

        sim.set_defaults()

        self.assertEqual(sim.model_options["thermal"], "x-full")
        self.assertEqual(
            sim.submesh_types["negative particle"].submesh_type, pybamm.Uniform1DSubMesh
        )
        self.assertEqual(sim.quick_plot_vars, None)
        self.assertIsInstance(sim.solver, pybamm.ScipySolver)

    def test_get_variable_array(self):

        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sim.solve()

        phi_s_n = sim.get_variable_array("Negative electrode potential")

        self.assertIsInstance(phi_s_n, np.ndarray)

        c_s_n_surf, c_e = sim.get_variable_array(
            "Negative particle surface concentration", "Electrolyte concentration"
        )

        self.assertIsInstance(c_s_n_surf, np.ndarray)
        self.assertIsInstance(c_e, np.ndarray)

    def test_set_external_variable(self):
        model_options = {
            "thermal": "x-lumped",
            "external submodels": ["thermal"],
        }
        model = pybamm.lithium_ion.SPMe(model_options)
        sim = pybamm.Simulation(model)

        T_av = 0

        dt = 0.001

        external_variables = {"X-averaged cell temperature": T_av}
        sim.step(dt, external_variables=external_variables)

    def test_step(self):

        dt = 0.001
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sim.step(dt)  # 1 step stores first two points
        self.assertEqual(sim.solution.t.size, 2)
        self.assertEqual(sim.solution.y[0, :].size, 2)
        self.assertEqual(sim.solution.t[0], 0)
        self.assertEqual(sim.solution.t[1], dt)
        sim.step(dt)  # automatically append the next step
        self.assertEqual(sim.solution.t.size, 3)
        self.assertEqual(sim.solution.y[0, :].size, 3)
        self.assertEqual(sim.solution.t[0], 0)
        self.assertEqual(sim.solution.t[1], dt)
        self.assertEqual(sim.solution.t[2], 2 * dt)
        sim.step(dt, save=False)  # now only store the two end step points
        self.assertEqual(sim.solution.t.size, 2)
        self.assertEqual(sim.solution.y[0, :].size, 2)
        self.assertEqual(sim.solution.t[0], 2 * dt)
        self.assertEqual(sim.solution.t[1], 3 * dt)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()

