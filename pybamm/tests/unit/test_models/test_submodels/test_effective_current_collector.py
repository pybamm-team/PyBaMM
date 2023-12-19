#
# Tests for the Effective Current Collector Resistance models
#
from tests import TestCase
import pybamm
import unittest
import numpy as np


class TestEffectiveResistance(TestCase):
    def test_well_posed(self):
        model = pybamm.current_collector.EffectiveResistance({"dimensionality": 1})
        model.check_well_posedness()

        model = pybamm.current_collector.EffectiveResistance({"dimensionality": 2})
        model.check_well_posedness()

    def test_default_parameters(self):
        model = pybamm.current_collector.EffectiveResistance({"dimensionality": 1})
        self.assertEqual(
            model.default_parameter_values, pybamm.ParameterValues("Marquis2019")
        )

    def test_default_geometry(self):
        model = pybamm.current_collector.EffectiveResistance({"dimensionality": 1})
        self.assertTrue("current collector" in model.default_geometry)
        self.assertNotIn("negative electrode", model.default_geometry)

        model = pybamm.current_collector.EffectiveResistance({"dimensionality": 2})
        self.assertTrue("current collector" in model.default_geometry)
        self.assertNotIn("negative electrode", model.default_geometry)

    def test_default_var_pts(self):
        model = pybamm.current_collector.EffectiveResistance({"dimensionality": 1})
        self.assertEqual(model.default_var_pts, {"y": 32, "z": 32})

    def test_default_solver(self):
        model = pybamm.current_collector.EffectiveResistance({"dimensionality": 1})
        self.assertIsInstance(model.default_solver, pybamm.CasadiAlgebraicSolver)

        model = pybamm.current_collector.EffectiveResistance({"dimensionality": 2})
        self.assertIsInstance(model.default_solver, pybamm.CasadiAlgebraicSolver)

    def test_bad_option(self):
        with self.assertRaisesRegex(pybamm.OptionError, "Dimension of"):
            pybamm.current_collector.EffectiveResistance({"dimensionality": 10})


class TestEffectiveResistancePostProcess(TestCase):
    def test_get_processed_variables(self):
        # solve cheap SPM to test post-processing (think of an alternative test?)
        models = [
            pybamm.lithium_ion.SPM(),
            pybamm.current_collector.EffectiveResistance({"dimensionality": 1}),
            pybamm.current_collector.EffectiveResistance({"dimensionality": 2}),
            pybamm.current_collector.AlternativeEffectiveResistance2D(),
        ]
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 5, "r_p": 5, "y": 5, "z": 5}
        param = models[0].default_parameter_values
        meshes = [None] * len(models)
        for i, model in enumerate(models):
            param.process_model(model)
            geometry = model.default_geometry
            param.process_geometry(geometry)
            meshes[i] = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
            disc = pybamm.Discretisation(meshes[i], model.default_spatial_methods)
            disc.process_model(model)
        t_eval = np.linspace(0, 100, 10)
        solution_1D = models[0].default_solver.solve(models[0], t_eval)
        # Process SPM V and I
        V = solution_1D["Voltage [V]"]
        I = solution_1D["Total current density [A.m-2]"]

        # Test potential can be constructed and evaluated without raising error
        # for each current collector model
        for model in models[1:]:
            solution = model.default_solver.solve(model)
            variables = model.post_process(solution, param, V, I)
            pts = np.array([0.1, 0.5, 0.9]) * min(
                param.evaluate(model.param.L_y), param.evaluate(model.param.L_z)
            )
            for var, processed_var in variables.items():
                if "Voltage [V]" in var:
                    processed_var(t=solution_1D.t[5])
                else:
                    if model.options["dimensionality"] == 1:
                        processed_var(t=solution_1D.t[5], z=pts)
                    else:
                        processed_var(t=solution_1D.t[5], y=pts, z=pts)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
