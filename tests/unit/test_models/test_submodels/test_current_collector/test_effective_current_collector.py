#
# Tests for the Effective Current Collector Resistance model
#
import pybamm
import unittest
import numpy as np


@unittest.skipIf(pybamm.have_scikit_fem(), "scikit-fem not installed")
class TestEffectiveResistance2D(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.current_collector.EffectiveResistance2D()
        model.check_well_posedness()

    def test_default_geometry(self):
        model = pybamm.current_collector.EffectiveResistance2D()
        self.assertIsInstance(model.default_geometry, pybamm.Geometry)
        self.assertTrue("current collector" in model.default_geometry)
        self.assertNotIn("negative electrode", model.default_geometry)

    def test_default_solver(self):
        model = pybamm.current_collector.EffectiveResistance2D()
        self.assertIsInstance(model.default_solver, pybamm.AlgebraicSolver)

    def test_get_processed_potentials(self):
        # solve cheap SPM to test processed potentials (think of an alternative test?)
        models = [
            pybamm.current_collector.EffectiveResistance2D(),
            pybamm.lithium_ion.SPM(),
        ]
        var = pybamm.standard_spatial_vars
        var_pts = {
            var.x_n: 5,
            var.x_s: 5,
            var.x_p: 5,
            var.r_n: 5,
            var.r_p: 5,
            var.y: 5,
            var.z: 5,
        }
        param = models[1].default_parameter_values
        meshes = [None] * len(models)
        for i, model in enumerate(models):
            param.process_model(model)
            geometry = model.default_geometry
            param.process_geometry(geometry)
            meshes[i] = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
            disc = pybamm.Discretisation(meshes[i], model.default_spatial_methods)
            disc.process_model(model)
        solutions = [None] * len(models)
        t_eval = np.linspace(0, 0.1, 10)
        solutions[0] = models[0].default_solver.solve(models[0])
        solutions[1] = models[1].default_solver.solve(models[1], t_eval)

        # Process SPM V and I
        V = pybamm.ProcessedVariable(
            models[1].variables["Terminal voltage"],
            solutions[1].t,
            solutions[1].y,
            mesh=meshes[1],
        )
        I = pybamm.ProcessedVariable(
            models[1].variables["Total current density"],
            solutions[1].t,
            solutions[1].y,
            mesh=meshes[1],
        )

        # Test potential can be constructed without raising error
        models[0].get_processed_potentials(solutions[0], meshes[0], param, V, I)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
