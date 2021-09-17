#
# Tests for the basic lithium-ion models
#
import pybamm
import unittest
from pybamm.geometry import half_cell_spatial_vars


class TestBasicModels(unittest.TestCase):
    def test_dfn_well_posed(self):
        model = pybamm.lithium_ion.BasicDFN()
        model.check_well_posedness()

        copy = model.new_copy()
        copy.check_well_posedness()

    def test_spm_well_posed(self):
        model = pybamm.lithium_ion.BasicSPM()
        model.check_well_posedness()

        copy = model.new_copy()
        copy.check_well_posedness()

    def test_dfn_half_cell_well_posed(self):
        options = {"working electrode": "positive"}
        model = pybamm.lithium_ion.BasicDFNHalfCell(options=options)
        model.check_well_posedness()

        copy = model.new_copy()
        copy.check_well_posedness()

        options = {"working electrode": "negative"}
        model = pybamm.lithium_ion.BasicDFNHalfCell(options=options)
        model.check_well_posedness()

        copy = model.new_copy()
        copy.check_well_posedness()

    def test_dfn_half_cell_simulation_with_experiment_error(self):
        options = {"working electrode": "negative"}
        model = pybamm.lithium_ion.BasicDFNHalfCell(options=options)
        experiment = pybamm.Experiment([
            ("Discharge at C/10 for 10 hours or until 3.5 V")])
        with self.assertRaisesRegex(
                NotImplementedError,
                "BasicDFNHalfCell is not compatible with experiment simulations yet."):
            pybamm.Simulation(model, experiment=experiment)

    def basic_dfn_half_cell_simulation(self):
        model = pybamm.lithium_ion.BasicDFNHalfCell(
            options={"working electrode": "positive"})
        chemistry = pybamm.parameter_sets.Chen2020
        param = pybamm.ParameterValues(chemistry=chemistry)
        param.update(
            {
                "Lithium counter electrode exchange-current density [A.m-2]": 12.6,
                "Lithium counter electrode conductivity [S.m-1]": 1.0776e7,
                "Lithium counter electrode thickness [m]": 250e-6,
            },
            check_already_exists=False, )
        param["Initial concentration in negative electrode [mol.m-3]"] = 1000
        param["Current function [A]"] = 2.5
        sim = pybamm.Simulation(model=model, parameter_values=param)
        sim.solve([0, 100])
        self.assertTrue(
            isinstance(sim.solution, pybamm.solvers.solution.Solution)
        )

    def test_dfn_half_cell_defaults(self):
        # test default geometry
        var = half_cell_spatial_vars

        model = pybamm.lithium_ion.BasicDFNHalfCell(
            options={"dimensionality": 0, "working electrode": "positive"}
        )
        self.assertEqual(
            model.default_geometry["current collector"][var.z]["position"], 1
        )
        model = pybamm.lithium_ion.BasicDFNHalfCell(
            options={"dimensionality": 1, "working electrode": "positive"}
        )
        self.assertEqual(model.default_geometry["current collector"][var.z]["min"], 0)
        model = pybamm.lithium_ion.BasicDFNHalfCell(
            options={"dimensionality": 2, "working electrode": "positive"}
        )
        self.assertEqual(model.default_geometry["current collector"][var.y]["min"], 0)

        # test default variable points
        var = half_cell_spatial_vars
        var_pts = {
            var.x_Li: 20,
            var.x_s: 20,
            var.x_w: 20,
            var.r_w: 30,
            var.y: 10,
            var.z: 10,
        }
        model = pybamm.lithium_ion.BasicDFNHalfCell(
            options={"dimensionality": 0, "working electrode": "positive"}
        )
        self.assertDictEqual(var_pts, model.default_var_pts)

        var_pts.update({var.x_Li: 10, var.x_s: 10, var.x_w: 10})
        model = pybamm.lithium_ion.BasicDFNHalfCell(
            options={"dimensionality": 2, "working electrode": "positive"}
        )
        self.assertDictEqual(var_pts, model.default_var_pts)

        # test default submesh types
        model = pybamm.lithium_ion.BasicDFNHalfCell(
            options={"dimensionality": 0, "working electrode": "positive"}
        )
        self.assertTrue(
            issubclass(
                model.default_submesh_types["current collector"].submesh_type,
                pybamm.SubMesh0D,
            )
        )
        model = pybamm.lithium_ion.BasicDFNHalfCell(
            options={"dimensionality": 1, "working electrode": "positive"}
        )
        self.assertTrue(
            issubclass(
                model.default_submesh_types["current collector"].submesh_type,
                pybamm.Uniform1DSubMesh,
            )
        )
        model = pybamm.lithium_ion.BasicDFNHalfCell(
            options={"dimensionality": 2, "working electrode": "positive"}
        )
        self.assertTrue(
            issubclass(
                model.default_submesh_types["current collector"].submesh_type,
                pybamm.ScikitUniform2DSubMesh,
            )
        )
        # test default spatial methods
        model = pybamm.lithium_ion.BasicDFNHalfCell(
            options={"dimensionality": 0, "working electrode": "positive"}
        )
        self.assertTrue(
            isinstance(
                model.default_spatial_methods["current collector"],
                pybamm.ZeroDimensionalSpatialMethod,
            )
        )
        model = pybamm.lithium_ion.BasicDFNHalfCell(
            options={"dimensionality": 1, "working electrode": "positive"}
        )
        self.assertTrue(
            isinstance(
                model.default_spatial_methods["current collector"], pybamm.FiniteVolume
            )
        )
        model = pybamm.lithium_ion.BasicDFNHalfCell(
            options={"dimensionality": 2, "working electrode": "positive"}
        )
        self.assertTrue(
            isinstance(
                model.default_spatial_methods["current collector"],
                pybamm.ScikitFiniteElement,
            )
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
