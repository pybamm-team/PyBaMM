#
# Tests for updating parameter values
#
import pybamm

import unittest
import numpy as np
import tests


class TestUpdateParameters(unittest.TestCase):
    def test_update_parameters_eqn(self):
        a = pybamm.Scalar(1)
        b = pybamm.Scalar(2, name="test parameter")
        c = pybamm.Scalar(3)
        eqn = a + b * c
        self.assertEqual(eqn.evaluate(), 7)

        parameter_values = pybamm.ParameterValues({"test parameter": 3})
        eqn_changed = parameter_values.update_scalars(eqn)
        self.assertEqual(eqn_changed.evaluate(), 10)

    def test_set_and_update_parameters(self):
        a = pybamm.Scalar(1)
        b = pybamm.Parameter(name="test parameter")
        c = pybamm.Scalar(3)
        eqn = a + b * c

        parameter_values = pybamm.ParameterValues({"test parameter": 2})
        eqn_processed = parameter_values.process_symbol(eqn)
        self.assertEqual(eqn_processed.evaluate(), 7)

        parameter_values = pybamm.ParameterValues({"test parameter": 3})
        eqn_updated = parameter_values.update_scalars(eqn_processed)
        self.assertEqual(eqn_updated.evaluate(), 10)

    def test_update_model(self):
        # test on simple lithium-ion model
        model1 = pybamm.lithium_ion.SPM()
        modeltest1 = tests.StandardModelTest(model1)
        t_eval = np.linspace(0, 0.1)

        modeltest1.test_all(t_eval=t_eval, skip_output_tests=True)
        Y1 = modeltest1.solution.y

        # double initial conditions
        model2 = pybamm.lithium_ion.SPM()
        # process and solve the model a first time
        modeltest2 = tests.StandardModelTest(model2)
        modeltest2.test_all(skip_output_tests=True)
        self.assertEqual(model2.variables["Current [A]"].evaluate(), 0.68)
        # process and solve with updated parameter values
        parameter_values_update = pybamm.ParameterValues(
            chemistry=pybamm.parameter_sets.Marquis2019
        )
        parameter_values_update.update({"Typical current [A]": 2})
        modeltest2.test_update_parameters(parameter_values_update)
        self.assertEqual(model2.variables["Current [A]"].evaluate(), 2)
        modeltest2.test_solving(t_eval=t_eval)
        Y2 = modeltest2.solution.y

        # results should be different
        self.assertNotEqual(np.linalg.norm(Y1 - Y2), 0)

        # test with new current function
        model3 = pybamm.lithium_ion.SPM()
        modeltest3 = tests.StandardModelTest(model3)
        modeltest3.test_all(skip_output_tests=True)
        parameter_values_update = pybamm.ParameterValues(
            chemistry=pybamm.parameter_sets.Marquis2019
        )
        parameter_values_update.update({"Current function": 0})
        modeltest3.test_update_parameters(parameter_values_update)
        modeltest3.test_solving(t_eval=t_eval)
        Y3 = modeltest3.solution.y

        # function.parameters should be pybamm.Scalar(0), but parameters_eval s
        # should be a float
        self.assertIsInstance(
            model3.variables["Current [A]"].function.parameters["Current [A]"],
            pybamm.Scalar,
        )
        self.assertEqual(
            model3.variables["Current [A]"].function.parameters_eval["Current [A]"], 0.0
        )

        # results should be different
        self.assertNotEqual(np.linalg.norm(Y1 - Y3), 0)

    def test_inplace(self):
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        new_model = param.process_model(model, inplace=False)

        for val in list(model.rhs.values()):
            self.assertTrue(val.has_symbol_of_classes(pybamm.Parameter))

        for val in list(new_model.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))

    def test_update_geometry(self):
        # test on simple lead-acid model
        model1 = pybamm.lead_acid.LOQS()
        modeltest1 = tests.StandardModelTest(model1)
        t_eval = np.linspace(0, 0.5)
        modeltest1.test_all(t_eval=t_eval, skip_output_tests=True)

        T1, Y1 = modeltest1.solution.t, modeltest1.solution.y

        # trying to update the geometry fails
        parameter_values_update = pybamm.ParameterValues(
            chemistry=pybamm.parameter_sets.Sulzer2019
        )
        parameter_values_update.update(
            {
                "Negative electrode thickness [m]": 0.0002,
                "Separator thickness [m]": 0.0003,
                "Positive electrode thickness [m]": 0.0004,
            }
        )
        with self.assertRaisesRegex(ValueError, "geometry has changed"):
            modeltest1.test_update_parameters(parameter_values_update)

        # instead we need to make a new model and re-discretise
        model2 = pybamm.lead_acid.LOQS()
        # nb: need to be careful make parameters a reasonable size
        modeltest2 = tests.StandardModelTest(model2)
        modeltest2.test_all(
            param=parameter_values_update, t_eval=t_eval, skip_output_tests=True
        )
        T2, Y2 = modeltest2.solution.t, modeltest2.solution.y
        # results should be different
        c1 = pybamm.ProcessedVariable(
            modeltest1.model.variables["Electrolyte concentration"],
            T1,
            Y1,
            mesh=modeltest1.disc.mesh,
        ).entries
        c2 = pybamm.ProcessedVariable(
            modeltest2.model.variables["Electrolyte concentration"],
            T2,
            Y2,
            mesh=modeltest2.disc.mesh,
        ).entries
        self.assertNotEqual(np.linalg.norm(c1 - c2), 0)
        self.assertNotEqual(np.linalg.norm(Y1 - Y2), 0)


if __name__ == "__main__":
    import sys

    print("Add -v for more debug output")

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
