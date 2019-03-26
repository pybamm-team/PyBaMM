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
        eqn_changed = parameter_values.process_symbol(eqn)
        self.assertEqual(eqn_changed.evaluate(), 10)

    def test_set_and_update_parameters(self):
        a = pybamm.Scalar(1)
        b = pybamm.Parameter(name="test parameter")
        c = pybamm.Scalar(3)
        eqn = a + b * c

        parameter_values = pybamm.ParameterValues({"test parameter": 2})
        eqn_changed = parameter_values.process_symbol(eqn)
        self.assertEqual(eqn_changed.evaluate(), 7)

        parameter_values = pybamm.ParameterValues({"test parameter": 3})
        eqn_changed_again = parameter_values.process_symbol(eqn_changed)
        self.assertEqual(eqn_changed_again.evaluate(), 10)

    def test_update_model(self):
        # standard model
        model1 = pybamm.ReactionDiffusionModel()
        modeltest1 = tests.StandardModelTest(model1)
        modeltest1.test_all()
        T1, Y1 = modeltest1.solver.t, modeltest1.solver.y

        # double initial conditions
        model2 = pybamm.ReactionDiffusionModel()
        modeltest2 = tests.StandardModelTest(model2)
        modeltest2.test_all()
        parameter_values_update = pybamm.ParameterValues({"Typical current density": 2})
        modeltest2.test_update_parameters(parameter_values_update)
        modeltest2.test_solving()
        T2, Y2 = modeltest2.solver.t, modeltest2.solver.y

        # results should be different
        self.assertNotEqual(np.linalg.norm(Y1 - Y2), 0)

    def test_update_geometry(self):
        # standard model
        model1 = pybamm.ReactionDiffusionModel()
        modeltest1 = tests.StandardModelTest(model1)
        modeltest1.test_all()
        T1, Y1 = modeltest1.solver.t, modeltest1.solver.y

        # results should be different
        # for idx in range(len(T1)):
        #     j1 = modeltest1.model.variables["Interfacial current density"].evaluate(
        #         T1[idx], Y1[:, idx]
        #     )
        #     j2 = modeltest2.model.variables["Interfacial current density"].evaluate(
        #         T2[idx], Y2[:, idx]
        #     )
        #     self.assertNotEqual(np.linalg.norm(j1 - j2), 0)
        # self.assertNotEqual(np.linalg.norm(Y1 - Y2), 0)

        # trying to update the model fails
        parameter_values_update = pybamm.ParameterValues(
            {
                "Negative electrode width": 0.5,
                "Separator width": 0.3,
                "Positive electrode width": 0.2,
            }
        )
        with self.assertRaisesRegex(ValueError, "geometry has changed"):
            modeltest1.test_update_parameters(parameter_values_update)


if __name__ == "__main__":
    import sys

    print("Add -v for more debug output")

    if "-v" in sys.argv:
        debug = True
    unittest.main()
