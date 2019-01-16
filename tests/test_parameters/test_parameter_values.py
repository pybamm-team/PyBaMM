#
# Tests for the Base Parameter Values class
#
import pybamm

import unittest
import numpy as np


class TestParameterValues(unittest.TestCase):
    def test_read_parameters_csv(self):
        data = pybamm.ParameterValues().read_parameters_csv(
            "input/parameters/lead-acid/default.csv"
        )
        self.assertEqual(data["R"], 8.314)

    def test_init(self):
        # from dict
        param = pybamm.ParameterValues({"a": 1})
        self.assertEqual(param["a"], 1)
        # from file
        param = pybamm.ParameterValues("input/parameters/lead-acid/default.csv")
        self.assertEqual(param["R"], 8.314)

    def test_overwrite(self):
        # from dicts
        param = pybamm.ParameterValues(
            base_parameters={"a": 1, "b": 2}, optional_parameters={"b": 3}
        )
        self.assertEqual(param["a"], 1)
        self.assertEqual(param["b"], 3)
        param.update({"a": 4})
        self.assertEqual(param["a"], 4)
        # from files
        param = pybamm.ParameterValues(
            base_parameters="input/parameters/lead-acid/default.csv",
            optional_parameters="input/parameters/lead-acid/optional_test.csv",
        )
        self.assertEqual(param["R"], 8.314)
        self.assertEqual(param["Ln"], 0.5)

    def test_get_parameter_value(self):
        parameter_values = pybamm.ParameterValues({"a": 1})
        param = pybamm.Parameter("a")
        self.assertEqual(parameter_values.get_parameter_value(param), 1)

    def test_process_symbol(self):
        parameter_values = pybamm.ParameterValues({"a": 1, "b": 2})
        # process parameter
        a = pybamm.Parameter("a")
        processed_a = parameter_values.process_symbol(a)
        self.assertTrue(isinstance(processed_a, pybamm.Scalar))
        self.assertEqual(processed_a.value, 1)

        # process binary operation
        b = pybamm.Parameter("b")
        sum = a + b
        processed_sum = parameter_values.process_symbol(sum)
        self.assertTrue(isinstance(processed_sum, pybamm.Addition))
        self.assertTrue(isinstance(processed_sum.children[0], pybamm.Scalar))
        self.assertTrue(isinstance(processed_sum.children[1], pybamm.Scalar))
        self.assertEqual(processed_sum.children[0].value, 1)
        self.assertEqual(processed_sum.children[1].value, 2)

        scal = pybamm.Scalar(34)
        mul = a * scal
        processed_mul = parameter_values.process_symbol(mul)
        self.assertTrue(isinstance(processed_mul, pybamm.Multiplication))
        self.assertTrue(isinstance(processed_mul.children[0], pybamm.Scalar))
        self.assertTrue(isinstance(processed_mul.children[1], pybamm.Scalar))
        self.assertEqual(processed_mul.children[0].value, 1)
        self.assertEqual(processed_mul.children[1].value, 34)

        # process unary operation
        grad = pybamm.Gradient(a)
        processed_grad = parameter_values.process_symbol(grad)
        self.assertTrue(isinstance(processed_grad, pybamm.Gradient))
        self.assertTrue(isinstance(processed_grad.children[0], pybamm.Scalar))
        self.assertEqual(processed_grad.children[0].value, 1)

        # process variable
        c = pybamm.Variable("c")
        processed_c = parameter_values.process_symbol(c)
        self.assertTrue(isinstance(processed_c, pybamm.Variable))
        self.assertEqual(processed_c.name, "c")

        # process scalar
        d = pybamm.Scalar(14)
        processed_d = parameter_values.process_symbol(d)
        self.assertTrue(isinstance(processed_d, pybamm.Scalar))
        self.assertEqual(processed_d.value, 14)

        # process array types
        e = pybamm.Vector(np.ones(4))
        processed_e = parameter_values.process_symbol(e)
        self.assertTrue(isinstance(processed_e, pybamm.Vector))
        np.testing.assert_array_equal(processed_e.evaluate(), np.ones(4))

        f = pybamm.Matrix(np.ones((5, 6)))
        processed_f = parameter_values.process_symbol(f)
        self.assertTrue(isinstance(processed_f, pybamm.Matrix))
        np.testing.assert_array_equal(processed_f.evaluate(), np.ones((5, 6)))

    def test_process_complex_expression(self):
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        par1 = pybamm.Parameter("par1")
        par2 = pybamm.Parameter("par2")
        scal1 = pybamm.Scalar("scal1")
        scal2 = pybamm.Scalar("scal2")
        expression = (scal1 * (par1 + var2)) / ((var1 - par2) + scal2)

        param = pybamm.ParameterValues(base_parameters={"par1": 1, "par2": 2})
        exp_param = param.process_symbol(expression)
        self.assertTrue(isinstance(exp_param, pybamm.Division))
        # left side
        self.assertTrue(isinstance(exp_param.children[0], pybamm.Multiplication))
        self.assertTrue(isinstance(exp_param.children[0].children[0], pybamm.Scalar))
        self.assertTrue(isinstance(exp_param.children[0].children[1], pybamm.Addition))
        self.assertTrue(
            isinstance(exp_param.children[0].children[1].children[0], pybamm.Scalar)
        )
        self.assertEqual(exp_param.children[0].children[1].children[0].value, 1)
        self.assertTrue(
            isinstance(exp_param.children[0].children[1].children[1], pybamm.Variable)
        )
        # right side
        self.assertTrue(isinstance(exp_param.children[1], pybamm.Addition))
        self.assertTrue(
            isinstance(exp_param.children[1].children[0], pybamm.Subtraction)
        )
        self.assertTrue(
            isinstance(exp_param.children[1].children[0].children[0], pybamm.Variable)
        )
        self.assertTrue(
            isinstance(exp_param.children[1].children[0].children[1], pybamm.Scalar)
        )
        self.assertEqual(exp_param.children[1].children[0].children[1].value, 2)
        self.assertTrue(isinstance(exp_param.children[1].children[1], pybamm.Scalar))

    def test_process_model(self):
        model = pybamm.BaseModel()
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")
        d = pybamm.Parameter("d")
        var = pybamm.Variable("var")
        model.rhs = {var: a * pybamm.grad(var)}
        model.initial_conditions = {var: b}
        model.boundary_conditions = {var: {"left": c, "right": d}}
        parameter_values = pybamm.ParameterValues({"a": 1, "b": 2, "c": 3, "d": 42})
        parameter_values.process_model(model)
        # rhs
        self.assertTrue(isinstance(model.rhs[var], pybamm.Multiplication))
        self.assertTrue(isinstance(model.rhs[var].children[0], pybamm.Scalar))
        self.assertTrue(isinstance(model.rhs[var].children[1], pybamm.Gradient))
        self.assertEqual(model.rhs[var].children[0].value, 1)
        # initial conditions
        self.assertTrue(isinstance(model.initial_conditions[var], pybamm.Scalar))
        self.assertEqual(model.initial_conditions[var].value, 2)
        # boundary conditions
        bc_key = list(model.boundary_conditions.keys())[0]
        self.assertTrue(isinstance(bc_key, pybamm.Variable))
        bc_value = list(model.boundary_conditions.values())[0]
        self.assertTrue(isinstance(bc_value["left"], pybamm.Scalar))
        self.assertEqual(bc_value["left"].value, 3)
        self.assertTrue(isinstance(bc_value["right"], pybamm.Scalar))
        self.assertEqual(bc_value["right"].value, 42)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
