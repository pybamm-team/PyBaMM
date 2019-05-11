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
        self.assertEqual(data["Reference temperature [K]"], 294.85)

    def test_init(self):
        # from dict
        param = pybamm.ParameterValues({"a": 1})
        self.assertEqual(param["a"], 1)
        # from file
        param = pybamm.ParameterValues("input/parameters/lead-acid/default.csv")
        self.assertEqual(param["Reference temperature [K]"], 294.85)

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
        self.assertEqual(param["Reference temperature [K]"], 294.85)
        self.assertEqual(param["Negative electrode width [m]"], 0.5)

    def test_process_symbol(self):
        parameter_values = pybamm.ParameterValues({"a": 1, "b": 2})
        # process parameter
        a = pybamm.Parameter("a")
        processed_a = parameter_values.process_symbol(a)
        self.assertIsInstance(processed_a, pybamm.Scalar)
        self.assertEqual(processed_a.value, 1)

        # process binary operation
        b = pybamm.Parameter("b")
        sum = a + b
        processed_sum = parameter_values.process_symbol(sum)
        self.assertIsInstance(processed_sum, pybamm.Addition)
        self.assertIsInstance(processed_sum.children[0], pybamm.Scalar)
        self.assertIsInstance(processed_sum.children[1], pybamm.Scalar)
        self.assertEqual(processed_sum.children[0].value, 1)
        self.assertEqual(processed_sum.children[1].value, 2)

        scal = pybamm.Scalar(34)
        mul = a * scal
        processed_mul = parameter_values.process_symbol(mul)
        self.assertIsInstance(processed_mul, pybamm.Multiplication)
        self.assertIsInstance(processed_mul.children[0], pybamm.Scalar)
        self.assertIsInstance(processed_mul.children[1], pybamm.Scalar)
        self.assertEqual(processed_mul.children[0].value, 1)
        self.assertEqual(processed_mul.children[1].value, 34)

        # process integral
        aa = pybamm.Parameter("a", domain=["negative electrode"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])
        integ = pybamm.Integral(aa, x)
        processed_integ = parameter_values.process_symbol(integ)
        self.assertIsInstance(processed_integ, pybamm.Integral)
        self.assertIsInstance(processed_integ.children[0], pybamm.Scalar)
        self.assertEqual(processed_integ.children[0].value, 1)
        self.assertEqual(processed_integ.integration_variable.id, x.id)

        # process unary operation
        grad = pybamm.Gradient(a)
        processed_grad = parameter_values.process_symbol(grad)
        self.assertIsInstance(processed_grad, pybamm.Gradient)
        self.assertIsInstance(processed_grad.children[0], pybamm.Scalar)
        self.assertEqual(processed_grad.children[0].value, 1)

        # process broadcast
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        broad = pybamm.Broadcast(a, whole_cell)
        processed_broad = parameter_values.process_symbol(broad)
        self.assertIsInstance(processed_broad, pybamm.Broadcast)
        self.assertEqual(processed_broad.domain, whole_cell)
        self.assertIsInstance(processed_broad.children[0], pybamm.Scalar)
        self.assertEqual(processed_broad.children[0].value, 1)

        # process concatenation
        conc = pybamm.Concatenation(a, b)
        processed_conc = parameter_values.process_symbol(conc)
        self.assertIsInstance(processed_conc.children[0], pybamm.Scalar)
        self.assertIsInstance(processed_conc.children[1], pybamm.Scalar)
        self.assertEqual(processed_conc.children[0].value, 1)
        self.assertEqual(processed_conc.children[1].value, 2)

        # process variable
        c = pybamm.Variable("c")
        processed_c = parameter_values.process_symbol(c)
        self.assertIsInstance(processed_c, pybamm.Variable)
        self.assertEqual(processed_c.name, "c")

        # process scalar
        d = pybamm.Scalar(14)
        processed_d = parameter_values.process_symbol(d)
        self.assertIsInstance(processed_d, pybamm.Scalar)
        self.assertEqual(processed_d.value, 14)

        # process array types
        e = pybamm.Vector(np.ones(4))
        processed_e = parameter_values.process_symbol(e)
        self.assertIsInstance(processed_e, pybamm.Vector)
        np.testing.assert_array_equal(processed_e.evaluate(), np.ones(4))

        f = pybamm.Matrix(np.ones((5, 6)))
        processed_f = parameter_values.process_symbol(f)
        self.assertIsInstance(processed_f, pybamm.Matrix)
        np.testing.assert_array_equal(processed_f.evaluate(), np.ones((5, 6)))

        # process statevector
        g = pybamm.StateVector(slice(0, 10))
        processed_g = parameter_values.process_symbol(g)
        self.assertIsInstance(processed_g, pybamm.StateVector)
        np.testing.assert_array_equal(processed_g.evaluate(y=np.ones(10)), np.ones(10))

        # not implemented
        sym = pybamm.Symbol("sym")
        with self.assertRaises(NotImplementedError):
            parameter_values.process_symbol(sym)

    def test_process_function_parameter(self):
        parameter_values = pybamm.ParameterValues(
            {
                "a": 3,
                "func": "process_symbol_test_function.py",
                "const": "process_symbol_test_constant_function.py",
            }
        )
        a = pybamm.Parameter("a")

        # process function
        func = pybamm.FunctionParameter("func", a)
        processed_func = parameter_values.process_symbol(func)
        self.assertIsInstance(processed_func, pybamm.Function)
        self.assertEqual(processed_func.evaluate(), 369)

        # process constant function
        const = pybamm.FunctionParameter("const", a)
        processed_const = parameter_values.process_symbol(const)
        self.assertIsInstance(processed_const, pybamm.Function)
        self.assertEqual(processed_const.evaluate(), 254)

        # process differentiated function parameter
        diff_func = func.diff(a)
        processed_diff_func = parameter_values.process_symbol(diff_func)
        self.assertEqual(processed_diff_func.evaluate(), 123)

    def test_process_inline_function_parameters(self):
        def D(c):
            return c ** 2

        parameter_values = pybamm.ParameterValues({"a": 3, "Diffusivity": D})

        a = pybamm.Parameter("a")
        func = pybamm.FunctionParameter("Diffusivity", a)

        processed_func = parameter_values.process_symbol(func)
        self.assertIsInstance(processed_func, pybamm.Function)
        self.assertEqual(processed_func.evaluate(), 9)

        # process differentiated function parameter
        diff_func = func.diff(a)
        processed_diff_func = parameter_values.process_symbol(diff_func)
        self.assertEqual(processed_diff_func.evaluate(), 6)

    def test_process_complex_expression(self):
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        par1 = pybamm.Parameter("par1")
        par2 = pybamm.Parameter("par2")
        scal1 = pybamm.Scalar(3)
        scal2 = pybamm.Scalar(4)
        expression = (scal1 * (par1 + var2)) / ((var1 - par2) + scal2)

        param = pybamm.ParameterValues(base_parameters={"par1": 1, "par2": 2})
        exp_param = param.process_symbol(expression)
        self.assertIsInstance(exp_param, pybamm.Division)
        # left side
        self.assertIsInstance(exp_param.children[0], pybamm.Multiplication)
        self.assertIsInstance(exp_param.children[0].children[0], pybamm.Scalar)
        self.assertIsInstance(exp_param.children[0].children[1], pybamm.Addition)
        self.assertTrue(
            isinstance(exp_param.children[0].children[1].children[0], pybamm.Scalar)
        )
        self.assertEqual(exp_param.children[0].children[1].children[0].value, 1)
        self.assertTrue(
            isinstance(exp_param.children[0].children[1].children[1], pybamm.Variable)
        )
        # right side
        self.assertIsInstance(exp_param.children[1], pybamm.Addition)
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
        self.assertIsInstance(exp_param.children[1].children[1], pybamm.Scalar)

    def test_process_model(self):
        model = pybamm.BaseModel()
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")
        d = pybamm.Parameter("d")
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        model.rhs = {var1: a * pybamm.grad(var1)}
        model.algebraic = {var2: c * var2}
        model.initial_conditions = {var1: b, var2: d}
        model.boundary_conditions = {
            var1: {"left": (c, "Dirichlet"), "right": (d, "Neumann")}
        }
        model.variables = {
            "var1": var1,
            "var2": var2,
            "grad_var1": pybamm.grad(var1),
            "d_var1": d * var1,
        }
        parameter_values = pybamm.ParameterValues({"a": 1, "b": 2, "c": 3, "d": 42})
        parameter_values.process_model(model)
        # rhs
        self.assertIsInstance(model.rhs[var1], pybamm.Multiplication)
        self.assertIsInstance(model.rhs[var1].children[0], pybamm.Scalar)
        self.assertIsInstance(model.rhs[var1].children[1], pybamm.Gradient)
        self.assertEqual(model.rhs[var1].children[0].value, 1)
        # algebraic
        self.assertIsInstance(model.algebraic[var2], pybamm.Multiplication)
        self.assertIsInstance(model.algebraic[var2].children[0], pybamm.Scalar)
        self.assertIsInstance(model.algebraic[var2].children[1], pybamm.Variable)
        self.assertEqual(model.algebraic[var2].children[0].value, 3)
        # initial conditions
        self.assertIsInstance(model.initial_conditions[var1], pybamm.Scalar)
        self.assertEqual(model.initial_conditions[var1].value, 2)
        # boundary conditions
        bc_key = list(model.boundary_conditions.keys())[0]
        self.assertIsInstance(bc_key, pybamm.Variable)
        bc_value = list(model.boundary_conditions.values())[0]
        self.assertIsInstance(bc_value["left"][0], pybamm.Scalar)
        self.assertEqual(bc_value["left"][0].value, 3)
        self.assertIsInstance(bc_value["right"][0], pybamm.Scalar)
        self.assertEqual(bc_value["right"][0].value, 42)
        # variables
        self.assertEqual(model.variables["var1"].id, var1.id)
        self.assertIsInstance(model.variables["grad_var1"], pybamm.Gradient)
        self.assertTrue(
            isinstance(model.variables["grad_var1"].children[0], pybamm.Variable)
        )
        self.assertEqual(
            model.variables["d_var1"].id, (pybamm.Scalar(42, name="d") * var1).id
        )
        self.assertIsInstance(model.variables["d_var1"].children[0], pybamm.Scalar)
        self.assertTrue(
            isinstance(model.variables["d_var1"].children[1], pybamm.Variable)
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
