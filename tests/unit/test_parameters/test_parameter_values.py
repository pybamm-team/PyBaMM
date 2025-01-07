#
# Tests for the Base Parameter Values class
#


import pytest
import os

import numpy as np
import pandas as pd

import pybamm
import tests.shared as shared
from pybamm.input.parameters.lithium_ion.Marquis2019 import (
    lico2_ocp_Dualfoil1998,
    lico2_diffusivity_Dualfoil1998,
)
from pybamm.expression_tree.exceptions import OptionError
import casadi
from pybamm.parameters.parameter_values import ParameterValues


class TestParameterValues:
    def test_init(self):
        # from dict
        param = pybamm.ParameterValues({"a": 1})
        assert param["a"] == 1
        assert "a" in param.keys()
        assert 1 in param.values()
        assert ("a", 1) in param.items()

        # from dict with strings
        param = pybamm.ParameterValues({"a": "1"})
        assert param["a"] == 1

        # from dict "chemistry" key gets removed
        param = pybamm.ParameterValues({"a": 1, "chemistry": "lithium-ion"})
        assert "chemistry" not in param.keys()

        # junk param values rejected
        with pytest.raises(ValueError, match="'Junk' is not a valid parameter set."):
            pybamm.ParameterValues("Junk")

    def test_repr(self):
        param = pybamm.ParameterValues({"a": 1})
        assert "'a': 1" in repr(param)
        assert param._ipython_key_completions_() == [
            "Ideal gas constant [J.K-1.mol-1]",
            "Faraday constant [C.mol-1]",
            "Boltzmann constant [J.K-1]",
            "Electron charge [C]",
            "a",
        ]

    def test_eq(self):
        assert pybamm.ParameterValues({"a": 1}) == pybamm.ParameterValues({"a": 1})

    def test_update(self):
        # equate values
        param = pybamm.ParameterValues({"a": 1})
        assert param["a"] == 1
        # no conflict
        param.update({"a": 2})
        assert param["a"] == 2
        param.update({"a": 2}, check_conflict=True)
        assert param["a"] == 2
        # with conflict
        param.update({"a": 3})
        # via __setitem__
        param["a"] = 2
        assert param["a"] == 2
        with pytest.raises(
            ValueError, match="parameter 'a' already defined with value '2'"
        ):
            param.update({"a": 4}, check_conflict=True)
        # with parameter not existing yet
        with pytest.raises(KeyError, match="Cannot update parameter"):
            param.update({"b": 1})

        # update with a ParameterValues object
        new_param = pybamm.ParameterValues(param)
        assert new_param["a"] == 2

        # test deleting a parameter
        del param["a"]
        assert "a" not in param.keys()

    def test_set_initial_stoichiometries(self):
        param = pybamm.ParameterValues("Chen2020")
        param.set_initial_stoichiometries(0.4)
        param_0 = param.set_initial_stoichiometries(0, inplace=False)
        param_100 = param.set_initial_stoichiometries(1, inplace=False)

        # check that the stoichiometry of param is linearly interpolated between
        # the min and max stoichiometries
        x = param["Initial concentration in negative electrode [mol.m-3]"]
        x_0 = param_0["Initial concentration in negative electrode [mol.m-3]"]
        x_100 = param_100["Initial concentration in negative electrode [mol.m-3]"]
        assert x == pytest.approx(x_0 + 0.4 * (x_100 - x_0))

        y = param["Initial concentration in positive electrode [mol.m-3]"]
        y_0 = param_0["Initial concentration in positive electrode [mol.m-3]"]
        y_100 = param_100["Initial concentration in positive electrode [mol.m-3]"]
        assert y == pytest.approx(y_0 - 0.4 * (y_0 - y_100))

    def test_set_initial_stoichiometry_half_cell(self):
        param = pybamm.lithium_ion.DFN(
            {"working electrode": "positive"}
        ).default_parameter_values
        param = param.set_initial_stoichiometry_half_cell(
            0.4, inplace=False, options={"working electrode": "positive"}
        )
        param_0 = param.set_initial_stoichiometry_half_cell(
            0, inplace=False, options={"working electrode": "positive"}
        )
        param_100 = param.set_initial_stoichiometry_half_cell(
            1, inplace=False, options={"working electrode": "positive"}
        )

        y = param["Initial concentration in positive electrode [mol.m-3]"]
        y_0 = param_0["Initial concentration in positive electrode [mol.m-3]"]
        y_100 = param_100["Initial concentration in positive electrode [mol.m-3]"]
        assert y == pytest.approx(y_0 - 0.4 * (y_0 - y_100))

        # inplace for 100% coverage
        param_t = pybamm.lithium_ion.DFN(
            {"working electrode": "positive"}
        ).default_parameter_values
        param_t.set_initial_stoichiometry_half_cell(
            0.4, inplace=True, options={"working electrode": "positive"}
        )
        y = param_t["Initial concentration in positive electrode [mol.m-3]"]
        param_0 = pybamm.lithium_ion.DFN(
            {"working electrode": "positive"}
        ).default_parameter_values
        param_0.set_initial_stoichiometry_half_cell(
            0, inplace=True, options={"working electrode": "positive"}
        )
        y_0 = param_0["Initial concentration in positive electrode [mol.m-3]"]
        param_100 = pybamm.lithium_ion.DFN(
            {"working electrode": "positive"}
        ).default_parameter_values
        param_100.set_initial_stoichiometry_half_cell(
            1, inplace=True, options={"working electrode": "positive"}
        )
        y_100 = param_100["Initial concentration in positive electrode [mol.m-3]"]
        assert y == pytest.approx(y_0 - 0.4 * (y_0 - y_100))

        # test error
        param = pybamm.ParameterValues("Chen2020")
        with pytest.raises(OptionError, match="working electrode"):
            param.set_initial_stoichiometry_half_cell(
                0.1, options={"working electrode": "negative"}
            )

    def test_set_initial_ocps(self):
        options = {
            "open-circuit potential": "MSMR",
            "particle": "MSMR",
            "number of MSMR reactions": ("6", "4"),
            "intercalation kinetics": "MSMR",
        }
        param_100 = pybamm.ParameterValues("MSMR_Example")
        param_100.set_initial_ocps(1, inplace=True, options=options)
        param_0 = param_100.set_initial_ocps(0, inplace=False, options=options)

        Un_0 = param_0["Initial voltage in negative electrode [V]"]
        Up_0 = param_0["Initial voltage in positive electrode [V]"]
        assert Up_0 - Un_0 == pytest.approx(2.8)

        Un_100 = param_100["Initial voltage in negative electrode [V]"]
        Up_100 = param_100["Initial voltage in positive electrode [V]"]
        assert Up_100 - Un_100 == pytest.approx(4.2)

    def test_check_parameter_values(self):
        with pytest.raises(ValueError, match="propotional term"):
            pybamm.ParameterValues(
                {"Negative electrode LAM constant propotional term": 1}
            )
            # The + character in "1 + dlnf/dlnc" is appended with a backslash (\+),
            # since + has other meanings in regex
        with pytest.raises(ValueError, match="Thermodynamic factor"):
            pybamm.ParameterValues({"1 + dlnf/dlnc": 1})

    def test_process_symbol(self):
        parameter_values = pybamm.ParameterValues({"a": 4, "b": 2, "c": 3})
        # process parameter
        a = pybamm.Parameter("a")
        processed_a = parameter_values.process_symbol(a)
        assert isinstance(processed_a, pybamm.Scalar)
        assert processed_a.value == 4

        # process binary operation
        var = pybamm.Variable("var")
        add = a + var
        processed_add = parameter_values.process_symbol(add)
        assert isinstance(processed_add, pybamm.Addition)
        assert isinstance(processed_add.children[0], pybamm.Scalar)
        assert isinstance(processed_add.children[1], pybamm.Variable)
        assert processed_add.children[0].value == 4

        b = pybamm.Parameter("b")
        add = a + b
        processed_add = parameter_values.process_symbol(add)
        assert isinstance(processed_add, pybamm.Scalar)
        assert processed_add.value == 6

        scal = pybamm.Scalar(34)
        mul = a * scal
        processed_mul = parameter_values.process_symbol(mul)
        assert isinstance(processed_mul, pybamm.Scalar)
        assert processed_mul.value == 136

        # process integral
        aa = pybamm.PrimaryBroadcast(pybamm.Parameter("a"), "negative electrode")
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])
        integ = pybamm.Integral(aa, x)
        processed_integ = parameter_values.process_symbol(integ)
        assert isinstance(processed_integ, pybamm.Integral)
        assert isinstance(processed_integ.children[0], pybamm.PrimaryBroadcast)
        assert processed_integ.children[0].child.value == 4
        assert processed_integ.integration_variable[0] == x

        # process unary operation
        v = pybamm.Variable("v", domain="test")
        grad = pybamm.Gradient(v)
        processed_grad = parameter_values.process_symbol(grad)
        assert isinstance(processed_grad, pybamm.Gradient)
        assert isinstance(processed_grad.children[0], pybamm.Variable)

        # process delta function
        aa = pybamm.Parameter("a")
        delta_aa = pybamm.DeltaFunction(aa, "left", "some domain")
        processed_delta_aa = parameter_values.process_symbol(delta_aa)
        assert isinstance(processed_delta_aa, pybamm.DeltaFunction)
        assert processed_delta_aa.side == "left"
        processed_a = processed_delta_aa.children[0]
        assert isinstance(processed_a, pybamm.Scalar)
        assert processed_a.value == 4

        # process boundary operator (test for BoundaryValue)
        aa = pybamm.Parameter("a")
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])
        boundary_op = pybamm.BoundaryValue(aa * x, "left")
        processed_boundary_op = parameter_values.process_symbol(boundary_op)
        assert isinstance(processed_boundary_op, pybamm.BoundaryOperator)
        processed_a = processed_boundary_op.children[0].children[0]
        processed_x = processed_boundary_op.children[0].children[1]
        assert isinstance(processed_a, pybamm.Scalar)
        assert processed_a.value == 4
        assert processed_x == x

        # process EvaluateAt
        evaluate_at = pybamm.EvaluateAt(x, aa)
        processed_evaluate_at = parameter_values.process_symbol(evaluate_at)
        assert isinstance(processed_evaluate_at, pybamm.EvaluateAt)
        assert processed_evaluate_at.children[0] == x
        assert processed_evaluate_at.position == 4
        with pytest.raises(ValueError, match="'position' in 'EvaluateAt'"):
            parameter_values.process_symbol(pybamm.EvaluateAt(x, x))

        # process broadcast
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        broad = pybamm.PrimaryBroadcast(a, whole_cell)
        processed_broad = parameter_values.process_symbol(broad)
        assert isinstance(processed_broad, pybamm.Broadcast)
        assert processed_broad.domain == whole_cell
        assert isinstance(processed_broad.children[0], pybamm.Scalar)
        assert processed_broad.children[0].evaluate() == 4

        # process concatenation
        conc = pybamm.concatenation(
            pybamm.Vector(np.ones(10), domain="test"),
            pybamm.Vector(2 * np.ones(15), domain="test 2"),
        )
        processed_conc = parameter_values.process_symbol(conc)
        assert isinstance(processed_conc.children[0], pybamm.Vector)
        assert isinstance(processed_conc.children[1], pybamm.Vector)
        np.testing.assert_array_equal(processed_conc.children[0].entries, 1)
        np.testing.assert_array_equal(processed_conc.children[1].entries, 2)

        # process domain concatenation
        c_e_n = pybamm.Variable("c_e_n", ["negative electrode"])
        c_e_s = pybamm.Variable("c_e_p", ["separator"])
        test_mesh = shared.get_mesh_for_testing()
        dom_con = pybamm.DomainConcatenation([a * c_e_n, b * c_e_s], test_mesh)
        processed_dom_con = parameter_values.process_symbol(dom_con)
        a_proc = processed_dom_con.children[0].children[0]
        b_proc = processed_dom_con.children[1].children[0]
        assert isinstance(a_proc, pybamm.Scalar)
        assert isinstance(b_proc, pybamm.Scalar)
        assert a_proc.value == 4
        assert b_proc.value == 2

        # process variable
        c = pybamm.Variable("c")
        processed_c = parameter_values.process_symbol(c)
        assert isinstance(processed_c, pybamm.Variable)
        assert processed_c.name == "c"

        # process scalar
        d = pybamm.Scalar(14)
        processed_d = parameter_values.process_symbol(d)
        assert isinstance(processed_d, pybamm.Scalar)
        assert processed_d.value == 14

        # process array types
        e = pybamm.Vector(np.ones(4))
        processed_e = parameter_values.process_symbol(e)
        assert isinstance(processed_e, pybamm.Vector)
        np.testing.assert_array_equal(processed_e.evaluate(), np.ones((4, 1)))

        f = pybamm.Matrix(np.ones((5, 6)))
        processed_f = parameter_values.process_symbol(f)
        assert isinstance(processed_f, pybamm.Matrix)
        np.testing.assert_array_equal(processed_f.evaluate(), np.ones((5, 6)))

        # process statevector
        g = pybamm.StateVector(slice(0, 10))
        processed_g = parameter_values.process_symbol(g)
        assert isinstance(processed_g, pybamm.StateVector)
        np.testing.assert_array_equal(
            processed_g.evaluate(y=np.ones(10)), np.ones((10, 1))
        )

        # not found
        with pytest.raises(KeyError):
            x = pybamm.Parameter("x")
            parameter_values.process_symbol(x)

        parameter_values = pybamm.ParameterValues({"x": np.nan})
        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            x = pybamm.Parameter("x")
            parameter_values.process_symbol(x)
        with pytest.raises(ValueError, match="possibly a function"):
            x = pybamm.FunctionParameter("x", {})
            parameter_values.process_symbol(x)

    def test_process_parameter_in_parameter(self):
        parameter_values = pybamm.ParameterValues(
            {"a": 2, "2a": pybamm.Parameter("a") * 2, "b": np.array([1, 2, 3])}
        )

        # process 2a parameter
        a = pybamm.Parameter("2a")
        processed_a = parameter_values.process_symbol(a)
        assert processed_a.evaluate() == 4

        # case where parameter can't be processed
        b = pybamm.Parameter("b")
        with pytest.raises(TypeError, match="Cannot process parameter"):
            parameter_values.process_symbol(b)

    def test_process_input_parameter(self):
        parameter_values = pybamm.ParameterValues(
            {"a": "[input]", "b": 3, "c times 2": pybamm.InputParameter("c") * 2}
        )
        # process input parameter
        a = pybamm.Parameter("a")
        processed_a = parameter_values.process_symbol(a)
        assert isinstance(processed_a, pybamm.InputParameter)
        assert processed_a.evaluate(inputs={"a": 5}) == 5

        # process binary operation
        b = pybamm.Parameter("b")
        add = a + b
        processed_add = parameter_values.process_symbol(add)
        assert processed_add == 3 + pybamm.InputParameter("a")

        # process complex input parameter
        c = pybamm.Parameter("c times 2")
        processed_c = parameter_values.process_symbol(c)
        assert processed_c.evaluate(inputs={"c": 5}) == 10

    def test_process_function_parameter(self):
        def test_function(var):
            return 123 * var

        parameter_values = pybamm.ParameterValues(
            {
                "a": 3,
                "func": test_function,
                "const": 254,
                "float_func": lambda x: 42,
                "mult": pybamm.InputParameter("b") * 5,
                "bad type": np.array([1, 2, 3]),
            }
        )
        a = pybamm.InputParameter("a")

        # process function
        func = pybamm.FunctionParameter("func", {"a": a})
        processed_func = parameter_values.process_symbol(func)
        assert processed_func.evaluate(inputs={"a": 3}) == 369

        # process constant function
        # this should work even if the parameter in the function is not provided
        const = pybamm.FunctionParameter(
            "const", {"a": pybamm.Parameter("not provided")}
        )
        processed_const = parameter_values.process_symbol(const)
        assert isinstance(processed_const, pybamm.Scalar)
        assert processed_const.evaluate() == 254

        # process case where parameter provided is a pybamm symbol
        # (e.g. a multiplication)
        mult = pybamm.FunctionParameter("mult", {"a": a})
        processed_mult = parameter_values.process_symbol(mult)
        assert processed_mult.evaluate(inputs={"a": 14, "b": 63}) == 63 * 5

        # process differentiated function parameter
        diff_func = func.diff(a)
        processed_diff_func = parameter_values.process_symbol(diff_func)
        assert processed_diff_func.evaluate(inputs={"a": 3}) == 123

        # make sure diff works, despite simplifications, when the child is constant
        a_const = pybamm.Scalar(3)
        func_const = pybamm.FunctionParameter("func", {"a": a_const})
        diff_func_const = func_const.diff(a_const)
        processed_diff_func_const = parameter_values.process_symbol(diff_func_const)
        assert processed_diff_func_const.evaluate() == 123

        # function parameter that returns a python float
        func = pybamm.FunctionParameter("float_func", {"a": a})
        processed_func = parameter_values.process_symbol(func)
        assert processed_func.evaluate() == 42

        # weird type raises error
        func = pybamm.FunctionParameter("bad type", {"a": a})
        with pytest.raises(TypeError, match="Parameter provided for"):
            parameter_values.process_symbol(func)

        # function itself as input (different to the variable being an input)
        parameter_values = pybamm.ParameterValues({"func": "[input]"})
        a = pybamm.Scalar(3)
        func = pybamm.FunctionParameter("func", {"a": a})
        processed_func = parameter_values.process_symbol(func)
        assert processed_func.evaluate(inputs={"func": 13}) == 13

        # make sure function keeps the domain of the original function

        def my_func(x):
            return 2 * x

        x = pybamm.SpatialVariable("x", "negative electrode")
        func = pybamm.FunctionParameter("func", {"x": x})

        parameter_values = pybamm.ParameterValues({"func": my_func})
        func1 = parameter_values.process_symbol(func)

        parameter_values = pybamm.ParameterValues({"func": pybamm.InputParameter("a")})
        func2 = parameter_values.process_symbol(func)

        parameter_values = pybamm.ParameterValues(
            {"func": pybamm.InputParameter("a", "negative electrode")}
        )
        func3 = parameter_values.process_symbol(func)

        assert func1.domains == func2.domains
        assert func1.domains == func3.domains

        # [function] is deprecated
        with pytest.raises(ValueError, match="[function]"):
            pybamm.ParameterValues({"func": "[function]something"})

    def test_process_inline_function_parameters(self):
        def D(c):
            return c**2

        parameter_values = pybamm.ParameterValues({"Diffusivity": D})

        a = pybamm.Scalar(3)
        func = pybamm.FunctionParameter("Diffusivity", {"a": a})

        processed_func = parameter_values.process_symbol(func)
        assert processed_func.evaluate() == 9

        # process differentiated function parameter
        diff_func = func.diff(a)
        processed_diff_func = parameter_values.process_symbol(diff_func)
        assert processed_diff_func.evaluate() == 6

    def test_multi_var_function_with_parameters(self):
        def D(a, b):
            return a * np.exp(b)

        parameter_values = pybamm.ParameterValues({"a": 3, "b": 0})
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        func = pybamm.Function(D, a, b)

        processed_func = parameter_values.process_symbol(func)
        # Function of scalars gets automatically simplified
        assert isinstance(processed_func, pybamm.Scalar)
        assert processed_func.evaluate() == 3

    def test_multi_var_function_parameter(self):
        def D(a, b):
            return a * pybamm.exp(b)

        parameter_values = pybamm.ParameterValues({"a": 3, "b": 0, "Diffusivity": D})

        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        func = pybamm.FunctionParameter("Diffusivity", {"a": a, "b": b})

        processed_func = parameter_values.process_symbol(func)
        assert processed_func.evaluate() == 3

    def test_process_interpolant(self):
        x = np.linspace(0, 10)[:, np.newaxis]
        data = np.hstack([x, 2 * x])
        parameter_values = pybamm.ParameterValues({"Times two": ("times two", data)})

        a = pybamm.InputParameter("a")
        func = pybamm.FunctionParameter("Times two", {"a": a})

        processed_func = parameter_values.process_symbol(func)
        assert isinstance(processed_func, pybamm.Interpolant)
        assert processed_func.evaluate(inputs={"a": 3.01}) == 6.02

        # interpolant defined up front
        interp = pybamm.Interpolant(data[:, 0], data[:, 1], a, interpolator="cubic")
        processed_interp = parameter_values.process_symbol(interp)
        assert processed_interp.evaluate(inputs={"a": 3.01}) == 6.02

        # process differentiated function parameter
        diff_interp = interp.diff(a)
        processed_diff_interp = parameter_values.process_symbol(diff_interp)
        assert processed_diff_interp.evaluate(inputs={"a": 3.01}) == 2

    def test_process_interpolant_2d(self):
        x_ = [np.linspace(0, 10), np.linspace(0, 20)]

        X = list(np.meshgrid(*x_, indexing="ij"))

        x = np.column_stack([el.reshape(-1, 1) for el in X])

        y = (2 * x).sum(axis=1)

        Y = y.reshape(*[len(el) for el in x_])

        data = x_, Y

        parameter_values = pybamm.ParameterValues({"Times two": ("times two", data)})

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        func = pybamm.FunctionParameter("Times two", {"a": a, "b": b})

        processed_func = parameter_values.process_symbol(func)
        assert isinstance(processed_func, pybamm.Interpolant)
        assert processed_func.evaluate(inputs={"a": 3.01, "b": 4.4}) == pytest.approx(
            14.82
        )

        # process differentiated function parameter
        # diff_func = func.diff(a)
        # processed_diff_func = parameter_values.process_symbol(diff_func)
        # self.assertEqual(processed_diff_func.evaluate(), 2)

        # interpolant defined up front
        interp2 = pybamm.Interpolant(data[0], data[1], children=(a, b))
        processed_interp2 = parameter_values.process_symbol(interp2)
        assert processed_interp2.evaluate(inputs={"a": 3.01, "b": 4.4}) == 14.82

        y3 = (3 * x).sum(axis=1)

        Y3 = y3.reshape(*[len(el) for el in x_])

        data3 = x_, Y3

        parameter_values = pybamm.ParameterValues(
            {"Times three": ("times three", data3)}
        )

        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        func = pybamm.FunctionParameter("Times three", {"a": a, "b": b})

        processed_func = parameter_values.process_symbol(func)
        assert isinstance(processed_func, pybamm.Interpolant)
        # self.assertEqual(processed_func.evaluate().flatten()[0], 22.23)
        np.testing.assert_almost_equal(
            processed_func.evaluate(inputs={"a": 3.01, "b": 4.4}).flatten()[0],
            22.23,
            decimal=4,
        )

        interp3 = pybamm.Interpolant(data3[0], data3[1], children=(a, b))
        processed_interp3 = parameter_values.process_symbol(interp3)
        # self.assertEqual(processed_interp3.evaluate().flatten()[0], 22.23)
        np.testing.assert_almost_equal(
            processed_interp3.evaluate(inputs={"a": 3.01, "b": 4.4}).flatten()[0],
            22.23,
            decimal=4,
        )

    def test_interpolant_against_function(self):
        parameter_values = pybamm.ParameterValues({"function": lico2_ocp_Dualfoil1998})
        # path is parent dir of this file
        path = os.path.abspath(os.path.dirname(__file__))
        lico2_ocv_example_data = pybamm.parameters.process_1D_data(
            "lico2_ocv_example.csv", path=path
        )

        def lico2_ocv_example(sto):
            name, (x, y) = lico2_ocv_example_data
            return pybamm.Interpolant(x, y, [sto], name=name)

        parameter_values.update(
            {"interpolation": lico2_ocv_example}, check_already_exists=False
        )

        a = pybamm.Scalar(0.6)
        func = pybamm.FunctionParameter("function", {"a": a})
        interp = pybamm.FunctionParameter("interpolation", {"a": a})

        processed_func = parameter_values.process_symbol(func)
        processed_interp = parameter_values.process_symbol(interp)
        np.testing.assert_array_almost_equal(
            processed_func.evaluate(), processed_interp.evaluate(), decimal=3
        )

    def test_interpolant_2d_from_json(self):
        parameter_values = pybamm.ParameterValues(
            {"function": lico2_diffusivity_Dualfoil1998}
        )
        # path is parent dir of this file
        path = os.path.abspath(os.path.dirname(__file__))
        lico2_diffusivity_Dualfoil1998_2D_data = pybamm.parameters.process_2D_data(
            "lico2_diffusivity_Dualfoil1998_2D.json", path=path
        )

        def lico2_diffusivity_Dualfoil1998_2D(c_s, T):
            name, (xs, y) = lico2_diffusivity_Dualfoil1998_2D_data
            return pybamm.Interpolant(xs, y, [c_s, T], name=name)

        parameter_values.update(
            {"interpolation": lico2_diffusivity_Dualfoil1998_2D},
            check_already_exists=False,
        )

        a = pybamm.Scalar(0.6)
        b = pybamm.Scalar(300.0)

        func = pybamm.FunctionParameter("function", {"a": a, "b": b})
        interp = pybamm.FunctionParameter("interpolation", {"a": a, "b": b})

        processed_func = parameter_values.process_symbol(func)
        processed_interp = parameter_values.process_symbol(interp)
        np.testing.assert_array_almost_equal(
            processed_func.evaluate(), processed_interp.evaluate(), decimal=4
        )

    def test_process_interpolant_3D_from_csv(self):
        name = "data_for_testing_3D"
        path = os.path.abspath(os.path.dirname(__file__))
        processed = pybamm.parameters.process_3D_data_csv(name, path)
        parameter_values = pybamm.ParameterValues({"interpolation": processed})

        x1 = pybamm.StateVector(slice(0, 1))
        x2 = pybamm.StateVector(slice(1, 2))
        x3 = pybamm.StateVector(slice(2, 3))
        interpolation = pybamm.FunctionParameter(
            "interpolation", {"x1": x1, "x2": x2, "x3": x3}
        )

        processed_interpolation = parameter_values.process_symbol(interpolation)

        filename, name = pybamm.parameters.process_parameter_data._process_name(
            name, path, ".csv"
        )
        raw_df = pd.read_csv(filename)

        # It's also helpful to check the casadi conversion here aswell
        # We check elsewhere but this helps catch additional bugs
        casadi_y = casadi.MX.sym("y", 3)
        interp_casadi = processed_interpolation.to_casadi(y=casadi_y)
        casadi_f = casadi.Function("f", [casadi_y], [interp_casadi])

        # check that passing the input columns give the correct output
        for values in raw_df.values:
            y = np.array([values[0], values[1], values[2]])
            f = values[3]
            casadi_sol = casadi_f(y)

            np.testing.assert_almost_equal(
                processed_interpolation.evaluate(y=y)[0][0],
                f,
                decimal=10,
            )

            np.testing.assert_almost_equal(
                f,
                casadi_sol.__float__(),
                decimal=10,
            )

    def test_process_interpolant_2D_from_csv(self):
        name = "data_for_testing_2D"
        path = os.path.abspath(os.path.dirname(__file__))
        processed = pybamm.parameters.process_2D_data_csv(name, path)
        parameter_values = pybamm.ParameterValues({"interpolation": processed})

        x1 = pybamm.StateVector(slice(0, 1))
        x2 = pybamm.StateVector(slice(1, 2))
        interpolation = pybamm.FunctionParameter("interpolation", {"x1": x1, "x2": x2})
        processed_interpolation = parameter_values.process_symbol(interpolation)

        # It's also helpful to check the casadi conversion here aswell
        # We check elsewhere but this helps catch additional bugs
        casadi_y = casadi.MX.sym("y", 2)
        interp_casadi = processed_interpolation.to_casadi(y=casadi_y)
        casadi_f = casadi.Function("f", [casadi_y], [interp_casadi])

        filename, name = pybamm.parameters.process_parameter_data._process_name(
            name, path, ".csv"
        )
        raw_df = pd.read_csv(filename)

        # check that passing the input columns give the correct output
        for values in raw_df.values:
            y = np.array([values[0], values[1]])
            f = values[2]

            casadi_sol = casadi_f(y)

            np.testing.assert_almost_equal(
                processed_interpolation.evaluate(y=y)[0][0],
                f,
                decimal=10,
            )

            np.testing.assert_almost_equal(
                f,
                casadi_sol.__float__(),
                decimal=10,
            )

    def test_process_integral_broadcast(self):
        # Test that the x-average of a broadcast gets processed correctly
        var = pybamm.Variable("var", domain="negative electrode")
        func = pybamm.x_average(pybamm.FunctionParameter("func", {"var": var}))

        param = pybamm.ParameterValues({"func": 2})
        func_proc = param.process_symbol(func)

        assert func_proc == pybamm.Scalar(2, name="func")

        # test with auxiliary domains

        # secondary
        var = pybamm.Variable(
            "var",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        func = pybamm.x_average(pybamm.FunctionParameter("func", {"var": var}))

        param = pybamm.ParameterValues({"func": 2})
        func_proc = param.process_symbol(func)

        assert func_proc == pybamm.PrimaryBroadcast(
            pybamm.Scalar(2, name="func"), "current collector"
        )

        # secondary and tertiary
        var = pybamm.Variable(
            "var",
            domain="negative particle",
            auxiliary_domains={
                "secondary": "negative electrode",
                "tertiary": "current collector",
            },
        )
        func = pybamm.x_average(pybamm.FunctionParameter("func", {"var": var}))

        param = pybamm.ParameterValues({"func": 2})
        func_proc = param.process_symbol(func)

        assert func_proc == pybamm.FullBroadcast(
            pybamm.Scalar(2, name="func"), "negative particle", "current collector"
        )

        # secondary, tertiary and quaternary
        var = pybamm.Variable(
            "var",
            domain="negative particle",
            auxiliary_domains={
                "secondary": "negative particle size",
                "tertiary": "negative electrode",
                "quaternary": "current collector",
            },
        )
        func = pybamm.x_average(pybamm.FunctionParameter("func", {"var": var}))

        param = pybamm.ParameterValues({"func": 2})
        func_proc = param.process_symbol(func)

        assert func_proc == pybamm.FullBroadcast(
            pybamm.Scalar(2, name="func"),
            "negative particle",
            {
                "secondary": "negative particle size",
                "tertiary": "current collector",
            },
        )

        # special case for integral of concatenations of broadcasts
        var_n = pybamm.Variable("var_n", domain="negative electrode")
        var_s = pybamm.Variable("var_s", domain="separator")
        var_p = pybamm.Variable("var_p", domain="positive electrode")
        func_n = pybamm.FunctionParameter("func_n", {"var_n": var_n})
        func_s = pybamm.FunctionParameter("func_s", {"var_s": var_s})
        func_p = pybamm.FunctionParameter("func_p", {"var_p": var_p})

        func = pybamm.x_average(pybamm.concatenation(func_n, func_s, func_p))
        param = pybamm.ParameterValues(
            {
                "func_n": 2,
                "func_s": 3,
                "func_p": 4,
                "Negative electrode thickness [m]": 1,
                "Separator thickness [m]": 1,
                "Positive electrode thickness [m]": 1,
            }
        )
        func_proc = param.process_symbol(func)

        assert func_proc == pybamm.Scalar(3)

        # with auxiliary domains
        var_n = pybamm.Variable(
            "var_n",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        var_s = pybamm.Variable(
            "var_s",
            domain="separator",
            auxiliary_domains={"secondary": "current collector"},
        )
        var_p = pybamm.Variable(
            "var_p",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        func_n = pybamm.FunctionParameter("func_n", {"var_n": var_n})
        func_s = pybamm.FunctionParameter("func_s", {"var_s": var_s})
        func_p = pybamm.FunctionParameter("func_p", {"var_p": var_p})

        func = pybamm.x_average(pybamm.concatenation(func_n, func_s, func_p))
        param = pybamm.ParameterValues(
            {
                "func_n": 2,
                "func_s": 3,
                "func_p": 4,
                "Negative electrode thickness [m]": 1,
                "Separator thickness [m]": 1,
                "Positive electrode thickness [m]": 1,
            }
        )
        func_proc = param.process_symbol(func)

        assert func_proc == pybamm.PrimaryBroadcast(
            pybamm.Scalar(3), "current collector"
        )

    def test_process_size_average(self):
        # Test that the x-average of a broadcast gets processed correctly
        var = pybamm.Variable("var", domain="negative particle size")
        var_av = pybamm.size_average(var)

        def dist(R):
            return R**2

        param = pybamm.ParameterValues(
            {
                "Negative particle radius [m]": 2,
                "Negative area-weighted particle-size distribution [m-1]": dist,
                "Negative electrode thickness [m]": 3,
                "Separator thickness [m]": 4,
                "Positive electrode thickness [m]": 5,
            }
        )
        var_av_proc = param.process_symbol(var_av)

        assert isinstance(var_av_proc, pybamm.SizeAverage)
        R = pybamm.SpatialVariable("R_n", "negative particle size")
        assert var_av_proc.f_a_dist == R**2

    def test_process_not_constant(self):
        param = pybamm.ParameterValues({"a": 4})

        a = pybamm.NotConstant(pybamm.Parameter("a"))
        assert isinstance(param.process_symbol(a), pybamm.NotConstant)
        assert param.process_symbol(a).evaluate() == 4

    def test_process_complex_expression(self):
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        par1 = pybamm.Parameter("par1")
        par2 = pybamm.Parameter("par2")
        expression = (3 * (par1**var2)) / ((var1 - par2) + var2)

        param = pybamm.ParameterValues({"par1": 2, "par2": 4})
        exp_param = param.process_symbol(expression)
        assert exp_param == 3.0 * (2.0**var2) / ((-4.0 + var1) + var2)

    def test_process_model(self):
        model = pybamm.BaseModel()
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")
        d = pybamm.Parameter("d")
        var1 = pybamm.Variable("var1", domain="test")
        var2 = pybamm.Variable("var2", domain="test")
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
        assert isinstance(model.rhs[var1], pybamm.Gradient)
        # algebraic
        assert isinstance(model.algebraic[var2], pybamm.Multiplication)
        assert isinstance(model.algebraic[var2].children[0], pybamm.Scalar)
        assert isinstance(model.algebraic[var2].children[1], pybamm.Variable)
        assert model.algebraic[var2].children[0].value == 3
        # initial conditions
        assert isinstance(model.initial_conditions[var1], pybamm.Scalar)
        assert model.initial_conditions[var1].value == 2
        # boundary conditions
        bc_key = next(iter(model.boundary_conditions.keys()))
        assert isinstance(bc_key, pybamm.Variable)
        bc_value = next(iter(model.boundary_conditions.values()))
        assert isinstance(bc_value["left"][0], pybamm.Scalar)
        assert bc_value["left"][0].value == 3
        assert isinstance(bc_value["right"][0], pybamm.Scalar)
        assert bc_value["right"][0].value == 42
        # variables
        assert model.variables["var1"] == var1
        assert isinstance(model.variables["grad_var1"], pybamm.Gradient)
        assert isinstance(model.variables["grad_var1"].children[0], pybamm.Variable)
        assert model.variables["d_var1"] == (pybamm.Scalar(42, name="d") * var1)
        assert isinstance(model.variables["d_var1"].children[0], pybamm.Scalar)
        assert isinstance(model.variables["d_var1"].children[1], pybamm.Variable)

        # bad boundary conditions
        model = pybamm.BaseModel()
        model.algebraic = {var1: var1}
        x = pybamm.Parameter("x")
        model.boundary_conditions = {var1: {"left": (x, "Dirichlet")}}
        with pytest.raises(KeyError):
            parameter_values.process_model(model)

    def test_process_geometry(self):
        var = pybamm.Variable("var")
        geometry = {"negative electrode": {"x": {"min": 0, "max": var}}}
        with pytest.raises(ValueError, match="Geometry parameters must be Scalars"):
            pybamm.ParameterValues({}).process_geometry(geometry)

    def test_inplace(self):
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        new_model = param.process_model(model, inplace=False)

        V = model.variables["Voltage [V]"]
        assert V.has_symbol_of_classes(pybamm.Parameter)

        V = new_model.variables["Voltage [V]"]
        assert not V.has_symbol_of_classes(pybamm.Parameter)

    def test_process_empty_model(self):
        model = pybamm.BaseModel()
        parameter_values = pybamm.ParameterValues({"a": 1, "b": 2, "c": 3, "d": 42})
        with pytest.raises(
            pybamm.ModelError, match="Cannot process parameters for empty model"
        ):
            parameter_values.process_model(model)

    def test_evaluate(self):
        parameter_values = pybamm.ParameterValues({"a": 1, "b": 2, "c": 3})
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")
        assert parameter_values.evaluate(a) == 1
        assert parameter_values.evaluate(a + (b * c)) == 7
        d = pybamm.Parameter("a") + pybamm.Parameter("b") * pybamm.Array([4, 5])
        np.testing.assert_array_equal(
            parameter_values.evaluate(d), np.array([9, 11])[:, np.newaxis]
        )

        y = pybamm.StateVector(slice(0, 1))
        with pytest.raises(ValueError):
            parameter_values.evaluate(y)

    def test_exchange_current_density_plating(self):
        parameter_values = pybamm.ParameterValues(
            {"Exchange-current density for plating [A.m-2]": 1}
        )
        param = pybamm.Parameter(
            "Exchange-current density for lithium metal electrode [A.m-2]"
        )
        with pytest.raises(
            KeyError,
            match="referring to the reaction at the surface of a lithium metal electrode",
        ):
            parameter_values.evaluate(param)

    def test_contains_method(self):
        """Test for __contains__ method to check the functionality of 'in' keyword"""
        parameter_values = ParameterValues(
            {"Negative particle radius [m]": 1e-6, "Positive particle radius [m]": 2e-6}
        )
        assert (
            "Negative particle radius [m]" in parameter_values
        ), "Key should be found in parameter_values"
        assert (
            "Invalid key" not in parameter_values
        ), "Non-existent key should not be found"

    def test_iter_method(self):
        """Test for __iter__ method to check if we can iterate over keys"""
        parameter_values = ParameterValues(
            values={"Negative particle radius [m]": 1e-6}
        )
        pv = [i for i in parameter_values]
        assert len(pv) == 5, "Should have 5 keys"
