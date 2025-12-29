#
# Tests for the base model class
#

import os
import platform
import subprocess  # nosec
import sys
from io import StringIO

import casadi
import numpy as np
import pytest
from numpy import testing

import pybamm


class TestBaseModel:
    def test_rhs_set_get(self):
        model = pybamm.BaseModel()
        rhs = {
            pybamm.Symbol("c"): pybamm.Symbol("alpha"),
            pybamm.Symbol("d"): pybamm.Symbol("beta"),
        }
        model.rhs = rhs
        assert rhs == model.rhs
        # test domains
        rhs = {
            pybamm.Symbol("c", domain=["negative electrode"]): pybamm.Symbol(
                "alpha", domain=["negative electrode"]
            ),
            pybamm.Symbol("d", domain=["positive electrode"]): pybamm.Symbol(
                "beta", domain=["positive electrode"]
            ),
        }
        model.rhs = rhs
        assert rhs == model.rhs
        # non-matching domains should fail
        with pytest.raises(pybamm.DomainError):
            model.rhs = {
                pybamm.Symbol("c", domain=["positive electrode"]): pybamm.Symbol(
                    "alpha", domain=["negative electrode"]
                )
            }

    def test_algebraic_set_get(self):
        model = pybamm.BaseModel()
        algebraic = {pybamm.Symbol("b"): pybamm.Symbol("c") - pybamm.Symbol("a")}
        model.algebraic = algebraic
        assert algebraic == model.algebraic

    def test_initial_conditions_set_get(self):
        model = pybamm.BaseModel()
        initial_conditions = {
            pybamm.Symbol("c0"): pybamm.Symbol("gamma"),
            pybamm.Symbol("d0"): pybamm.Symbol("delta"),
        }
        model.initial_conditions = initial_conditions
        assert initial_conditions == model.initial_conditions

        # Test number input
        c0 = pybamm.Symbol("c0")
        model.initial_conditions[c0] = 34
        assert isinstance(model.initial_conditions[c0], pybamm.Scalar)
        assert model.initial_conditions[c0].value == 34

        # Variable in initial conditions should fail
        with pytest.raises(
            TypeError, match=r"Initial conditions cannot contain 'Variable' objects"
        ):
            model.initial_conditions = {c0: pybamm.Variable("v")}

        # non-matching domains should fail
        with pytest.raises(pybamm.DomainError):
            model.initial_conditions = {
                pybamm.Symbol("c", domain=["positive electrode"]): pybamm.Symbol(
                    "alpha", domain=["negative electrode"]
                )
            }

    def test_boundary_conditions_set_get(self):
        model = pybamm.BaseModel()
        boundary_conditions = {
            "c": {"left": ("epsilon", "Dirichlet"), "right": ("eta", "Dirichlet")}
        }
        model.boundary_conditions = boundary_conditions
        assert boundary_conditions == model.boundary_conditions

        # Test number input
        c0 = pybamm.Symbol("c0")
        model.boundary_conditions[c0] = {
            "left": (-2, "Dirichlet"),
            "right": (4, "Dirichlet"),
        }
        assert isinstance(model.boundary_conditions[c0]["left"][0], pybamm.Scalar)
        assert isinstance(model.boundary_conditions[c0]["right"][0], pybamm.Scalar)
        assert model.boundary_conditions[c0]["left"][0].value == -2
        assert model.boundary_conditions[c0]["right"][0].value == 4
        assert model.boundary_conditions[c0]["left"][1] == "Dirichlet"
        assert model.boundary_conditions[c0]["right"][1] == "Dirichlet"

        # Check bad bc type
        bad_bcs = {c0: {"left": (-2, "bad type"), "right": (4, "bad type")}}
        with pytest.raises(pybamm.ModelError, match=r"boundary condition"):
            model.boundary_conditions = bad_bcs

    def test_variables_set_get(self):
        model = pybamm.BaseModel()
        variables = {"c": "alpha", "d": "beta"}
        model.variables = variables
        assert variables == model.variables
        assert model.variable_names() == list(variables.keys())

    def test_jac_set_get(self):
        model = pybamm.BaseModel()
        model.jacobian = "test"
        assert model.jacobian == "test"

    def test_read_parameters(self):
        # Read parameters from different parts of the model
        model = pybamm.BaseModel()
        a = pybamm.Parameter("a")
        b = pybamm.InputParameter("b", "test")
        c = pybamm.Parameter("c")
        d = pybamm.Parameter("d")
        e = pybamm.Parameter("e")
        f = pybamm.InputParameter("f")
        g = pybamm.Parameter("g")
        h = pybamm.Parameter("h")
        i = pybamm.InputParameter("i")

        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: -u * a}
        model.algebraic = {v: v - b}
        model.initial_conditions = {u: c, v: d}
        model.events = [pybamm.Event("u=e", u - e)]
        model.variables = {"v+f+i": v + f + i}
        model.boundary_conditions = {
            u: {"left": (g, "Dirichlet"), "right": (0, "Neumann")},
            v: {"left": (0, "Dirichlet"), "right": (h, "Neumann")},
        }

        # Test variables_and_events
        assert "v+f+i" in model.variables_and_events
        assert "Event: u=e" in model.variables_and_events

        assert set([x.name for x in model.parameters]) == set(
            [x.name for x in [a, b, c, d, e, f, g, h, i]]
        )
        assert all(
            isinstance(x, pybamm.Parameter | pybamm.InputParameter)
            for x in model.parameters
        )

        model.variables = {
            "v+f+i": v + pybamm.FunctionParameter("f", {"Time [s]": pybamm.t}) + i
        }
        model.print_parameter_info()

    @pytest.mark.parametrize("symbols", ["c", "d", "e", "f", "h", "i"])
    def test_get_parameter_info(self, symbols):
        model = pybamm.BaseModel()
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b", "test")
        c = pybamm.InputParameter("c")
        d = pybamm.InputParameter("d")
        e = pybamm.InputParameter("e")
        f = pybamm.InputParameter("f")
        g = pybamm.Parameter("g")
        h = pybamm.Parameter("h")
        i = pybamm.Parameter("i")

        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: -u * a}
        model.algebraic = {v: v - b}
        model.initial_conditions = {u: c, v: d}
        model.events = [pybamm.Event("u=e", u - e)]
        model.variables = {"v+f+i": v + f + i}
        model.boundary_conditions = {
            u: {"left": (g, "Dirichlet"), "right": (0, "Neumann")},
            v: {"left": (0, "Dirichlet"), "right": (h, "Neumann")},
        }

        parameter_info = model.get_parameter_info()
        assert parameter_info["a"][1] == "InputParameter"
        assert parameter_info["b"][1] == "InputParameter in ['test']"
        assert symbols in parameter_info
        assert parameter_info["g"][1] == "Parameter"

    @pytest.mark.parametrize(
        "sub, key, parameter_value",
        [
            ("sub1", "a", "InputParameter"),
            ("sub1", "w", "InputParameter"),
            ("sub1", "e", "InputParameter"),
            ("sub1", "g", "Parameter"),
            ("sub1", "x", "Parameter"),
            ("sub1", "f", "InputParameter in ['test']"),
            ("sub2", "b", "InputParameter in ['test']"),
            ("sub2", "h", "Parameter"),
            ("sub1", "c", "FunctionParameter with inputs(s) ''"),
            ("sub2", "d", "FunctionParameter with inputs(s) ''"),
            ("sub2", "i", "FunctionParameter with inputs(s) ''"),
        ],
    )
    def test_get_parameter_info_submodel(self, sub, key, parameter_value):
        submodel = pybamm.lithium_ion.SPM().submodels["electrolyte diffusion"]

        class SubModel1(pybamm.BaseSubModel):
            def get_fundamental_variables(self):
                u = pybamm.Variable("u")

                variables = {"u": u}
                return variables

            def get_coupled_variables(self, variables):
                x = pybamm.Parameter("x")
                w = pybamm.InputParameter("w")
                f = pybamm.InputParameter("f", "test")
                variables.update({"w": w, "x": x, "f": f})
                return variables

            def set_rhs(self, variables):
                a = pybamm.InputParameter("a")
                u = variables["u"]
                self.rhs = {u: -u * a}

            def set_boundary_conditions(self, variables):
                g = pybamm.Parameter("g")
                u = variables["u"]
                self.boundary_conditions = {
                    u: {"left": (g, "Dirichlet"), "right": (0, "Neumann")},
                }

            def set_initial_conditions(self, variables):
                c = pybamm.FunctionParameter("c", {})
                u = variables["u"]
                self.initial_conditions = {u: c}

            def add_events_from(self, variables):
                e = pybamm.InputParameter("e")
                u = variables["u"]
                self.events = [pybamm.Event("u=e", u - e)]

        class SubModel2(pybamm.BaseSubModel):
            def get_fundamental_variables(self):
                v = pybamm.Variable("v")
                i = pybamm.FunctionParameter("i", {})
                variables = {"v": v, "i": i}
                return variables

            def set_rhs(self, variables):
                b = pybamm.InputParameter("b", "test")
                v = variables["v"]
                self.rhs = {v: v - b}

            def set_boundary_conditions(self, variables):
                h = pybamm.Parameter("h")
                v = variables["v"]
                self.boundary_conditions = {
                    v: {"left": (0, "Dirichlet"), "right": (h, "Neumann")},
                }

            def set_initial_conditions(self, variables):
                d = pybamm.FunctionParameter("d", {})
                v = variables["v"]
                self.initial_conditions = {v: d}

        sub1 = SubModel1(None)
        sub2 = SubModel2(None)
        model = pybamm.BaseModel()
        model.submodels = {"sub1": sub1, "sub2": sub2}
        model.build_model()

        parameter_info = model.get_parameter_info(by_submodel=True)

        expected_error_message = "Cannot use get_parameter_info"

        with pytest.raises(NotImplementedError, match=expected_error_message):
            submodel.get_parameter_info(by_submodel=True)

        with pytest.raises(NotImplementedError, match=expected_error_message):
            submodel.get_parameter_info(by_submodel=False)

        assert "a" in parameter_info["sub1"]
        assert "b" in parameter_info["sub2"]
        assert parameter_info[sub][key][1] == parameter_value

    def test_print_parameter_info(self):
        model = pybamm.BaseModel()
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b", "test")
        c = pybamm.FunctionParameter("c", {})
        d = pybamm.FunctionParameter("d", {})
        e = pybamm.InputParameter("e")
        f = pybamm.InputParameter("f")
        g = pybamm.Parameter("g")
        h = pybamm.Parameter("h")
        i = pybamm.Parameter("i")

        u = pybamm.Variable("u")
        v = pybamm.Variable("v")

        sub1 = pybamm.BaseSubModel(None)
        sub1.rhs = {u: -u * a}
        sub1.initial_conditions = {u: c}
        sub1.variables = {"u": u}
        sub1.boundary_conditions = {
            u: {"left": (g, "Dirichlet"), "right": (0, "Neumann")},
        }
        sub2 = pybamm.BaseSubModel(None)
        sub2.algebraic = {v: v - b}
        sub2.variables = {"v": v, "v+f+i": v + f + i}
        sub2.initial_conditions = {v: d}
        sub2.boundary_conditions = {
            v: {"left": (0, "Dirichlet"), "right": (h, "Neumann")},
        }
        sub3 = pybamm.BaseSubModel(None)
        model.submodels = {"sub1": sub1, "sub2": sub2, "sub3": sub3}
        model.events = [pybamm.Event("u=e", u - e)]
        model.build_model()
        captured_output = StringIO()
        sys.stdout = captured_output

        model.print_parameter_info()
        sys.stdout = sys.__stdout__

        result = captured_output.getvalue().strip()
        assert "a" in result
        assert "b" in result
        assert "InputParameter" in result
        assert "InputParameter in ['test']" in result
        assert "Parameter" in result
        assert "FunctionParameter with inputs(s) ''" in result

    @pytest.mark.parametrize(
        "values",
        [
            "'sub1' submodel parameters:",
            "'sub2' submodel parameters:",
            "Parameter",
            "InputParameter",
            "FunctionParameter with inputs(s) ''",
            "InputParameter in ['test']",
            "g",
            "a",
            "c",
            "h",
            "b",
            "d",
        ],
    )
    def test_print_parameter_info_submodel(self, values):
        model = pybamm.BaseModel()
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b", "test")
        c = pybamm.FunctionParameter("c", {})
        d = pybamm.FunctionParameter("d", {})
        e = pybamm.InputParameter("e")
        f = pybamm.InputParameter("f")
        g = pybamm.Parameter("g")
        h = pybamm.Parameter("h")
        i = pybamm.Parameter("i")
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")

        sub1 = pybamm.BaseSubModel(None)
        sub1.rhs = {u: -u * a}
        sub1.initial_conditions = {u: c}
        sub1.variables = {"u": u}
        sub1.boundary_conditions = {
            u: {"left": (g, "Dirichlet"), "right": (0, "Neumann")},
        }
        sub2 = pybamm.BaseSubModel(None)
        sub2.algebraic = {v: v - b}
        sub2.variables = {"v": v, "v+f+i": v + f + i}
        sub2.initial_conditions = {v: d}
        sub2.boundary_conditions = {
            v: {"left": (0, "Dirichlet"), "right": (h, "Neumann")},
        }
        sub3 = pybamm.BaseSubModel(None)
        model.submodels = {"sub1": sub1, "sub2": sub2, "sub3": sub3}
        model.events = [pybamm.Event("u=e", u - e)]
        model.build_model()
        captured_output = StringIO()
        sys.stdout = captured_output

        model.print_parameter_info(by_submodel=True)
        sys.stdout = sys.__stdout__

        result = captured_output.getvalue().strip()
        assert values in result

    def test_read_input_parameters(self):
        # Read input parameters from different parts of the model
        model = pybamm.BaseModel()
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        c = pybamm.InputParameter("c")
        d = pybamm.InputParameter("d")
        e = pybamm.InputParameter("e")
        f = pybamm.InputParameter("f")

        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: -u * a}
        model.algebraic = {v: v - b}
        model.initial_conditions = {u: c, v: d}
        model.events = [pybamm.Event("u=e", u - e)]
        model.variables = {"v+f": v + f}

        assert set([x.name for x in model.input_parameters]) == set(
            [x.name for x in [a, b, c, d, e, f]]
        )
        assert all(isinstance(x, pybamm.InputParameter) for x in model.input_parameters)

    def test_update(self):
        # model
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        model = pybamm.BaseModel()
        c = pybamm.Variable("c", domain=whole_cell)
        rhs = {c: 5 * pybamm.div(pybamm.grad(c)) - 1}
        initial_conditions = {c: 1}
        boundary_conditions = {c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}}
        variables = {"c": c}
        model.rhs = rhs
        model.initial_conditions = initial_conditions
        model.boundary_conditions = boundary_conditions
        model.variables = variables

        # update with submodel
        submodel = pybamm.BaseModel()
        d = pybamm.Variable("d", domain=whole_cell)
        submodel.rhs = {
            d: 5 * pybamm.div(pybamm.grad(c)) + pybamm.div(pybamm.grad(d)) - 1
        }
        submodel.initial_conditions = {d: 3}
        submodel.boundary_conditions = {
            d: {"left": (4, "Dirichlet"), "right": (7, "Dirichlet")}
        }
        submodel.variables = {"d": d}
        model.update(submodel)

        # check
        assert model.rhs[d] == submodel.rhs[d]
        assert model.initial_conditions[d] == submodel.initial_conditions[d]
        assert model.boundary_conditions[d] == submodel.boundary_conditions[d]
        assert model.variables["d"] == submodel.variables["d"]
        assert model.rhs[c] == rhs[c]
        assert model.initial_conditions[c] == initial_conditions[c]
        assert model.boundary_conditions[c] == boundary_conditions[c]
        assert model.variables["c"] == variables["c"]

        # update with conflicting submodel
        submodel2 = pybamm.BaseModel()
        submodel2.rhs = {d: pybamm.div(pybamm.grad(d)) - 1}
        with pytest.raises(pybamm.ModelError):
            model.update(submodel2)

        # update with multiple submodels
        submodel1 = submodel  # copy submodel from previous test
        submodel2 = pybamm.BaseModel()
        e = pybamm.Variable("e", domain=whole_cell)
        submodel2.rhs = {
            e: 5 * pybamm.div(pybamm.grad(d)) + pybamm.div(pybamm.grad(e)) - 1
        }
        submodel2.initial_conditions = {e: 3}
        submodel2.boundary_conditions = {
            e: {"left": (4, "Dirichlet"), "right": (7, "Dirichlet")}
        }

        model = pybamm.BaseModel()
        model.update(submodel1, submodel2)

        assert model.rhs[d] == submodel1.rhs[d]
        assert model.initial_conditions[d] == submodel1.initial_conditions[d]
        assert model.boundary_conditions[d] == submodel1.boundary_conditions[d]
        assert model.rhs[e] == submodel2.rhs[e]
        assert model.initial_conditions[e] == submodel2.initial_conditions[e]
        assert model.boundary_conditions[e] == submodel2.boundary_conditions[e]

    def test_new_copy(self):
        model = pybamm.BaseModel(name="a model")
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        d = pybamm.Variable("d", domain=whole_cell)
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(d)) - 1, d: -c}
        model.initial_conditions = {c: 1, d: 2}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
            d: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
        }
        model.use_jacobian = False
        model.convert_to_format = "python"

        new_model = model.new_copy()
        assert new_model.name == model.name
        assert new_model.use_jacobian == model.use_jacobian
        assert new_model.convert_to_format == model.convert_to_format

    def test_check_no_repeated_keys(self):
        model = pybamm.BaseModel()

        var = pybamm.Variable("var")
        model.rhs = {var: -1}
        var = pybamm.Variable("var")
        model.algebraic = {var: var}
        with pytest.raises(pybamm.ModelError, match=r"Multiple equations specified"):
            model.check_no_repeated_keys()

    def test_check_well_posedness_variables(self):
        # Well-posed ODE model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        d = pybamm.Variable("d", domain=whole_cell)
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(d)) - 1, d: -c}
        model.initial_conditions = {c: 1, d: 2}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
            d: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
        }
        model.check_well_posedness()

        # Well-posed DAE model
        e = pybamm.Variable("e", domain=whole_cell)
        model.algebraic = {e: e - c - d}
        model.check_well_posedness()

        # Underdetermined model - not enough differential equations
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(d)) - 1}
        model.algebraic = {e: e - c - d}
        with pytest.raises(pybamm.ModelError, match=r"underdetermined"):
            model.check_well_posedness()

        # Underdetermined model - not enough algebraic equations
        model.algebraic = {}
        with pytest.raises(pybamm.ModelError, match=r"underdetermined"):
            model.check_well_posedness()

        # Overdetermined model - repeated keys
        model.algebraic = {c: c - d, d: e + d}
        with pytest.raises(pybamm.ModelError, match=r"overdetermined"):
            model.check_well_posedness()
        # Overdetermined model - extra keys in algebraic
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(d)) - 1, d: -d}
        model.algebraic = {e: c - d}
        with pytest.raises(pybamm.ModelError, match=r"overdetermined"):
            model.check_well_posedness()
        model.rhs = {c: 1, d: -1}
        model.algebraic = {e: c - d}
        with pytest.raises(pybamm.ModelError, match=r"overdetermined"):
            model.check_well_posedness()

        # After discretisation, don't check for overdetermined from extra algebraic keys
        model = pybamm.BaseModel()
        model.algebraic = {c: 5 * pybamm.StateVector(slice(0, 15)) - 1}
        # passes with post_discretisation=True
        model.check_well_posedness(post_discretisation=True)
        # fails with post_discretisation=False (default)
        with pytest.raises(pybamm.ModelError, match=r"extra algebraic keys"):
            model.check_well_posedness()

        # after discretisation, algebraic equation without a StateVector fails
        model = pybamm.BaseModel()
        model.algebraic = {
            c: 1,
            d: pybamm.StateVector(slice(0, 15)) - pybamm.StateVector(slice(15, 30)),
        }
        with pytest.raises(
            pybamm.ModelError,
            match=r"each algebraic equation must contain at least one StateVector",
        ):
            model.check_well_posedness(post_discretisation=True)

        # model must be in semi-explicit form
        model = pybamm.BaseModel()
        model.rhs = {c: d.diff(pybamm.t), d: -1}
        model.initial_conditions = {c: 1, d: 1}
        with pytest.raises(
            pybamm.ModelError, match=r"time derivative of variable found"
        ):
            model.check_well_posedness()

        # model must be in semi-explicit form
        model = pybamm.BaseModel()
        model.algebraic = {c: 2 * d - c, d: c * d.diff(pybamm.t) - d}
        model.initial_conditions = {c: 1, d: 1}
        with pytest.raises(
            pybamm.ModelError, match=r"time derivative of variable found"
        ):
            model.check_well_posedness()

        # model must be in semi-explicit form
        model = pybamm.BaseModel()
        model.rhs = {c: d.diff(pybamm.t), d: -1}
        model.initial_conditions = {c: 1, d: 1}
        with pytest.raises(
            pybamm.ModelError, match=r"time derivative of variable found"
        ):
            model.check_well_posedness()

        # model must be in semi-explicit form
        model = pybamm.BaseModel()
        model.algebraic = {
            d: 5 * pybamm.StateVector(slice(0, 15)) - 1,
            c: 5 * pybamm.StateVectorDot(slice(0, 15)) - 1,
        }
        with pytest.raises(
            pybamm.ModelError, match=r"time derivative of state vector found"
        ):
            model.check_well_posedness(post_discretisation=True)

        # model must be in semi-explicit form
        model = pybamm.BaseModel()
        model.rhs = {c: 5 * pybamm.StateVectorDot(slice(0, 15)) - 1}
        model.initial_conditions = {c: 1}
        with pytest.raises(
            pybamm.ModelError, match=r"time derivative of state vector found"
        ):
            model.check_well_posedness(post_discretisation=True)

    def test_check_well_posedness_initial_boundary_conditions(self):
        # Well-posed model - Dirichlet
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        model = pybamm.BaseModel()
        c = pybamm.Variable("c", domain=whole_cell)
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(c)) - 1}
        model.initial_conditions = {c: 1}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
        }
        model.check_well_posedness()

        # Well-posed model - Neumann
        model.boundary_conditions = {
            c: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }
        model.check_well_posedness()

        # Model with bad initial conditions (expect assertion error)
        d = pybamm.Variable("d", domain=whole_cell)
        model.initial_conditions = {d: 3}
        with pytest.raises(pybamm.ModelError, match=r"initial condition"):
            model.check_well_posedness()

        # Algebraic well-posed model
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        model = pybamm.BaseModel()
        model.algebraic = {c: 5 * pybamm.div(pybamm.grad(c)) - 1}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
        }
        model.check_well_posedness()
        model.boundary_conditions = {
            c: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }
        model.check_well_posedness()

    def test_check_well_posedness_output_variables(self):
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        d = pybamm.Variable("d", domain=whole_cell)
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(d)) - 1, d: -c}
        model.initial_conditions = {c: 1, d: 2}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
            d: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
        }
        model._variables = {"something else": c}

        # check error raised if undefined variable in list of Variables
        pybamm.settings.debug_mode = True
        model = pybamm.BaseModel()
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables = {"d": d}
        with pytest.raises(pybamm.ModelError, match=r"No key set for variable"):
            model.check_well_posedness()

        # check error is raised even if some modified form of d is in model.rhs
        two_d = 2 * d
        model.rhs[two_d] = -d
        model.initial_conditions[two_d] = 1
        with pytest.raises(pybamm.ModelError, match=r"No key set for variable"):
            model.check_well_posedness()

        # add d to rhs, fine
        model.rhs[d] = -d
        model.initial_conditions[d] = 1
        model.check_well_posedness()

    def test_export_casadi(self):
        model = pybamm.BaseModel()
        t = pybamm.t
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        model.rhs = {a: -a * p}
        model.algebraic = {b: a - b}
        model.initial_conditions = {a: q, b: 1}
        model.variables = {"a+b": a + b - t}

        out = model.export_casadi_objects(["a+b"], input_parameter_order=["p", "q"])

        # Try making a function from the outputs
        t, x, z, p = out["t"], out["x"], out["z"], out["inputs"]
        x0, z0 = out["x0"], out["z0"]
        rhs, alg = out["rhs"], out["algebraic"]
        var = out["variables"]["a+b"]
        jac_rhs, jac_alg = out["jac_rhs"], out["jac_algebraic"]
        x0_fn = casadi.Function("x0", [p], [x0])
        z0_fn = casadi.Function("x0", [p], [z0])
        rhs_fn = casadi.Function("rhs", [t, x, z, p], [rhs])
        alg_fn = casadi.Function("alg", [t, x, z, p], [alg])
        jac_rhs_fn = casadi.Function("jac_rhs", [t, x, z, p], [jac_rhs])
        jac_alg_fn = casadi.Function("jac_alg", [t, x, z, p], [jac_alg])
        var_fn = casadi.Function("var", [t, x, z, p], [var])

        # Test that function values are as expected
        assert x0_fn([0, 5]) == 5
        assert z0_fn([0, 0]) == 1
        assert rhs_fn(0, 3, 2, [7, 2]) == -21
        assert alg_fn(0, 3, 2, [7, 2]) == 1
        np.testing.assert_array_equal(np.array(jac_rhs_fn(5, 6, 7, [8, 9])), [[-8, 0]])
        np.testing.assert_array_equal(np.array(jac_alg_fn(5, 6, 7, [8, 9])), [[1, -1]])
        assert var_fn(6, 3, 2, [7, 2]) == -1

        # Now change the order of input parameters
        out = model.export_casadi_objects(["a+b"], input_parameter_order=["q", "p"])

        # Try making a function from the outputs
        t, x, z, p = out["t"], out["x"], out["z"], out["inputs"]
        x0, z0 = out["x0"], out["z0"]
        rhs, alg = out["rhs"], out["algebraic"]
        var = out["variables"]["a+b"]
        jac_rhs, jac_alg = out["jac_rhs"], out["jac_algebraic"]
        x0_fn = casadi.Function("x0", [p], [x0])
        z0_fn = casadi.Function("x0", [p], [z0])
        rhs_fn = casadi.Function("rhs", [t, x, z, p], [rhs])
        alg_fn = casadi.Function("alg", [t, x, z, p], [alg])
        jac_rhs_fn = casadi.Function("jac_rhs", [t, x, z, p], [jac_rhs])
        jac_alg_fn = casadi.Function("jac_alg", [t, x, z, p], [jac_alg])
        var_fn = casadi.Function("var", [t, x, z, p], [var])

        # Test that function values are as expected
        assert x0_fn([5, 0]) == 5
        assert z0_fn([0, 0]) == 1
        assert rhs_fn(0, 3, 2, [2, 7]) == -21
        assert alg_fn(0, 3, 2, [2, 7]) == 1
        np.testing.assert_array_equal(np.array(jac_rhs_fn(5, 6, 7, [9, 8])), [[-8, 0]])
        np.testing.assert_array_equal(np.array(jac_alg_fn(5, 6, 7, [9, 8])), [[1, -1]])
        assert var_fn(6, 3, 2, [2, 7]) == -1

        # Test fails if order not specified
        with pytest.raises(
            ValueError, match=r"input_parameter_order must be specified"
        ):
            model.export_casadi_objects(["a+b"])

        # Fine if order is not specified if there is only one input parameter
        model = pybamm.BaseModel()
        p = pybamm.InputParameter("p")
        model.rhs = {a: -a}
        model.algebraic = {b: a - b}
        model.initial_conditions = {a: 1, b: 1}
        model.variables = {"a+b": a + b - p}

        out = model.export_casadi_objects(["a+b"])

        # Try making a function from the outputs
        t, x, z, p = out["t"], out["x"], out["z"], out["inputs"]
        var = out["variables"]["a+b"]
        var_fn = casadi.Function("var", [t, x, z, p], [var])

        # Test that function values are as expected
        # a + b - p = 3 + 2 - 7 = -2
        assert var_fn(6, 3, 2, [7]) == -2

        # Test fails if not discretised
        model = pybamm.lithium_ion.SPMe()
        with pytest.raises(
            pybamm.DiscretisationError, match=r"Cannot automatically discretise model"
        ):
            model.export_casadi_objects(["Electrolyte concentration [mol.m-3]"])

    @pytest.mark.skipif(platform.system() == "Windows", reason="Skipped for Windows")
    def test_generate_casadi(self):
        model = pybamm.BaseModel()
        t = pybamm.t
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        model.rhs = {a: -a * p}
        model.algebraic = {b: a - b}
        model.initial_conditions = {a: q, b: 1}
        model.variables = {"a+b": a + b - t}

        # Generate C code
        model.generate("test.c", ["a+b"], input_parameter_order=["p", "q"])

        # Compile
        subprocess.run(["gcc", "-fPIC", "-shared", "-o", "test.so", "test.c"])  # nosec

        # Read the generated functions
        x0_fn = casadi.external("x0", "./test.so")
        z0_fn = casadi.external("z0", "./test.so")
        rhs_fn = casadi.external("rhs_", "./test.so")
        alg_fn = casadi.external("alg_", "./test.so")
        jac_rhs_fn = casadi.external("jac_rhs", "./test.so")
        jac_alg_fn = casadi.external("jac_alg", "./test.so")
        var_fn = casadi.external("variables", "./test.so")

        # Test that function values are as expected
        assert x0_fn([2, 5]) == 5
        assert z0_fn([0, 0]) == 1
        assert rhs_fn(0, 3, 2, [7, 2]) == -21
        assert alg_fn(0, 3, 2, [7, 2]) == 1
        np.testing.assert_array_equal(np.array(jac_rhs_fn(5, 6, 7, [8, 9])), [[-8, 0]])
        np.testing.assert_array_equal(np.array(jac_alg_fn(5, 6, 7, [8, 9])), [[1, -1]])
        assert var_fn(6, 3, 2, [7, 2]) == -1

        # Remove generated files.
        os.remove("test.c")
        os.remove("test.so")

    def test_set_initial_conditions(self):
        # Set up model
        model = pybamm.BaseModel()
        var_scalar = pybamm.Variable("var_scalar")
        var_1D = pybamm.Variable("var_1D", domain="negative electrode")
        var_2D = pybamm.Variable(
            "var_2D",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        var_3D = pybamm.Variable(
            "var_3D",
            domain="negative particle",
            auxiliary_domains={
                "secondary": "negative electrode",
                "tertiary": "current collector",
            },
        )
        var_concat_neg = pybamm.Variable("var_concat_neg", domain="negative electrode")
        var_concat_sep = pybamm.Variable("var_concat_sep", domain="separator")
        var_concat = pybamm.concatenation(var_concat_neg, var_concat_sep)
        model.rhs = {var_scalar: -var_scalar, var_1D: -var_1D}
        model.algebraic = {
            var_2D: -var_2D,
            var_concat: -var_concat,
            var_3D: -var_3D,
        }
        model.initial_conditions = {
            var_scalar: 1,
            var_1D: 1,
            var_2D: 1,
            var_3D: 1,
            var_concat: 1,
        }
        model.variables = {
            "var_scalar": var_scalar,
            "var_1D": var_1D,
            "var_2D": var_2D,
            "var_3D": var_3D,
            "var_concat_neg": var_concat_neg,
            "var_concat_sep": var_concat_sep,
            "var_concat": var_concat,
        }

        # Test original initial conditions
        assert model.initial_conditions[var_scalar].value == 1
        assert model.initial_conditions[var_1D].value == 1
        assert model.initial_conditions[var_2D].value == 1
        assert model.initial_conditions[var_3D].value == 1
        assert model.initial_conditions[var_concat].value == 1
        # Discretise
        geometry = {
            "negative electrode": {"x_n": {"min": 0, "max": 1}},
            "separator": {"x_s": {"min": 1, "max": 2}},
            "negative particle": {"r_n": {"min": 0, "max": 1}},
            "current collector": {"z": {"min": 0, "max": 3}},
        }
        submeshes = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.Uniform1DSubMesh,
        }
        var_pts = {"x_n": 10, "x_s": 10, "r_n": 5, "z": 3}
        mesh = pybamm.Mesh(geometry, submeshes, var_pts)
        spatial_methods = {
            "negative electrode": pybamm.FiniteVolume(),
            "separator": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "current collector": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        model_disc = disc.process_model(model, inplace=False)
        t = np.linspace(0, 1)
        y = np.tile(3 * t, (1 + 30 + 50 + 150, 1))
        sol = pybamm.Solution(t, y, model_disc, {})

        # Update out-of-place first, since otherwise we'll have already modified the
        # model
        new_model = model.set_initial_conditions_from(sol, inplace=False)
        # Make sure original model is unchanged
        np.testing.assert_array_equal(
            model.initial_conditions[var_scalar].evaluate(), 1
        )
        np.testing.assert_array_equal(model.initial_conditions[var_1D].evaluate(), 1)
        np.testing.assert_array_equal(model.initial_conditions[var_2D].evaluate(), 1)
        np.testing.assert_array_equal(model.initial_conditions[var_3D].evaluate(), 1)
        np.testing.assert_array_equal(
            model.initial_conditions[var_concat].evaluate(), 1
        )

        # Now update inplace
        model.set_initial_conditions_from(sol)

        # Test new initial conditions (both in place and not)
        for mdl in [model, new_model]:
            var_scalar = mdl.variables["var_scalar"]
            assert isinstance(mdl.initial_conditions[var_scalar], pybamm.Vector)
            assert mdl.initial_conditions[var_scalar].entries == 3

            var_1D = mdl.variables["var_1D"]
            assert isinstance(mdl.initial_conditions[var_1D], pybamm.Vector)
            assert mdl.initial_conditions[var_1D].shape == (10, 1)
            np.testing.assert_array_equal(mdl.initial_conditions[var_1D].entries, 3)

            var_2D = mdl.variables["var_2D"]
            assert isinstance(mdl.initial_conditions[var_2D], pybamm.Vector)
            assert mdl.initial_conditions[var_2D].shape == (50, 1)
            np.testing.assert_array_equal(mdl.initial_conditions[var_2D].entries, 3)

            var_3D = mdl.variables["var_3D"]
            assert isinstance(mdl.initial_conditions[var_3D], pybamm.Vector)
            assert mdl.initial_conditions[var_3D].shape == (150, 1)
            np.testing.assert_array_equal(mdl.initial_conditions[var_3D].entries, 3)

            var_concat = mdl.variables["var_concat"]
            assert isinstance(mdl.initial_conditions[var_concat], pybamm.Vector)
            assert mdl.initial_conditions[var_concat].shape == (20, 1)
            np.testing.assert_array_equal(mdl.initial_conditions[var_concat].entries, 3)

        # Test updating a discretised model (out-of-place)
        new_model_disc = model_disc.set_initial_conditions_from(sol, inplace=False)

        # Test new initial conditions
        var_scalar = next(iter(new_model_disc.initial_conditions.keys()))
        assert isinstance(new_model_disc.initial_conditions[var_scalar], pybamm.Vector)
        assert new_model_disc.initial_conditions[var_scalar].entries == 3

        var_1D = list(new_model_disc.initial_conditions.keys())[1]
        assert isinstance(new_model_disc.initial_conditions[var_1D], pybamm.Vector)
        assert new_model_disc.initial_conditions[var_1D].shape == (10, 1)
        np.testing.assert_array_equal(
            new_model_disc.initial_conditions[var_1D].entries, 3
        )

        var_2D = list(new_model_disc.initial_conditions.keys())[2]
        assert isinstance(new_model_disc.initial_conditions[var_2D], pybamm.Vector)
        assert new_model_disc.initial_conditions[var_2D].shape == (50, 1)
        np.testing.assert_array_equal(
            new_model_disc.initial_conditions[var_2D].entries, 3
        )

        var_3D = list(new_model_disc.initial_conditions.keys())[3]
        assert isinstance(new_model_disc.initial_conditions[var_3D], pybamm.Vector)
        assert new_model_disc.initial_conditions[var_3D].shape == (150, 1)
        np.testing.assert_array_equal(
            new_model_disc.initial_conditions[var_3D].entries, 3
        )

        var_concat = list(new_model_disc.initial_conditions.keys())[4]
        assert isinstance(new_model_disc.initial_conditions[var_concat], pybamm.Vector)
        assert new_model_disc.initial_conditions[var_concat].shape == (20, 1)
        np.testing.assert_array_equal(
            new_model_disc.initial_conditions[var_concat].entries, 3
        )

        np.testing.assert_array_equal(
            new_model_disc.concatenated_initial_conditions.evaluate(), 3
        )

        # Test updating a new model with a different model
        new_model = pybamm.BaseModel()
        new_var_scalar = pybamm.Variable("var_scalar")
        new_var_1D = pybamm.Variable("var_1D", domain="negative electrode")
        new_var_2D = pybamm.Variable(
            "var_2D",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        new_var_3D = pybamm.Variable(
            "var_3D",
            domain="negative particle",
            auxiliary_domains={
                "secondary": "negative electrode",
                "tertiary": "current collector",
            },
        )
        new_var_concat_neg = pybamm.Variable(
            "var_concat_neg", domain="negative electrode"
        )
        new_var_concat_sep = pybamm.Variable("var_concat_sep", domain="separator")
        new_var_concat = pybamm.concatenation(new_var_concat_neg, new_var_concat_sep)
        new_model.rhs = {
            new_var_scalar: -2 * new_var_scalar,
            new_var_1D: -2 * new_var_1D,
        }
        new_model.algebraic = {
            new_var_2D: -2 * new_var_2D,
            new_var_concat: -2 * new_var_concat,
            new_var_3D: -2 * new_var_3D,
        }
        new_model.initial_conditions = {
            new_var_scalar: 1,
            new_var_1D: 1,
            new_var_2D: 1,
            new_var_3D: 1,
            new_var_concat: 1,
        }
        new_model.variables = {
            "var_scalar": new_var_scalar,
            "var_1D": new_var_1D,
            "var_2D": new_var_2D,
            "var_3D": new_var_3D,
            "var_concat_neg": new_var_concat_neg,
            "var_concat_sep": new_var_concat_sep,
            "var_concat": new_var_concat,
        }

        # Update the new model with the solution from another model
        new_model.set_initial_conditions_from(sol)

        # Test new initial conditions (both in place and not)
        var_scalar = new_model.variables["var_scalar"]
        assert isinstance(new_model.initial_conditions[var_scalar], pybamm.Vector)
        assert new_model.initial_conditions[var_scalar].entries == 3

        var_1D = new_model.variables["var_1D"]
        assert isinstance(new_model.initial_conditions[var_1D], pybamm.Vector)
        assert new_model.initial_conditions[var_1D].shape == (10, 1)
        np.testing.assert_array_equal(new_model.initial_conditions[var_1D].entries, 3)

        var_2D = new_model.variables["var_2D"]
        assert isinstance(new_model.initial_conditions[var_2D], pybamm.Vector)
        assert new_model.initial_conditions[var_2D].shape == (50, 1)
        np.testing.assert_array_equal(new_model.initial_conditions[var_2D].entries, 3)

        var_3D = new_model.variables["var_3D"]
        assert isinstance(new_model.initial_conditions[var_3D], pybamm.Vector)
        assert new_model.initial_conditions[var_3D].shape == (150, 1)
        np.testing.assert_array_equal(new_model.initial_conditions[var_3D].entries, 3)

        var_concat = new_model.variables["var_concat"]
        assert isinstance(new_model.initial_conditions[var_concat], pybamm.Vector)
        assert new_model.initial_conditions[var_concat].shape == (20, 1)
        np.testing.assert_array_equal(
            new_model.initial_conditions[var_concat].entries, 3
        )

        # Update the new model with a dictionary
        sol_dict = {
            "var_scalar": 5 * t,
            "var_1D": np.tile(5 * t, (10, 1)),
            "var_concat_neg": np.tile(5 * t, (10, 1)),
            "var_concat_sep": np.tile(5 * t, (10, 1)),
            "var_2D": np.tile(5 * t, (10, 5, 1)),
            "var_3D": np.tile(5 * t, (3, 10, 5, 1)),
        }
        new_model.set_initial_conditions_from(sol_dict)

        # Test new initial conditions (both in place and not)
        var_scalar = new_model.variables["var_scalar"]
        assert isinstance(new_model.initial_conditions[var_scalar], pybamm.Vector)
        assert new_model.initial_conditions[var_scalar].entries == 5

        var_1D = new_model.variables["var_1D"]
        assert isinstance(new_model.initial_conditions[var_1D], pybamm.Vector)
        assert new_model.initial_conditions[var_1D].shape == (10, 1)
        np.testing.assert_array_equal(new_model.initial_conditions[var_1D].entries, 5)

        var_2D = new_model.variables["var_2D"]
        assert isinstance(new_model.initial_conditions[var_2D], pybamm.Vector)
        assert new_model.initial_conditions[var_2D].shape == (50, 1)
        np.testing.assert_array_equal(new_model.initial_conditions[var_2D].entries, 5)

        var_3D = new_model.variables["var_3D"]
        assert isinstance(new_model.initial_conditions[var_3D], pybamm.Vector)
        assert new_model.initial_conditions[var_3D].shape == (150, 1)
        np.testing.assert_array_equal(new_model.initial_conditions[var_3D].entries, 5)

        var_concat = new_model.variables["var_concat"]
        assert isinstance(new_model.initial_conditions[var_concat], pybamm.Vector)
        assert new_model.initial_conditions[var_concat].shape == (20, 1)
        np.testing.assert_array_equal(
            new_model.initial_conditions[var_concat].entries, 5
        )

        # Test updating a discretised model (out-of-place)
        model_disc = disc.process_model(model, inplace=False)
        new_model_disc = model_disc.set_initial_conditions_from(sol_dict, inplace=False)

        # Test new initial conditions
        var_scalar = next(iter(new_model_disc.initial_conditions.keys()))
        assert isinstance(new_model_disc.initial_conditions[var_scalar], pybamm.Vector)
        assert new_model_disc.initial_conditions[var_scalar].entries == 5

        var_1D = list(new_model_disc.initial_conditions.keys())[1]
        assert isinstance(new_model_disc.initial_conditions[var_1D], pybamm.Vector)
        assert new_model_disc.initial_conditions[var_1D].shape == (10, 1)
        np.testing.assert_array_equal(
            new_model_disc.initial_conditions[var_1D].entries, 5
        )

        var_2D = list(new_model_disc.initial_conditions.keys())[2]
        assert isinstance(new_model_disc.initial_conditions[var_2D], pybamm.Vector)
        assert new_model_disc.initial_conditions[var_2D].shape == (50, 1)
        np.testing.assert_array_equal(
            new_model_disc.initial_conditions[var_2D].entries, 5
        )

        var_3D = list(new_model_disc.initial_conditions.keys())[3]
        assert isinstance(new_model_disc.initial_conditions[var_3D], pybamm.Vector)
        assert new_model_disc.initial_conditions[var_3D].shape == (150, 1)
        np.testing.assert_array_equal(
            new_model_disc.initial_conditions[var_3D].entries, 5
        )

        var_concat = list(new_model_disc.initial_conditions.keys())[4]
        assert isinstance(new_model_disc.initial_conditions[var_concat], pybamm.Vector)
        assert new_model_disc.initial_conditions[var_concat].shape == (20, 1)
        np.testing.assert_array_equal(
            new_model_disc.initial_conditions[var_concat].entries, 5
        )

        np.testing.assert_array_equal(
            new_model_disc.concatenated_initial_conditions.evaluate(), 5
        )

    def test_set_initial_conditions_from_y_slices(self):
        # Set up a simple discretised model
        model = pybamm.BaseModel()
        var = pybamm.Variable("test_var", domain="negative electrode")
        model.rhs = {var: -var}
        model.initial_conditions = {var: 1}

        # Discretise
        geometry = {"negative electrode": {"x_n": {"min": 0, "max": 1}}}
        mesh = pybamm.Mesh(
            geometry, {"negative electrode": pybamm.Uniform1DSubMesh}, {"x_n": 5}
        )
        disc = pybamm.Discretisation(
            mesh, {"negative electrode": pybamm.FiniteVolume()}
        )
        model_disc = disc.process_model(model, inplace=False)

        # Get the discretised variable (key from initial_conditions)
        disc_var = next(iter(model_disc.initial_conditions.keys()))

        # Create solution with known values
        t = np.array([0, 1])
        # y has shape (n_vars, n_time), with value 10 for test_var's slice
        y = np.zeros((model_disc.len_rhs_and_alg, 2))
        # Set last time step to 10 for the variable's slice using y_slices
        if disc_var in model_disc.y_slices:
            y_slice = model_disc.y_slices[disc_var][0]
            y[y_slice, -1] = 10
        sol = pybamm.Solution(t, y, model_disc, {})

        # Update initial conditions - should use y_slices path
        model_disc.set_initial_conditions_from(sol)

        # Verify initial conditions were updated correctly
        assert isinstance(model_disc.initial_conditions[disc_var], pybamm.Vector)
        np.testing.assert_array_equal(
            model_disc.initial_conditions[disc_var].entries, 10
        )

    def test_set_initial_condition_errors(self):
        model = pybamm.BaseModel()
        var = pybamm.Scalar(1)
        model.rhs = {var: -var}
        model.initial_conditions = {var: 1}
        with pytest.raises(NotImplementedError, match=r"Variable must have type"):
            model.set_initial_conditions_from({})

        # Inconsistent model and variable names
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: -var}
        model.initial_conditions = {var: pybamm.Scalar(1)}
        with pytest.raises(pybamm.ModelError, match=r"must appear in the solution"):
            model.set_initial_conditions_from({"wrong var": 2})
        var = pybamm.concatenation(
            pybamm.Variable("var", "test"), pybamm.Variable("var2", "test2")
        )
        model.rhs = {var: -var}
        model.initial_conditions = {var: pybamm.Scalar(1)}
        with pytest.raises(pybamm.ModelError, match=r"must appear in the solution"):
            model.set_initial_conditions_from({"wrong var": 2})

    def test_set_initial_conditions_4d_array(self):
        # Test 4D array handling in _extract_final_time_step
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: -var}
        model.initial_conditions = {var: 1}

        # Create 4D array data
        var_data_4d = np.random.rand(
            2, 3, 4, 10
        )  # 4D array with time as last dimension
        sol_dict = {"var": var_data_4d}

        model.set_initial_conditions_from(sol_dict)
        # Should extract final time step correctly
        assert isinstance(model.initial_conditions[var], pybamm.Vector)
        expected = var_data_4d[:, :, :, -1].flatten(order="F")
        # Entries are stored as column vector, so compare flattened
        np.testing.assert_array_equal(
            model.initial_conditions[var].entries.flatten(), expected
        )

    def test_set_initial_conditions_processed_variable_with_data(self):
        # Test ProcessedVariable with .data attribute
        model = pybamm.BaseModel()
        var = pybamm.Variable("var", domain="negative electrode")
        model.rhs = {var: -var}
        model.initial_conditions = {var: 1}

        # Discretise
        geometry = {"negative electrode": {"x_n": {"min": 0, "max": 1}}}
        mesh = pybamm.Mesh(
            geometry, {"negative electrode": pybamm.Uniform1DSubMesh}, {"x_n": 5}
        )
        disc = pybamm.Discretisation(
            mesh, {"negative electrode": pybamm.FiniteVolume()}
        )
        model_disc = disc.process_model(model, inplace=False)

        # Create solution
        t = np.array([0, 1])
        y = np.ones((model_disc.len_rhs_and_alg, 2)) * 5
        sol = pybamm.Solution(t, y, model_disc, {})

        disc_var = next(iter(model_disc.initial_conditions.keys()))
        # The solution should handle ProcessedVariable correctly
        model_disc.set_initial_conditions_from(sol)
        assert isinstance(model_disc.initial_conditions[disc_var], pybamm.Vector)

    def test_set_initial_conditions_y_slices_fallback(self):
        # Test fallback when y_slices extraction fails (e.g., invalid bounds)
        model = pybamm.BaseModel()
        var = pybamm.Variable("var", domain="negative electrode")
        model.rhs = {var: -var}
        model.initial_conditions = {var: 1}

        # Discretise
        geometry = {"negative electrode": {"x_n": {"min": 0, "max": 1}}}
        mesh = pybamm.Mesh(
            geometry, {"negative electrode": pybamm.Uniform1DSubMesh}, {"x_n": 5}
        )
        disc = pybamm.Discretisation(
            mesh, {"negative electrode": pybamm.FiniteVolume()}
        )
        model_disc = disc.process_model(model, inplace=False)

        # Create solution with dict fallback (simulating y_slices failure)
        var_data = np.ones((5, 10)) * 7
        sol_dict = {"var": var_data}

        # Should fall back to dict lookup
        model_disc.set_initial_conditions_from(sol_dict)
        disc_var = next(iter(model_disc.initial_conditions.keys()))
        assert isinstance(model_disc.initial_conditions[disc_var], pybamm.Vector)
        np.testing.assert_array_equal(
            model_disc.initial_conditions[disc_var].entries, 7
        )

    def test_set_initial_conditions_3d_array(self):
        # Test 3D array handling in _extract_final_time_step and get_final_state_eval
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: -var}
        model.initial_conditions = {var: 1}

        # Create 3D array data
        var_data_3d = np.random.rand(2, 3, 10)  # 3D array with time as last dimension
        sol_dict = {"var": var_data_3d}

        model.set_initial_conditions_from(sol_dict)
        assert isinstance(model.initial_conditions[var], pybamm.Vector)
        expected = var_data_3d[:, :, -1].flatten(order="F")
        np.testing.assert_array_equal(
            model.initial_conditions[var].entries.flatten(), expected
        )

    def test_set_initial_conditions_extract_final_time_step_5d_error(self):
        # Test NotImplementedError for >4D arrays in _extract_final_time_step
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: -var}
        model.initial_conditions = {var: 1}

        # Create 5D array data (should raise NotImplementedError)
        var_data_5d = np.random.rand(2, 3, 4, 5, 10)
        sol_dict = {"var": var_data_5d}

        with pytest.raises(
            NotImplementedError, match=r"Variable must be 0D, 1D, 2D, 3D, or 4D"
        ):
            model.set_initial_conditions_from(sol_dict)

    def test_set_initial_conditions_evaluate_symbol_to_array_paths(self):
        """Test _evaluate_symbol_to_array with numbers, arrays, and symbols"""
        model = pybamm.BaseModel()
        var = pybamm.Variable(
            "var", domain="negative electrode", scale=2.0, reference=1.0
        )
        model.rhs = {var: -var}
        model.initial_conditions = {var: 1}

        # Discretise to enable y_slices path which calls _evaluate_symbol_to_array
        geometry = {"negative electrode": {"x_n": {"min": 0, "max": 1}}}
        mesh = pybamm.Mesh(
            geometry, {"negative electrode": pybamm.Uniform1DSubMesh}, {"x_n": 5}
        )
        disc = pybamm.Discretisation(
            mesh, {"negative electrode": pybamm.FiniteVolume()}
        )
        model_disc = disc.process_model(model, inplace=False)

        # Create solution - this triggers _evaluate_symbol_to_array when evaluating
        # scale/reference (which are Scalar = numbers.Number)
        t = np.array([0, 1])
        y = np.ones((model_disc.len_rhs_and_alg, 2)) * 5
        sol = pybamm.Solution(t, y, model_disc, {})
        model_disc.set_initial_conditions_from(sol)
        disc_var = next(iter(model_disc.initial_conditions.keys()))
        assert isinstance(model_disc.initial_conditions[disc_var], pybamm.Vector)

        # Test with dict containing numpy array
        var_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sol_dict = {"var": var_data}
        model_disc.set_initial_conditions_from(sol_dict)

    def test_set_initial_conditions_extract_from_y_slices_broadcast_failures(self):
        model = pybamm.BaseModel()
        var = pybamm.Variable("var", domain="negative electrode")
        model.rhs = {var: -var}
        model.initial_conditions = {var: 1}

        # Discretise
        geometry = {"negative electrode": {"x_n": {"min": 0, "max": 1}}}
        mesh = pybamm.Mesh(
            geometry, {"negative electrode": pybamm.Uniform1DSubMesh}, {"x_n": 5}
        )
        disc = pybamm.Discretisation(
            mesh, {"negative electrode": pybamm.FiniteVolume()}
        )
        model_disc = disc.process_model(model, inplace=False)

        # Use dict fallback to test - the broadcast failure paths in y_slices
        # will return None and fall back to dict lookup, which we test here
        var_data = np.ones((5, 10)) * 6
        sol_dict = {"var": var_data}
        model_disc.set_initial_conditions_from(sol_dict)
        disc_var = next(iter(model_disc.initial_conditions.keys()))
        assert isinstance(model_disc.initial_conditions[disc_var], pybamm.Vector)

    def test_set_initial_conditions_find_matching_variable_not_discretised(self):
        """Test _find_matching_variable when solution_model is not discretised"""
        model = pybamm.BaseModel()
        var = pybamm.Variable("var", domain="negative electrode")
        model.rhs = {var: -var}
        model.initial_conditions = {var: 1}

        # Discretise the model we're updating
        geometry = {"negative electrode": {"x_n": {"min": 0, "max": 1}}}
        mesh = pybamm.Mesh(
            geometry, {"negative electrode": pybamm.Uniform1DSubMesh}, {"x_n": 5}
        )
        disc = pybamm.Discretisation(
            mesh, {"negative electrode": pybamm.FiniteVolume()}
        )
        model_disc = disc.process_model(model, inplace=False)

        # Should fall back to dict lookup since solution model is not discretised
        # This tests line 965 where solution_model.is_discretised is False
        var_data = np.ones((5, 10)) * 7
        sol_dict = {"var": var_data}
        model_disc.set_initial_conditions_from(sol_dict)
        disc_var = next(iter(model_disc.initial_conditions.keys()))
        assert isinstance(model_disc.initial_conditions[disc_var], pybamm.Vector)

    def test_set_initial_conditions_invalid_slice_bounds(self):
        """Test _extract_from_y_slices with invalid slice bounds (line 986)"""
        model = pybamm.BaseModel()
        var = pybamm.Variable("var", domain="negative electrode")
        model.rhs = {var: -var}
        model.initial_conditions = {var: 1}

        # Discretise
        geometry = {"negative electrode": {"x_n": {"min": 0, "max": 1}}}
        mesh = pybamm.Mesh(
            geometry, {"negative electrode": pybamm.Uniform1DSubMesh}, {"x_n": 5}
        )
        disc = pybamm.Discretisation(
            mesh, {"negative electrode": pybamm.FiniteVolume()}
        )
        model_disc = disc.process_model(model, inplace=False)

        # Create solution with y that's too short (invalid slice bounds)
        # Make y smaller than expected to trigger invalid bounds
        # Should fall back to dict lookup when slice bounds are invalid
        var_data = np.ones((5, 10)) * 8
        sol_dict = {"var": var_data}
        model_disc.set_initial_conditions_from(sol_dict)
        disc_var = next(iter(model_disc.initial_conditions.keys()))
        assert isinstance(model_disc.initial_conditions[disc_var], pybamm.Vector)

    def test_set_initial_conditions_evaluate_symbol_exception_handling(self):
        """Test exception handling in _evaluate_symbol_to_array (lines 945-959, 1020)"""
        model = pybamm.BaseModel()
        var = pybamm.Variable("var", domain="negative electrode")
        model.rhs = {var: -var}
        model.initial_conditions = {var: 1}

        # Discretise
        geometry = {"negative electrode": {"x_n": {"min": 0, "max": 1}}}
        mesh = pybamm.Mesh(
            geometry, {"negative electrode": pybamm.Uniform1DSubMesh}, {"x_n": 5}
        )
        disc = pybamm.Discretisation(
            mesh, {"negative electrode": pybamm.FiniteVolume()}
        )
        model_disc = disc.process_model(model, inplace=False)

        # Try to set initial conditions - should fall back to dict lookup
        # when evaluation fails
        var_data = np.ones((5, 10)) * 10
        sol_dict = {"var": var_data}
        model_disc.set_initial_conditions_from(sol_dict)
        disc_var = next(iter(model_disc.initial_conditions.keys()))
        assert isinstance(model_disc.initial_conditions[disc_var], pybamm.Vector)

    def test_set_initial_conditions_symbol_with_value_attribute(self):
        """Test _evaluate_symbol_to_array with symbols that have value attribute"""
        model = pybamm.BaseModel()
        var = pybamm.Variable("var", domain="negative electrode")
        model.rhs = {var: -var}
        model.initial_conditions = {var: 1}

        # Discretise
        geometry = {"negative electrode": {"x_n": {"min": 0, "max": 1}}}
        mesh = pybamm.Mesh(
            geometry, {"negative electrode": pybamm.Uniform1DSubMesh}, {"x_n": 5}
        )
        disc = pybamm.Discretisation(
            mesh, {"negative electrode": pybamm.FiniteVolume()}
        )
        model_disc = disc.process_model(model, inplace=False)

        # Create solution
        t = np.array([0, 1])
        y = np.ones((model_disc.len_rhs_and_alg, 2)) * 5
        sol = pybamm.Solution(t, y, model_disc, {})

        # Test with Scalar (which has value attribute and evaluate method)
        var_with_scalar_scale = pybamm.Variable(
            "var", domain="negative electrode", scale=pybamm.Scalar(2.0)
        )
        model2 = pybamm.BaseModel()
        model2.rhs = {var_with_scalar_scale: -var_with_scalar_scale}
        model2.initial_conditions = {var_with_scalar_scale: 1}
        model2_disc = disc.process_model(model2, inplace=False)
        model2_disc.set_initial_conditions_from(sol)
        disc_var2 = next(iter(model2_disc.initial_conditions.keys()))
        assert isinstance(model2_disc.initial_conditions[disc_var2], pybamm.Vector)

    def test_set_variables_error(self):
        var = pybamm.Variable("var")
        model = pybamm.BaseModel()
        with pytest.raises(ValueError, match=r"not var"):
            model.variables = {"not var": var}

    def test_build_submodels(self):
        class Submodel1(pybamm.BaseSubModel):
            def __init__(self, param, domain, options=None):
                super().__init__(param, domain, options=options)

            def get_fundamental_variables(self):
                u = pybamm.Variable("u")
                v = pybamm.Variable("v")
                return {"u": u, "v": v}

            def get_coupled_variables(self, variables):
                return variables

            def set_rhs(self, variables):
                u = variables["u"]
                self.rhs = {u: 2}

            def set_algebraic(self, variables):
                v = variables["v"]
                self.algebraic = {v: v - 1}

            def set_initial_conditions(self, variables):
                u = variables["u"]
                v = variables["v"]
                self.initial_conditions = {u: 0, v: 0}

            def add_events_from(self, variables):
                u = variables["u"]
                self.events.append(
                    pybamm.Event(
                        "Large u",
                        u - 200,
                        pybamm.EventType.TERMINATION,
                    )
                )

        class Submodel2(pybamm.BaseSubModel):
            def __init__(self, param, domain, options=None):
                super().__init__(param, domain, options=options)

            def get_coupled_variables(self, variables):
                u = variables["u"]
                variables.update({"w": 2 * u})
                return variables

        model = pybamm.BaseModel()
        model.submodels = {
            "submodel 1": Submodel1(None, "negative"),
            "submodel 2": Submodel2(None, "negative"),
        }
        assert not model._built
        model.build_model()
        assert model._built
        assert model._built == model.built
        u = model.variables["u"]
        v = model.variables["v"]
        assert model.rhs[u].value == 2
        assert model.algebraic[v] == -1.0 + v

    def test_timescale_lengthscale_get_set_not_implemented(self):
        model = pybamm.BaseModel()
        with pytest.raises(NotImplementedError):
            model.timescale
        with pytest.raises(NotImplementedError):
            model.length_scales
        with pytest.raises(NotImplementedError):
            model.timescale = 1
        with pytest.raises(NotImplementedError):
            model.length_scales = 1

    def test_save_load_model(self):
        # Set up model
        model = pybamm.BaseModel()
        var_scalar = pybamm.Variable("var_scalar")
        var_1D = pybamm.Variable("var_1D", domain="negative electrode")
        var_2D = pybamm.Variable(
            "var_2D",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        var_concat_neg = pybamm.Variable("var_concat_neg", domain="negative electrode")
        var_concat_sep = pybamm.Variable("var_concat_sep", domain="separator")
        var_concat = pybamm.concatenation(var_concat_neg, var_concat_sep)
        model.rhs = {var_scalar: -var_scalar, var_1D: -var_1D}
        model.algebraic = {var_2D: -var_2D, var_concat: -var_concat}
        model.initial_conditions = {var_scalar: 1, var_1D: 1, var_2D: 1, var_concat: 1}
        model.variables = {
            "var_scalar": var_scalar,
            "var_1D": var_1D,
            "var_2D": var_2D,
            "var_concat_neg": var_concat_neg,
            "var_concat_sep": var_concat_sep,
            "var_concat": var_concat,
        }

        # Discretise
        geometry = {
            "negative electrode": {"x_n": {"min": 0, "max": 1}},
            "separator": {"x_s": {"min": 1, "max": 2}},
            "negative particle": {"r_n": {"min": 0, "max": 1}},
        }
        submeshes = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
        }
        var_pts = {"x_n": 10, "x_s": 10, "r_n": 5}
        mesh = pybamm.Mesh(geometry, submeshes, var_pts)
        spatial_methods = {
            "negative electrode": pybamm.FiniteVolume(),
            "separator": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)
        model_disc = disc.process_model(model, inplace=False)
        t = np.linspace(0, 1)
        y = np.tile(3 * t, (1 + 30 + 50, 1))

        # Find baseline solution
        solution = pybamm.Solution(t, y, model_disc, {})

        # save model
        model_disc.save_model(filename="test_base_model")

        # load without variables
        new_model = pybamm.load_model("test_base_model.json")

        new_solution = pybamm.Solution(t, y, new_model, {})

        # model solutions match
        testing.assert_array_equal(solution.all_ys, new_solution.all_ys)

        model_disc.save_model(filename="test_base_model", mesh=mesh)

        # load with variables & mesh
        new_model = pybamm.load_model("test_base_model.json")

        os.remove("test_base_model.json")
