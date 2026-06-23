"""Tests for the RegulariseSqrtAndPower class."""

import pytest

import pybamm


class TestRegulariseSqrtAndPower:
    def test_basic_sqrt_replacement(self):
        """Test that Sqrt nodes are replaced with regularised expressions."""
        c_e = pybamm.Variable("c_e")
        c_s = pybamm.Variable("c_s")

        inputs = {"c_e": c_e, "c_s": c_s}
        regulariser = pybamm.RegulariseSqrtAndPower(
            {
                c_e: pybamm.Scalar(1000.0),
                c_s: pybamm.Scalar(50000.0),
            },
            inputs=inputs,
        )

        expr = pybamm.sqrt(c_e) + pybamm.sqrt(c_s)
        result = regulariser(expr, inputs=inputs)

        # Check that result has no Sqrt nodes
        has_sqrt = any(isinstance(n, pybamm.Sqrt) for n in result.pre_order())
        assert not has_sqrt

    def test_basic_power_replacement(self):
        """Test that Power nodes are replaced with regularised expressions."""
        c_e = pybamm.Variable("c_e")

        inputs = {"c_e": c_e}
        regulariser = pybamm.RegulariseSqrtAndPower(
            {c_e: pybamm.Scalar(1000.0)},
            inputs=inputs,
        )

        expr = c_e**0.5
        result = regulariser(expr, inputs=inputs)

        # Matched base: RegPower with the registered scale
        assert isinstance(result, pybamm.RegPower)
        assert result.children[2] == pybamm.Scalar(1000.0)

    def test_state_dependent_base_default_scale(self):
        """State-dependent base with no registered scale: regularised with
        the default scale of 1."""
        c_e = pybamm.Variable("c_e")
        c_s = pybamm.Variable("c_s")

        inputs = {"c_e": c_e, "c_s": c_s}
        regulariser = pybamm.RegulariseSqrtAndPower(
            {c_e: pybamm.Scalar(1000.0)},
            inputs=inputs,
        )

        expr = pybamm.sqrt(c_s)
        result = regulariser(expr, inputs=inputs)

        assert isinstance(result, pybamm.RegPower)
        assert result.children[2] == pybamm.Scalar(1)

    def test_normalised_state_dependent_base_still_regularised(self):
        """Normalised base (ORegan2022-style (c_e / c_e_ref) ** (1 - alpha))
        matches no scale but is still regularised with the default scale."""
        c_e = pybamm.Variable("c_e")
        inputs = {"c_e": c_e}
        regulariser = pybamm.RegulariseSqrtAndPower(
            {c_e: pybamm.Scalar(1000.0)},
            inputs=inputs,
        )

        c_e_ref = pybamm.Parameter("Reference concentration")
        expr = (c_e / c_e_ref) ** 0.208
        result = regulariser(expr, inputs=inputs)

        assert isinstance(result, pybamm.RegPower)
        assert result.children[2] == pybamm.Scalar(1)

    def test_parameter_power_not_corrupted(self):
        """State-independent Parameter base is not regularised: RegPower with
        the default scale would corrupt a small rate constant."""
        c_e = pybamm.Variable("c_e")
        inputs = {"c_e": c_e}
        regulariser = pybamm.RegulariseSqrtAndPower(
            {c_e: pybamm.Scalar(1000.0)},
            inputs=inputs,
        )

        rate_constant = pybamm.Parameter("Rate constant")
        expr = rate_constant**0.5
        result = regulariser(expr, inputs=inputs)

        assert isinstance(result, pybamm.Power)
        assert not any(isinstance(n, pybamm.RegPower) for n in result.pre_order())

        parameter_values = pybamm.ParameterValues({"Rate constant": 5e-9})
        value = parameter_values.process_symbol(result).evaluate()
        assert value == pytest.approx(5e-9**0.5, rel=1e-12)

    def test_input_parameter_power_not_corrupted(self):
        """State-independent InputParameter base is not regularised."""
        c_e = pybamm.Variable("c_e")
        inputs = {"c_e": c_e}
        regulariser = pybamm.RegulariseSqrtAndPower(
            {c_e: pybamm.Scalar(1000.0)},
            inputs=inputs,
        )

        rate_constant = pybamm.InputParameter("Rate constant")
        expr = rate_constant**0.5
        result = regulariser(expr, inputs=inputs)

        assert isinstance(result, pybamm.Power)
        value = result.evaluate(inputs={"Rate constant": 5e-9})
        assert value == pytest.approx(5e-9**0.5, rel=1e-12)

    def test_constant_base_state_dependent_exponent(self):
        """Constant base, state-dependent exponent: outer Power kept, sqrt in
        the exponent regularised."""
        c_e = pybamm.Variable("c_e")
        inputs = {"c_e": c_e}
        regulariser = pybamm.RegulariseSqrtAndPower(
            {c_e: pybamm.Scalar(1000.0)},
            inputs=inputs,
        )

        base = pybamm.Parameter("Rate constant")
        expr = base ** pybamm.sqrt(c_e)
        result = regulariser(expr, inputs=inputs)

        assert isinstance(result, pybamm.Power)
        assert isinstance(result.children[0], pybamm.Parameter)
        assert any(isinstance(n, pybamm.RegPower) for n in result.pre_order())
        assert not any(isinstance(n, pybamm.Sqrt) for n in result.pre_order())

    def test_existing_reg_power_preserved(self):
        """A manual RegPower node (e.g. MSMR j0_j) is a Function, not a
        Sqrt/Power, so it is left untouched, even when nested."""
        c_e = pybamm.Variable("c_e")
        c_e_ref = pybamm.Parameter("c_e_ref")
        aj = pybamm.Parameter("aj")
        inputs = {"c_e": c_e}
        regulariser = pybamm.RegulariseSqrtAndPower(
            {c_e: pybamm.Scalar(1000.0)},
            inputs=inputs,
        )

        expr = pybamm.reg_power(c_e / c_e_ref, 1 - aj)
        result = regulariser(expr, inputs=inputs)
        assert result is expr

        # Nested: reg_power preserved, sibling Power regularised
        nested = pybamm.reg_power(c_e / c_e_ref, 1 - aj) * c_e**0.5
        result = regulariser(nested, inputs=inputs)
        n_reg_power = sum(
            1 for n in result.pre_order() if isinstance(n, pybamm.RegPower)
        )
        assert n_reg_power == 2
        assert not any(isinstance(n, pybamm.Power) for n in result.pre_order())
        assert not any(isinstance(n, pybamm.Sqrt) for n in result.pre_order())

    @pytest.mark.parametrize("set_name", ["Chen2020", "ORegan2022", "Ecker2015"])
    def test_j0_pipeline_produces_regpower(self, set_name):
        """Built-in j0 pipeline regularises all concentration powers,
        including ORegan2022's unmatched normalised bases."""
        param = pybamm.LithiumIonParameters()
        c_e = pybamm.Variable("c_e")
        c_s_surf = pybamm.Variable("c_s_surf")
        T = pybamm.Variable("T")
        j0 = param.n.prim.j0(c_e, c_s_surf, T)

        parameter_values = pybamm.ParameterValues(set_name)
        processed = parameter_values.process_symbol(j0)

        n_reg_power = sum(
            1 for n in processed.pre_order() if isinstance(n, pybamm.RegPower)
        )
        n_plain_power = sum(
            1 for n in processed.pre_order() if isinstance(n, pybamm.Power)
        )
        assert n_reg_power == 3, f"{set_name}: expected 3 RegPower, got {n_reg_power}"
        assert n_plain_power == 0, (
            f"{set_name}: expected no plain Power nodes, got {n_plain_power}"
        )

    def test_function_parameter_post_processor_small_parameter(self):
        """Full j0-style path: a FunctionParameter raising a Parameter to a
        fractional power must evaluate exactly."""

        def exchange_current(c_e):
            rate_constant = pybamm.Parameter("Rate constant")
            return rate_constant**0.5 * c_e**0.5

        c_e = pybamm.Variable("c_e")
        inputs = {"Electrolyte concentration [mol.m-3]": c_e}
        regulariser = pybamm.RegulariseSqrtAndPower(
            {c_e: pybamm.Scalar(1000.0)},
            inputs=inputs,
        )
        function_parameter = pybamm.FunctionParameter(
            "Exchange-current density [A.m-2]",
            {"Electrolyte concentration [mol.m-3]": pybamm.Scalar(1000.0)},
            post_processor=regulariser,
        )
        parameter_values = pybamm.ParameterValues(
            {
                "Exchange-current density [A.m-2]": exchange_current,
                "Rate constant": 5e-9,
            }
        )
        value = parameter_values.process_symbol(function_parameter).evaluate()
        expected = 5e-9**0.5 * 1000.0**0.5
        assert value == pytest.approx(expected, rel=1e-6)

    def test_exact_match_only(self):
        """Test that only exact matches are used for scales."""
        c_s = pybamm.Variable("c_s")
        c_s_max = pybamm.Parameter("c_s_max")

        # c_s has scale, but c_s / c_s_max should not inherit it
        inputs = {"c_s": c_s, "c_s_max": c_s_max}
        regulariser = pybamm.RegulariseSqrtAndPower(
            {c_s: c_s_max},
            inputs=inputs,
        )

        # sqrt(c_s) should be replaced
        expr1 = pybamm.sqrt(c_s)
        result1 = regulariser(expr1, inputs=inputs)
        has_sqrt = any(isinstance(n, pybamm.Sqrt) for n in result1.pre_order())
        assert not has_sqrt

        # sqrt(c_s / c_s_max) should also be replaced (state-dependent base)
        expr2 = pybamm.sqrt(c_s / c_s_max)
        result2 = regulariser(expr2, inputs=inputs)
        has_sqrt = any(isinstance(n, pybamm.Sqrt) for n in result2.pre_order())
        assert not has_sqrt

    def test_division_does_not_inherit_scale(self):
        """Test that c_s / c_max does NOT incorrectly inherit scale from c_s.

        This is a critical test to ensure we don't apply scale to expressions
        that merely contain a scaled variable. Only exact matches or explicitly
        registered patterns should get a scale.
        """
        c_s = pybamm.Variable("c_s")
        c_s_max = pybamm.Parameter("c_s_max")

        inputs = {"c_s": c_s, "c_s_max": c_s_max}
        regulariser = pybamm.RegulariseSqrtAndPower(
            {c_s: c_s_max},  # Only c_s has scale, NOT c_s / c_s_max
            inputs=inputs,
        )

        # sqrt(c_s) should get scale=c_s_max
        expr1 = pybamm.sqrt(c_s)
        result1 = regulariser(expr1, inputs=inputs)
        assert isinstance(result1, pybamm.RegPower)
        # Scale is the third child
        assert result1.children[2] == c_s_max

        # sqrt(c_s / c_s_max) should get scale=1 (NOT c_s_max!)
        # because c_s / c_s_max is a different expression that wasn't registered
        expr2 = pybamm.sqrt(c_s / c_s_max)
        result2 = regulariser(expr2, inputs=inputs)
        assert isinstance(result2, pybamm.RegPower)
        # CRITICAL: must be Scalar(1), not c_s_max
        assert result2.children[2] == pybamm.Scalar(1)

    def test_explicit_pattern_matching(self):
        """Test that explicit patterns like c_max - c_s can be added."""
        c_s = pybamm.Variable("c_s")
        c_s_max = pybamm.Parameter("c_s_max")

        inputs = {"c_s": c_s, "c_s_max": c_s_max}
        regulariser = pybamm.RegulariseSqrtAndPower(
            {
                c_s: c_s_max,
                c_s_max - c_s: c_s_max,  # explicit pattern
            },
            inputs=inputs,
        )

        expr = pybamm.sqrt(c_s_max - c_s)
        result = regulariser(expr, inputs=inputs)

        has_sqrt = any(isinstance(n, pybamm.Sqrt) for n in result.pre_order())
        assert not has_sqrt

    def test_nested_expression(self):
        """Test that nested expressions are handled correctly."""
        c_e = pybamm.Variable("c_e")
        c_s = pybamm.Variable("c_s")

        inputs = {"c_e": c_e, "c_s": c_s}
        regulariser = pybamm.RegulariseSqrtAndPower(
            {
                c_e: pybamm.Scalar(1000.0),
                c_s: pybamm.Scalar(50000.0),
            },
            inputs=inputs,
        )

        # Nested expression with multiple sqrt
        expr = pybamm.sqrt(c_e) * pybamm.sqrt(c_s)

        result = regulariser(expr, inputs=inputs)

        # Should have no Sqrt nodes
        has_sqrt = any(isinstance(n, pybamm.Sqrt) for n in result.pre_order())
        assert not has_sqrt

    def test_processed_inputs(self):
        """Test that the regulariser works with processed (different) inputs."""
        # Original symbols (before processing)
        c_e_orig = pybamm.Variable("c_e")
        c_s_orig = pybamm.Variable("c_s")
        c_s_max_orig = pybamm.Parameter("c_s_max")

        original_inputs = {
            "c_e": c_e_orig,
            "c_s": c_s_orig,
            "c_s_max": c_s_max_orig,
        }

        regulariser = pybamm.RegulariseSqrtAndPower(
            {
                c_e_orig: pybamm.Scalar(1000.0),
                c_s_orig: c_s_max_orig,
                c_s_max_orig - c_s_orig: c_s_max_orig,
            },
            inputs=original_inputs,
        )

        # Simulated "processed" symbols (what ParameterSubstitutor would produce)
        c_e_proc = pybamm.StateVector(slice(0, 10), name="c_e")
        c_s_proc = pybamm.StateVector(slice(10, 20), name="c_s")
        c_s_max_proc = pybamm.Scalar(51765.0, name="c_s_max")

        processed_inputs = {
            "c_e": c_e_proc,
            "c_s": c_s_proc,
            "c_s_max": c_s_max_proc,
        }

        # Expression built from processed symbols (as returned by user's function)
        expr = (
            pybamm.sqrt(c_e_proc)
            * pybamm.sqrt(c_s_proc)
            * pybamm.sqrt(c_s_max_proc - c_s_proc)
        )
        result = regulariser(expr, inputs=processed_inputs)

        # Check that all sqrts were replaced
        has_sqrt = any(isinstance(n, pybamm.Sqrt) for n in result.pre_order())
        assert not has_sqrt
