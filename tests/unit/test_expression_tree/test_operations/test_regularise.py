#
# Tests for the RegulariseSqrtAndPower class
#

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

        # The original c_e**0.5 Power node should be replaced
        # (result will have different Power nodes from reg_power formula)
        assert result != expr

    def test_scale_default_to_one(self):
        """Test that unrecognized expressions get scale=None."""
        c_e = pybamm.Variable("c_e")
        c_s = pybamm.Variable("c_s")

        # Only c_e has a scale, c_s should get scale=None
        inputs = {"c_e": c_e, "c_s": c_s}
        regulariser = pybamm.RegulariseSqrtAndPower(
            {c_e: pybamm.Scalar(1000.0)},
            inputs=inputs,
        )

        expr = pybamm.sqrt(c_s)
        result = regulariser(expr, inputs=inputs)

        # Should be replaced with RegPower (no Sqrt)
        has_sqrt = any(isinstance(n, pybamm.Sqrt) for n in result.pre_order())
        assert not has_sqrt
        # Check it's a RegPower with scale=1 (default)
        assert isinstance(result, pybamm.RegPower)
        # Scale is the third child
        assert result.children[2] == pybamm.Scalar(1)

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

        # sqrt(c_s / c_s_max) should also be replaced
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
