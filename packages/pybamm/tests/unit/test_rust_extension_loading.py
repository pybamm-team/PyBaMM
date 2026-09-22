"""The ``pybamm.rust`` facade re-exports the compiled extension."""

import pybamm.rust


class TestFacadeExports:
    def test_core_is_importable_through_the_facade(self):
        assert pybamm.rust.CompiledModel is not None
        assert pybamm.rust.ExprGraph is not None
