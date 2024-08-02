#
# Tests the settings class.
#

import pybamm
import pytest


class TestSettings:
    def test_simplify(self):
        with pytest.raises(TypeError):
            pybamm.settings.simplify = "Not Bool"

        assert pybamm.settings.simplify

        pybamm.settings.simplify = False
        assert not pybamm.settings.simplify

        pybamm.settings.simplify = True

    def test_debug_mode(self):
        with pytest.raises(TypeError):
            pybamm.settings.debug_mode = "Not bool"

    def test_smoothing_parameters(self):
        assert pybamm.settings.min_max_mode == "exact"
        assert pybamm.settings.heaviside_smoothing == "exact"
        assert pybamm.settings.abs_smoothing == "exact"

        pybamm.settings.set_smoothing_parameters(10)
        assert pybamm.settings.min_max_smoothing == 10
        assert pybamm.settings.heaviside_smoothing == 10
        assert pybamm.settings.abs_smoothing == 10
        pybamm.settings.set_smoothing_parameters("exact")

        # Test errors
        with pytest.raises(ValueError, match="greater than 1"):
            pybamm.settings.min_max_mode = "smooth"
            pybamm.settings.min_max_smoothing = 0.9
        with pytest.raises(ValueError, match="positive number"):
            pybamm.settings.min_max_mode = "soft"
            pybamm.settings.min_max_smoothing = -10
        with pytest.raises(ValueError, match="positive number"):
            pybamm.settings.heaviside_smoothing = -10
        with pytest.raises(ValueError, match="positive number"):
            pybamm.settings.abs_smoothing = -10
        with pytest.raises(ValueError, match="'soft', or 'smooth'"):
            pybamm.settings.min_max_mode = "unknown"
        pybamm.settings.set_smoothing_parameters("exact")
