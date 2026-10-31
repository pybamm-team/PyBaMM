"""Tests of the transient closure fit and the isolated-sphere closure variable."""

import numpy as np
import pytest

import pybamm
from pybamm_model_zoo.dual_continuum import (
    fit_transient_closure,
    isolated_sphere_closure,
    parameter_sets,
)
from pybamm_model_zoo.dual_continuum.isolated_sphere import (
    FARADAY,
    isolated_sphere_modes,
    sphere_eigenvalues,
)
from pybamm_model_zoo.dual_continuum.transient_closure import short_time_limit

TIMES = np.geomspace(1e-3, 1e3, 300)


def exponential_sum(t, steady_value, weights, time_constants):
    return steady_value * ((1 - np.exp(-np.outer(t, 1 / time_constants))) @ weights)


class TestIsolatedSphereClosure:
    def test_eigenvalues(self):
        eigenvalues = sphere_eigenvalues()
        np.testing.assert_allclose(np.tan(eigenvalues[:200]), eigenvalues[:200])
        np.testing.assert_allclose(eigenvalues[0], 4.493409457909064)
        # Beyond the exact roots the asymptotic expansion is used
        x = eigenvalues[200:210]
        np.testing.assert_allclose(np.sin(x) - x * np.cos(x), 0, atol=1e-9)

    def test_limits(self):
        s = isolated_sphere_closure([0.0, 1e-9, 1e-7, 10.0])
        assert s[0] == 0
        # Short times: flat-wall limit; long times: -1/5
        np.testing.assert_allclose(
            s[1:3], -2 / np.sqrt(np.pi) * np.sqrt([1e-9, 1e-7]), rtol=1e-3
        )
        np.testing.assert_allclose(s[3], -0.2)

    def test_dimensional(self):
        radius, diffusivity = 5e-6, 1e-14
        t = np.array([1.0, 100.0, 1e5])
        np.testing.assert_allclose(
            isolated_sphere_closure(t, radius, diffusivity),
            isolated_sphere_closure(t * diffusivity / radius**2)
            * radius
            / (diffusivity * FARADAY),
        )
        np.testing.assert_allclose(
            isolated_sphere_closure(1e5, radius, diffusivity),
            -radius / (5 * diffusivity * FARADAY),
        )

    def test_errors(self):
        with pytest.raises(ValueError, match="both"):
            isolated_sphere_closure(1.0, radius=1e-6)
        with pytest.raises(ValueError, match="non-negative"):
            isolated_sphere_closure(-1.0)

    @pytest.mark.parametrize("n_modes", [1, 4, 8, 12])
    def test_built_in_modes(self, n_modes):
        weights, time_constants = isolated_sphere_modes(n_modes)
        assert len(weights) == len(time_constants) == n_modes
        np.testing.assert_allclose(sum(weights), 1)
        if n_modes >= 8:
            t = np.geomspace(1e-6, 1, 200)
            fitted = exponential_sum(
                t, -0.2, np.array(weights), np.array(time_constants)
            )
            assert np.abs(fitted - isolated_sphere_closure(t)).max() < 0.005 * 0.2


class TestFitTransientClosure:
    def test_recovers_known_modes(self):
        time_constants = np.geomspace(0.1, 100, 4)
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        s = exponential_sum(TIMES, -400.0, weights, time_constants)
        fit = fit_transient_closure(
            TIMES, s, steady_value=-400.0, n_modes=4, time_constant_range=(0.1, 100)
        )
        np.testing.assert_allclose(fit.weights, weights, atol=1e-6)
        np.testing.assert_allclose(fit.time_constants, time_constants)
        assert fit.steady_value == pytest.approx(-400.0, rel=1e-6)
        assert fit.n_modes == 4
        assert fit.max_error < 1e-3
        assert fit.warnings == ()

    def test_starts_at_zero_and_reaches_steady_value(self):
        fit = fit_transient_closure(TIMES, isolated_sphere_closure(TIMES))
        assert fit.evaluate(0.0)[0] == 0
        np.testing.assert_allclose(fit.evaluate(1e6), fit.steady_value)
        np.testing.assert_allclose(fit.weights.sum(), 1)
        assert np.all(fit.weights > 0)

    def test_automatic_number_of_modes(self):
        s = isolated_sphere_closure(TIMES)
        loose = fit_transient_closure(TIMES, s, tolerance=0.02)
        tight = fit_transient_closure(TIMES, s, tolerance=0.002)
        assert loose.max_error <= 0.02 * 0.2
        assert tight.max_error <= 0.002 * 0.2
        assert loose.n_modes < tight.n_modes

    def test_callable_and_steady_value(self):
        fit = fit_transient_closure(TIMES, isolated_sphere_closure, steady_value=-0.2)
        assert fit.steady_value == -0.2
        np.testing.assert_allclose(
            fit.evaluate(TIMES), isolated_sphere_closure(TIMES), atol=0.005 * 0.2
        )

    def test_parameter_values(self):
        fit = fit_transient_closure(TIMES, isolated_sphere_closure(TIMES), n_modes=3)
        values = fit.parameter_values("Negative")
        assert values["Negative electrode s0 surface average"] == fit.steady_value
        assert values["Negative electrode transient closure weight 1"] == fit.weights[0]
        assert (
            values["Negative electrode transient closure time constant 1 [s]"]
            == fit.time_constants[0]
        )
        dimensionless = fit_transient_closure(
            TIMES, isolated_sphere_closure(TIMES), n_modes=3, dimensionless=True
        ).parameter_values()
        assert "Positive electrode s0 surface average dimensionless" in dimensionless
        assert (
            "Positive electrode transient closure time constant 1 dimensionless"
            in dimensionless
        )
        with pytest.raises(pybamm.DomainError, match="Positive"):
            fit.parameter_values("separator")
        # Electrodes with several particle phases use PyBaMM's phase prefixes
        secondary = fit.parameter_values("Negative", phase="secondary")
        assert secondary["Secondary: Negative electrode s0 surface average"] == (
            fit.steady_value
        )
        assert (
            "Secondary: Negative electrode transient closure time constant 1 [s]"
            in secondary
        )
        with pytest.raises(pybamm.OptionError, match="phase"):
            fit.parameter_values("Negative", phase="tertiary")

    def test_report(self):
        text = str(
            fit_transient_closure(TIMES, isolated_sphere_closure(TIMES), n_modes=3)
        )
        assert text.startswith("Transient closure fit: ")
        assert "time constant [s]" in text

    def test_warns_about_unresolved_early_times(self):
        # The 3D closure data reaches -52 by 1.2 ms, ~25 times the flat-wall limit
        closure = parameter_sets.load_3d_case_data()["closure"]
        t, s = closure["Time [s]"], closure["Closure variable [mol.m-1.A-1]"]
        fit = fit_transient_closure(t, s, diffusivity=4e-14)
        assert any("short-time limit" in w for w in fit.warnings)
        # The implied cell size is in metres, or in units of L when dimensionless
        assert any("cells of size ~3.83e-07 m " in w for w in fit.warnings)
        length = 75.62e-6
        fit = fit_transient_closure(
            t * 4e-14 / length**2, s * 4e-14 * FARADAY / length, dimensionless=True
        )
        assert any("~0.00506 (in units of L)" in w for w in fit.warnings)
        # Without the diffusivity, the share of s0 reached at once is flagged
        fit = fit_transient_closure(t, s)
        assert any("first data point" in w for w in fit.warnings)

    def test_fix_early_times(self):
        closure = parameter_sets.load_3d_case_data()["closure"]
        t, s = closure["Time [s]"], closure["Closure variable [mol.m-1.A-1]"]
        fit = fit_transient_closure(t, s, diffusivity=4e-14, fix_early_times=True)
        fix = fit.early_time_fix
        # The offset h / (2 D F) implies cells of ~0.4 um, resolved after 10 h^2 / D
        assert fix["cell_size"] == pytest.approx(3.8e-7, rel=0.05)
        assert fix["resolved_after"] == pytest.approx(
            10 * fix["cell_size"] ** 2 / 4e-14
        )
        early = np.geomspace(1e-2, fix["short_time_until"], 5)
        np.testing.assert_allclose(
            fit.evaluate(early), short_time_limit(early, -1, 4e-14), rtol=0.1
        )
        late = t >= fix["resolved_after"]
        np.testing.assert_allclose(fit.evaluate(t[late]), s[late], atol=0.01 * 417)
        with pytest.raises(ValueError, match="diffusivity"):
            fit_transient_closure(t, s, fix_early_times=True)

    def test_warns_about_overshoot_and_unmet_tolerance(self):
        s = isolated_sphere_closure(TIMES) * (
            1 + 0.05 * np.exp(-((np.log(TIMES) - 0.0) ** 2))
        )
        fit = fit_transient_closure(TIMES, s, steady_value=-0.2, max_modes=3)
        assert any("overshoots" in w for w in fit.warnings)
        assert any("tolerance" in w for w in fit.warnings)

    def test_errors(self):
        with pytest.raises(ValueError, match="same length"):
            fit_transient_closure([1, 2, 3], [1, 2])
        with pytest.raises(ValueError, match="three"):
            fit_transient_closure([0, 1, 2], [0, -1, -2])
        with pytest.raises(ValueError, match="non-zero"):
            fit_transient_closure(TIMES, -np.ones_like(TIMES), steady_value=0)
        with pytest.raises(ValueError, match="time_constant_range"):
            fit_transient_closure(
                TIMES, isolated_sphere_closure(TIMES), time_constant_range=(1, 0.1)
            )

    def test_plot(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        closure = parameter_sets.load_3d_case_data()["closure"]
        fit = fit_transient_closure(
            closure["Time [s]"],
            closure["Closure variable [mol.m-1.A-1]"],
            diffusivity=4e-14,
            fix_early_times=True,
        )
        figure = fit.plot(show_plot=False)
        assert len(figure.axes) == 2
        matplotlib.pyplot.close(figure)
