"""Regression tests for #4930.

When ``Initial concentration in <Domain> electrode [mol.m-3]`` is set
to a Python callable ``f(r, x)``, the corresponding ``c_init``
:class:`pybamm.FunctionParameter` evaluates the user's function on
:class:`pybamm.SpatialVariable` leaves (``r_n``, ``x_n``). This bubbles
into ``c_init_av = xyz_average(r_average(c_init))`` and then into the
:attr:`pybamm.Variable.reference` of every electrolyte / electrode
potential Variable (whose reference is ``-U_init`` and so depends on
``c_init_av``).

Before the fix, :meth:`pybamm.Discretisation._process_symbol` and
:meth:`pybamm.Discretisation.process_equation` substituted
``Variable.reference`` (and ``Variable.scale``) *without* discretising
them first, so the un-folded ``r_n`` / ``x_n`` leaves leaked into the
discretised rhs and initial-condition vectors. The resulting model
then crashed in ``test_shape`` / ``check_initial_conditions`` with::

    NotImplementedError: method self.evaluate() not implemented for
    symbol r_n of type <class 'SpatialVariable'>

These tests pin the fix: a parameter-derived ``Variable.reference`` /
``Variable.scale`` that still holds undiscretised symbolic averages
must be discretised by the discretisation layer, so that
:class:`pybamm.lithium_ion.DFN` (and adjacent reduced-order models)
solve end-to-end with an ``r``/``x``-dependent initial concentration.

The same regression happens with ``pybamm.Simulation`` both with and
without ``pybamm.Experiment``, so both paths are covered."""

from __future__ import annotations

import numpy as np
import pytest

import pybamm


def _f_rx(r, x):
    """User callable for the initial concentration. The bug is
    triggered as soon as the result genuinely depends on the
    :class:`SpatialVariable` arguments — a constant
    ``17038.0 + 0 * r + 0 * x`` would simplify away."""
    return 17038.0 + r + x


def _patched_parameter_values():
    pv = pybamm.ParameterValues("Chen2020")
    pv.update({"Initial concentration in negative electrode [mol.m-3]": _f_rx})
    return pv


@pytest.fixture
def parameter_values():
    return _patched_parameter_values()


def test_dfn_solves_with_function_of_r_and_x_for_initial_concentration(
    parameter_values,
):
    """The exact reproducer from #4930: a plain
    :class:`pybamm.Simulation` (no experiment) on ``DFN`` with a
    callable ``Initial concentration in negative electrode [mol.m-3]``
    used to raise ``NotImplementedError`` for ``r_n``."""
    model = pybamm.lithium_ion.DFN()
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sol = sim.solve([0, 1])
    voltage = sol["Voltage [V]"].entries
    assert sol.t[-1] == pytest.approx(1.0)
    # Voltage must be finite and in the cell's usable range — guards
    # against the IC having been silently zeroed by a future
    # regression.
    assert np.all(np.isfinite(voltage))
    assert 2.0 < float(voltage[0]) < 5.0


def test_dfn_with_experiment_solves_with_function_of_r_and_x(parameter_values):
    """Original failing path from #4930's reproducer: ``DFN`` wrapped
    in ``pybamm.Experiment(["Rest for 1 sec"])``. This is the path
    that surfaces the discretisation-time
    ``test_shape`` / ``check_initial_conditions`` failures, because
    the experiment build pipes through ``_discretise_experiment_models``."""
    model = pybamm.lithium_ion.DFN()
    experiment = pybamm.Experiment(["Rest for 1 sec"])
    sim = pybamm.Simulation(
        model, experiment=experiment, parameter_values=parameter_values
    )
    sol = sim.solve()
    voltage = sol["Voltage [V]"].entries
    assert np.all(np.isfinite(voltage))
    assert 2.0 < float(voltage[0]) < 5.0


def test_spm_and_spme_solve_with_function_of_r_and_x(parameter_values):
    """Belt-and-braces: the discretisation-time fix is on the generic
    ``Variable`` / ``VariableDot`` / ``ConcatenationVariable`` branches,
    not on DFN-specific code, so reduced-order models that historically
    *did* work with this kind of callable must keep working. SPM
    doesn't carry the electrolyte potential Variable that triggered
    the original crash; SPMe does."""
    for model_cls in (pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe):
        model = model_cls()
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        sol = sim.solve([0, 1])
        voltage = sol["Voltage [V]"].entries
        assert np.all(np.isfinite(voltage))
        assert 2.0 < float(voltage[0]) < 5.0


def test_discretised_rhs_has_no_spatial_variable_leaves(parameter_values):
    """Direct guard on the discretisation invariant the fix restores:
    after discretisation, no ``rhs`` / ``algebraic`` / ``initial_conditions``
    equation may carry an un-folded :class:`SpatialVariable` leaf.
    This is what eventually trips ``test_shape`` (line 880) and
    ``check_initial_conditions`` (line 1292) at runtime."""
    model = pybamm.lithium_ion.DFN()
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    # Force the full build without solving, so the assertion is
    # against the discretised model itself rather than via a solve
    # error message.
    sim.build()
    built = sim.built_model

    def _spatial_variable_leaves(expr):
        leaves = expr.post_order(filter=lambda node: len(node.children) == 0)
        return [leaf for leaf in leaves if isinstance(leaf, pybamm.SpatialVariable)]

    for label, eqn_dict in (
        ("rhs", built.rhs),
        ("algebraic", built.algebraic),
        ("initial_conditions", built.initial_conditions),
    ):
        for key, eqn in eqn_dict.items():
            sps = _spatial_variable_leaves(eqn)
            assert not sps, (
                f"{label}[{key!r}] still carries undiscretised "
                f"SpatialVariable leaves {[s.name for s in sps]}"
            )


def test_positive_electrode_callable_initial_concentration(parameter_values):
    """The fix is symmetric in the negative / positive electrode
    pathway because both Variables route through the same
    ``Variable.reference`` substitution. Pin the positive-electrode
    side as well so a future refactor that only fixes
    ``self.param.n`` keeps the ``self.param.p`` side honest."""

    def f_pos(r, x):
        return 33133.0 + r + x

    parameter_values.update(
        {"Initial concentration in positive electrode [mol.m-3]": f_pos}
    )
    model = pybamm.lithium_ion.DFN()
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sol = sim.solve([0, 1])
    voltage = sol["Voltage [V]"].entries
    assert np.all(np.isfinite(voltage))
    assert 2.0 < float(voltage[0]) < 5.0
