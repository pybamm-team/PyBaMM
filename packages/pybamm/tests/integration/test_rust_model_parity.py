"""CasADi parity of compiled expressions from full battery models."""

import functools

import casadi
import numpy as np
import pytest

import pybamm
from pybamm.rust import ExprGraph
from tests.conftest_rust import jacobian_entries

_INPUTS = {
    "Current function [A]": 5.0,
    "Negative particle diffusivity [m2.s-1]": 3.3e-14,
}


def _drive_cycle():
    parameter_values = pybamm.ParameterValues("Chen2020")
    times = np.linspace(0.0, 3600.0, 50)
    # Discharge, then charge, so the last state meets |I| with I negative
    parameter_values["Current function [A]"] = pybamm.Interpolant(
        times, 1.5 * np.cos(np.pi * times / 3000.0), pybamm.t
    )
    return parameter_values


def _with_inputs():
    parameter_values = pybamm.ParameterValues("Chen2020")
    parameter_values.update(dict.fromkeys(_INPUTS, "[input]"))
    return parameter_values


def _marcus_hush_chidsey():
    parameter_values = pybamm.ParameterValues("Chen2020")
    parameter_values.update(
        {
            f"{electrode} electrode reorganization energy [eV]": 0.35
            for electrode in ("Negative", "Positive")
        },
        check_already_exists=False,
    )
    return parameter_values


# Model class, options, parameter values and inputs; each adds submodels or nodes
_MODELS = {
    "SPM": (pybamm.lithium_ion.SPM, {}, None, {}),
    "SPMe": (pybamm.lithium_ion.SPMe, {}, None, {}),
    "DFN": (pybamm.lithium_ion.DFN, {}, None, {}),
    "DFN x-full thermal": (pybamm.lithium_ion.DFN, {"thermal": "x-full"}, None, {}),
    "DFN SEI and plating": (
        pybamm.lithium_ion.DFN,
        {"SEI": "solvent-diffusion limited", "lithium plating": "irreversible"},
        lambda: pybamm.ParameterValues("OKane2022"),
        {},
    ),
    "SPM 1+1D": (
        pybamm.lithium_ion.SPM,
        {"current collector": "potential pair", "dimensionality": 1},
        None,
        {},
    ),
    "MPM": (pybamm.lithium_ion.MPM, {}, None, {}),
    "lead-acid Full": (pybamm.lead_acid.Full, {}, None, {}),
    "SPM drive cycle": (pybamm.lithium_ion.SPM, {}, _drive_cycle, {}),
    "SPM with inputs": (pybamm.lithium_ion.SPM, {}, _with_inputs, _INPUTS),
    "SPM Marcus-Hush-Chidsey": (
        pybamm.lithium_ion.SPM,
        {"intercalation kinetics": "Marcus-Hush-Chidsey"},
        _marcus_hush_chidsey,
        {},
    ),
    "Thevenin": (pybamm.equivalent_circuit.Thevenin, {}, None, {}),
}


def _casadi_functions(exprs, n, input_names):
    """CasADi functions of ``(t, y, p)`` for the values of ``exprs`` and, when
    there is one expression, its jacobians with respect to ``y`` and ``p``."""
    t = casadi.MX.sym("t")
    y = casadi.MX.sym("y", n)
    inputs = {name: casadi.MX.sym(name) for name in input_names}
    p = casadi.vertcat(*inputs.values()) if inputs else casadi.MX.sym("p", 0)
    values = [expr.to_casadi(t, y, inputs=inputs) for expr in exprs]
    functions = {"value": casadi.Function("value", [t, y, p], values)}
    if len(values) == 1:
        for name, wrt in (("jac_y", y), ("jac_p", p)):
            jacobian = casadi.jacobian(values[0], wrt)
            functions[name] = casadi.Function(name, [t, y, p], [jacobian])
    return functions


def _assert_close(actual, expected, scale, message):
    """Assert each entry agrees to 1e-9 of the larger of its value and ``scale``,
    and each non-finite entry exactly."""
    # An infinite derivative makes the scale non-finite, and bounds nothing
    scale = np.where(np.isfinite(scale), scale, 0.0)
    tolerance = 1e-12 + 1e-9 * np.maximum(np.abs(expected), scale)
    with np.errstate(invalid="ignore"):
        within = (np.abs(actual - expected) <= tolerance) & np.isfinite(expected)
    close = within | (actual == expected) | (np.isnan(actual) & np.isnan(expected))
    entries = np.argwhere(~close)[:5]
    assert not entries.size, (
        f"{message}: entries {entries.tolist()} give {actual[~close][:5]}, CasADi "
        f"{expected[~close][:5]}, within {tolerance[~close][:5]}"
    )


@pytest.fixture(scope="module", params=list(_MODELS))
def case(request):
    """A built model, its inputs, and states along its discharge with their times."""
    model_class, options, parameter_values, inputs = _MODELS[request.param]
    model = model_class(options)
    # The reference states come from CasADi, whatever the default format
    model.convert_to_format = "casadi"
    sim = pybamm.Simulation(
        model,
        parameter_values=None if parameter_values is None else parameter_values(),
    )
    solution = sim.solve([0, 3600], inputs=inputs)
    # Start, middle and end: states that start at zero move, and late branches open
    indices = (0, len(solution.t) // 2, -1)
    points = [(solution.t[i], np.asarray(solution.y[:, i]).ravel()) for i in indices]
    return sim.built_model, inputs, points


class TestModelParity:
    def test_equations_match_casadi(self, case):
        model, inputs, points = case
        # The solver evaluates each equation against the full state vector
        n = model.concatenated_initial_conditions.size
        equations = {
            "rhs": model.concatenated_rhs,
            "algebraic": model.concatenated_algebraic,
        }
        for name, expr in equations.items():
            if expr.size == 0:
                continue
            graph = ExprGraph()
            compiled = graph.compile(expr.to_rust(graph, {}), name=name, n_states=n)
            names = compiled.input_names
            reference = _casadi_functions([expr], n, names)
            p = {key: inputs[key] for key in names}
            p_stacked = [inputs[key] for key in names]
            jacobians = {"y": compiled.jacobian("y")}
            # Rust takes df/dp as dense, since each input is a sweep of its own
            if names:
                jacobians["p"] = compiled.jacobian("p")
            pattern = reference["jac_y"].sparsity_out(0)
            extra = jacobian_entries(*jacobians["y"].sparsity()) - jacobian_entries(
                pattern.colind(), pattern.row()
            )
            assert not extra, f"{name} df/dy entries CasADi's pattern lacks: {extra}"
            for t, y in points:
                expected = {
                    wrt: np.asarray(reference[f"jac_{wrt}"](t, y, p_stacked))
                    for wrt in ("y", "p")
                }
                # A residual cancels terms far larger than itself; scale by those
                terms = np.abs(expected["y"]) @ np.abs(y)
                terms += np.abs(expected["p"]) @ np.abs(p_stacked)
                _assert_close(
                    compiled(t, y, p),
                    np.asarray(reference["value"](t, y, p_stacked)).ravel(),
                    terms,
                    f"{name} value at t={t}",
                )
                for wrt, jacobian in jacobians.items():
                    x = np.abs(y if wrt == "y" else p_stacked)
                    # An entry's error times its variable gets the residual's
                    # tolerance; a zero variable bounds nothing, so the flat one stays
                    scale = np.divide(
                        terms[:, None],
                        x,
                        out=np.zeros(expected[wrt].shape),
                        where=x > 0,
                    )
                    _assert_close(
                        jacobian(t, y, p).toarray(),
                        expected[wrt],
                        scale,
                        f"{name} d/d{wrt} at t={t}",
                    )

    def test_variables_and_events_match_casadi(self, case):
        model, inputs, points = case
        n = model.concatenated_initial_conditions.size
        outputs = {name: model.get_processed_variable(name) for name in model.variables}
        outputs.update(
            {f"Event: {event.name}": event.expression for event in model.events}
        )
        graph, symbols = ExprGraph(), {}
        compiled = graph.compile_outputs(
            {name: expr.to_rust(graph, symbols) for name, expr in outputs.items()},
            n_states=n,
        )
        names = compiled.input_names
        reference = _casadi_functions(
            [outputs[name] for name in compiled.output_names], n, names
        )["value"]
        p = {key: inputs[key] for key in names}
        p_stacked = [inputs[key] for key in names]

        @functools.cache
        def output_jacobians(name):
            return _casadi_functions([outputs[name]], n, names)

        for t, y in points:
            values = compiled.split(compiled(t, y, p))
            expected = reference.call([t, y, p_stacked])
            for name, value, reference_value in zip(
                compiled.output_names, values, expected, strict=True
            ):
                actual = np.ravel(value)
                desired = np.asarray(reference_value).ravel()
                # A variable can cancel terms far larger than itself. They only widen the
                # flat rule, and every output's jacobian is slow, so find them where it fails
                terms = 0.0
                if not np.allclose(
                    actual, desired, rtol=1e-9, atol=1e-12, equal_nan=True
                ):
                    jacobians = output_jacobians(name)
                    terms = sum(
                        abs(jacobians[f"jac_{wrt}"](t, y, p_stacked).sparse())
                        @ np.abs(x)
                        for wrt, x in (("y", y), ("p", p_stacked))
                    )
                _assert_close(actual, desired, terms, f"{name} at t={t}")


class TestSPMParity:
    @pytest.fixture(scope="class")
    def spm(self):
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sim.build()
        return sim.built_model

    def test_named_outputs_trajectory(self, spm):
        rust_symbols = {}
        g = ExprGraph()
        names = ["Voltage [V]", "Current [A]"]
        # Raw variables_and_events hold nodes that can't be lowered.
        exprs = {name: spm.get_processed_variable_or_event(name) for name in names}
        # y0_list only exists after a solver set_up runs.
        y0 = np.asarray(
            spm.concatenated_initial_conditions.evaluate(), dtype=np.float64
        ).ravel()
        # Outputs reference the algebraic voltage state.
        n = y0.shape[0]
        fn = g.compile_outputs(
            {name: e.to_rust(g, rust_symbols) for name, e in exprs.items()},
            n_states=n,
        )
        n_t = 25
        ts = np.linspace(0.0, 3600.0, n_t)
        Y = np.tile(y0[:, None], (1, n_t)) * np.linspace(1.0, 1.05, n_t)
        p = np.array([])
        results = fn.split(fn.eval_trajectory(ts, Y, p))
        for name, out in zip(fn.output_names, results, strict=True):
            cf = _casadi_functions([exprs[name]], n, ())["value"]
            ref = np.column_stack(
                [np.asarray(cf(tj, Y[:, j], [])).ravel() for j, tj in enumerate(ts)]
            )
            np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-12)
