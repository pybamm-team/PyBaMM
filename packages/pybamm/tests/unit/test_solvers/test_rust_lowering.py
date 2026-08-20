"""Invariants every Rust-backed solver's model lowering must share.

Each solver composes its own lowering, so these pin the parts that must agree
between them: output variables slice by flattened component count, graph input
indices follow the inputs dict, the state residual spans every state, and one
set_up lowers into one graph.
"""

import numpy as np
import pytest

import pybamm
from pybamm.solvers.rust_lowering import RustModelLowering

_VECTOR_VAR = "Negative particle surface concentration [mol.m-3]"
_OUTPUTS = ["Voltage [V]", _VECTOR_VAR, "Current [A]"]


def _built_model(with_input=False, options=None):
    """Build an SPM pinned to the Rust format, not the ambient default."""
    model = pybamm.lithium_ion.SPM(options)
    model.convert_to_format = "rust"
    params = model.default_parameter_values
    if with_input:
        params["Current function [A]"] = "[input]"
    sim = pybamm.Simulation(model, parameter_values=params)
    sim.build()
    built = sim.built_model
    built.convert_to_format = "rust"
    return built


def _termination_events(model):
    return [
        event
        for event in model.events
        if event.event_type == pybamm.EventType.TERMINATION
    ]


class TestOutputLengths:
    def test_lengths_are_the_flattened_component_counts(self):
        model = _built_model()
        lowering = RustModelLowering(model, {})
        lowering.state_residual()

        _, lengths = lowering.outputs(_OUTPUTS)

        assert lengths[0] == 1
        assert lengths[1] > 1, "a spatial variable must contribute every component"
        assert lengths[2] == 1

    def test_tensor_field_output_is_rejected(self):
        """A stacked field would be mis-sliced by its tensor shape, so refuse it."""
        model = _built_model()
        lowering = RustModelLowering(model, {})
        lowering.state_residual()
        component = model.get_processed_variable_or_event(_VECTOR_VAR)
        model.variables_and_events["field"] = pybamm.VectorField(component, component)

        with pytest.raises(pybamm.SolverError, match=r"'field' is a tensor field"):
            lowering.outputs(["field"])

    def test_idaklu_and_diffsol_agree_on_output_lengths(self):
        t_eval = np.linspace(0, 100, 5)

        idaklu = pybamm.IDAKLUSolver(output_variables=_OUTPUTS)
        idaklu.solve(_built_model(), t_eval)

        diffsol = pybamm.DiffsolSolver(output_variables=_OUTPUTS)
        diffsol.solve(_built_model(), t_eval)

        assert idaklu._setup["output_assembly"].lens == diffsol._output_assembly.lens


class TestInputRegistration:
    def test_graph_input_order_follows_the_inputs_dict(self):
        model = _built_model(with_input=True)
        inputs = {"Current function [A]": 0.5}

        lowering = RustModelLowering(model, inputs)

        assert lowering.input_name_order == list(inputs)

    def test_sensitivity_indices_point_into_the_input_order(self):
        model = _built_model(with_input=True)
        inputs = {"Current function [A]": 0.5}
        model.calculate_sensitivities = ["Current function [A]"]

        lowering = RustModelLowering(model, inputs)
        indices, names = lowering.sensitivity_indices(model.calculate_sensitivities)

        assert indices == [0]
        assert names == ["Current function [A]"]

    def test_sensitivity_names_not_supplied_as_inputs_are_dropped(self):
        model = _built_model(with_input=True)
        inputs = {"Current function [A]": 0.5}

        lowering = RustModelLowering(model, inputs)
        indices, names = lowering.sensitivity_indices(["Not an input"])

        assert indices == []
        assert names == []


class TestStateResidual:
    def test_dae_residual_spans_every_state(self):
        model = _built_model()
        lowering = RustModelLowering(model, {})
        lowering.state_residual()

        compiled = lowering.compile()

        assert compiled.n_states == model.len_rhs_and_alg

    def test_compile_before_state_residual_is_rejected(self):
        lowering = RustModelLowering(_built_model(), {})

        with pytest.raises(pybamm.SolverError, match=r"state residual"):
            lowering.compile()


class TestGenericEvaluators:
    """The slots BaseSolver's backend-agnostic helpers read on the native path."""

    def test_rhs_evaluator_spans_the_differential_block_only(self):
        """A DAE's residual is wider than ydot0, so the rhs needs its own root."""
        model = _built_model(options={"surface form": "algebraic"})
        assert model.len_alg > 0, "expected a DAE"
        lowering = RustModelLowering(model, {})
        lowering.state_residual()

        rhs = lowering.rhs_evaluator()
        y0 = np.asarray(model.concatenated_initial_conditions.evaluate()).reshape(-1)
        values = np.asarray(rhs(0.0, y0, np.array([]))).reshape(-1)

        assert values.shape[0] == model.len_rhs

    def test_rhs_evaluator_matches_the_residual_it_shares_nodes_with(self):
        model = _built_model(options={"surface form": "algebraic"})
        lowering = RustModelLowering(model, {})
        lowering.state_residual()
        rhs = lowering.rhs_evaluator()
        compiled = lowering.compile()

        y0 = np.asarray(model.concatenated_initial_conditions.evaluate()).reshape(-1)
        residual = np.asarray(compiled.rhs(0.0, y0, np.array([]))).reshape(-1)

        np.testing.assert_array_equal(
            np.asarray(rhs(0.0, y0, np.array([]))).reshape(-1),
            residual[: model.len_rhs],
        )

    def test_bound_events_are_views_not_copies(self):
        """Views onto the roots the fused root-finding tape is built from."""
        model = _built_model()
        lowering = RustModelLowering(model, {})
        lowering.state_residual()
        lowering.termination_events()
        compiled = lowering.compile()

        lowering.bind_generic_evaluators(compiled)

        assert [
            evaluator._cf for evaluator in model.terminate_events_eval
        ] == compiled.events

    def test_bound_events_follow_model_events_order(self):
        model = _built_model()
        lowering = RustModelLowering(model, {})
        lowering.state_residual()
        lowering.termination_events()
        lowering.bind_generic_evaluators(lowering.compile())

        events = _termination_events(model)
        assert len(model.terminate_events_eval) == len(events)
        y0 = np.asarray(model.concatenated_initial_conditions.evaluate()).reshape(-1)
        for event, evaluator in zip(events, model.terminate_events_eval, strict=True):
            np.testing.assert_allclose(
                float(np.asarray(evaluator(0.0, y0, np.array([]))).ravel()[0]),
                float(np.asarray(event.expression.evaluate(0.0, y0, inputs={})).item()),
                rtol=1e-12,
            )

    @pytest.mark.parametrize(
        "solver_class", [pybamm.IDAKLUSolver, pybamm.DiffsolSolver]
    )
    def test_every_native_solver_binds_both_slots(self, solver_class):
        """An unbound events slot would silently skip the event-violation check."""
        model = _built_model()
        events = _termination_events(model)
        assert events, "expected events the check could silently skip"

        solver_class().set_up(model, inputs={}, t_eval=np.linspace(0, 100, 3))

        assert model.rhs_eval is not None
        assert len(model.terminate_events_eval) == len(events)


class TestOneGraphPerSetUp:
    """One lowering seam per ``set_up``: every native evaluator off one graph."""

    @staticmethod
    def _count_graphs(monkeypatch, solver, model):
        from pybamm.solvers import rust_lowering

        built = []
        original = rust_lowering.rust_graph_with_inputs

        def counting(*args, **kwargs):
            built.append(None)
            return original(*args, **kwargs)

        monkeypatch.setattr(rust_lowering, "rust_graph_with_inputs", counting)
        solver.set_up(model, inputs={}, t_eval=np.linspace(0, 100, 3))
        return len(built)

    @pytest.mark.parametrize(
        "solver_class", [pybamm.IDAKLUSolver, pybamm.DiffsolSolver]
    )
    def test_native_set_up_builds_one_graph_beside_the_initial_conditions(
        self, monkeypatch, solver_class
    ):
        """Two graphs: the initial conditions, and the solver's shared lowering.

        Anything more means an expression the shared graph already holds was
        lowered a second time.
        """
        model = _built_model()
        assert _termination_events(model), "expected events that could be re-lowered"

        assert self._count_graphs(monkeypatch, solver_class(), model) == 2

    def test_casadi_path_still_lowers_per_group(self, monkeypatch):
        """The skip is native-only; a CasADi model builds no Rust graph at all."""
        model = pybamm.lithium_ion.SPM()
        model.convert_to_format = "casadi"
        sim = pybamm.Simulation(model)
        sim.build()
        built = sim.built_model
        built.convert_to_format = "casadi"

        count = self._count_graphs(monkeypatch, pybamm.IDAKLUSolver(), built)

        assert count == 0
        assert built.rhs_eval is not None
        assert len(built.terminate_events_eval) == len(_termination_events(built))
