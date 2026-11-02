//! Observables of a compiled model: output variables and events, and their derivatives.

use std::sync::Arc;

use crate::arena::{Arena, NodeId};
use crate::error::CoreError;
use crate::eval::CompiledExpr;
use crate::observable::{ObservableKind, ObservableScratch, ObservableSet, seed_param_tangent};
use crate::params::{ParamRegistry, resolve_input_width};

use super::{CompiledModel, Workspace, verify_state_extent};

impl Workspace {
    /// The `kind` observables' buffers and the parameter-tangent buffer, borrowed together.
    #[inline]
    fn observable_tangent(&mut self, kind: ObservableKind) -> (&mut ObservableScratch, &mut [f64]) {
        let scratch = match kind {
            ObservableKind::Outputs => &mut self.output_scratch,
            ObservableKind::Events => &mut self.event_scratch,
        };
        (scratch, &mut self.sens_dp_buffer)
    }

    #[inline]
    fn observable_scratch(&mut self, kind: ObservableKind) -> &mut ObservableScratch {
        self.observable_tangent(kind).0
    }
}

impl CompiledModel {
    /// Compile `roots` as this model's observables of `kind`, replacing any set before.
    ///
    /// Every root is checked against the declared states and inputs before any compiles. Two or
    /// more also fuse into one tape, so shared subexpressions run once; that allocates a `Concat`
    /// in `arena`.
    pub fn set_observables(
        &mut self,
        kind: ObservableKind,
        arena: &mut Arena,
        roots: &[NodeId],
    ) -> Result<(), CoreError> {
        for &root in roots {
            verify_state_extent(arena, root, self.n_states)?;
        }
        // Whole-arena, as for the rhs; a parameter registered since the build widens `dp`.
        let registry = ParamRegistry::from_arena(arena)?;
        resolve_input_width(Some(self.input_width), registry.packed_width())?;
        if let Some((index, ..)) = self
            .params
            .registered()
            .find(|&(index, _, slot)| registry.slot(index) != Some(slot))
        {
            return Err(CoreError::ObservableParamSlot {
                index: index.index(),
            });
        }
        let set = ObservableSet::compile(arena, roots, self.has_sensitivities());
        self.param_extent = self.param_extent.max(registry.index_extent());
        match kind {
            ObservableKind::Outputs => self.outputs = set,
            ObservableKind::Events => self.events = set,
        }
        Ok(())
    }

    /// The observables of `kind`: their tapes, layout and scratch sizing.
    #[inline]
    pub(crate) const fn observables(&self, kind: ObservableKind) -> &ObservableSet {
        match kind {
            ObservableKind::Outputs => &self.outputs,
            ObservableKind::Events => &self.events,
        }
    }

    /// Each observable of `kind`'s compiled tape, in the order they were set.
    pub fn observable_exprs(
        &self,
        kind: ObservableKind,
    ) -> impl ExactSizeIterator<Item = &Arc<CompiledExpr>> {
        self.observables(kind).exprs()
    }

    #[inline]
    pub const fn n_outputs(&self) -> usize {
        self.outputs.count()
    }

    #[inline]
    pub(crate) fn output_len_at(&self, var_idx: usize) -> usize {
        self.outputs.len_at(var_idx)
    }

    /// Length of all output variables concatenated.
    #[inline]
    pub const fn total_output_len(&self) -> usize {
        self.outputs.total_len()
    }

    #[inline]
    pub const fn n_events(&self) -> usize {
        self.events.count()
    }

    /// Length of all events concatenated.
    #[inline]
    pub const fn total_event_len(&self) -> usize {
        self.events.total_len()
    }

    /// Evaluate every observable of `kind` into `output`, concatenated.
    pub(crate) fn eval_observables(
        &self,
        ws: &mut Workspace,
        kind: ObservableKind,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        output: &mut [f64],
    ) {
        self.observables(kind)
            .eval_all(ws.observable_scratch(kind), t, y, inputs, output);
    }

    /// Batch-evaluate every output over `k` points: `y_cols` `(n_states, k)`, `out`
    /// `(total_output_len, k)`, both F-contiguous.
    pub(crate) fn eval_outputs_batch(
        &self,
        ws: &mut Workspace,
        k: usize,
        ts: &[f64],
        y_cols: &[f64],
        inputs: &[f64],
        out: &mut [f64],
    ) {
        self.outputs.eval_batch(
            ws.observable_scratch(ObservableKind::Outputs),
            k,
            ts,
            y_cols,
            self.n_states,
            inputs,
            out,
        );
    }

    /// Compute `dH/dp · v` over every observable of `kind`, where `v[i]` weighs sensitivity
    /// column `i`.
    ///
    /// Panics on a model compiled without sensitivities, whose observables carry no tangents.
    #[allow(clippy::too_many_arguments)]
    pub fn observable_sens_action(
        &self,
        ws: &mut Workspace,
        kind: ObservableKind,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        v: &[f64],
        output: &mut [f64],
    ) {
        let (scratch, dp) = ws.observable_tangent(kind);
        seed_param_tangent(dp, &self.sens_param_indices, v);
        self.observables(kind)
            .sens_action(scratch, t, y, inputs, dp, output);
    }

    /// Compute dH/dy · v over every observable of `kind`, into `output`.
    ///
    /// Panics on a model compiled without sensitivities, whose observables carry no tangents.
    #[allow(clippy::too_many_arguments)]
    pub fn observable_jac_action(
        &self,
        ws: &mut Workspace,
        kind: ObservableKind,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        v: &[f64],
        output: &mut [f64],
    ) {
        self.observables(kind)
            .jac_action(ws.observable_scratch(kind), t, y, inputs, v, output);
    }

    /// Project state sensitivities onto output sensitivities.
    ///
    /// Layouts are `y_sens[k * n_states + j]` and `out[k * total_output_len + o]`.
    pub(crate) fn output_sens_project(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        y_sens: &[f64],
        out: &mut [f64],
    ) {
        let (scratch, dp) = ws.observable_tangent(ObservableKind::Outputs);
        self.outputs.sens_project(
            scratch,
            dp,
            t,
            y,
            inputs,
            &self.sens_param_indices,
            y_sens,
            self.n_states,
            out,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::{
        build_trailing_param_sens_model, compile, compile_with_sens, element, identity_mass_matrix,
        input,
    };
    use super::*;
    use crate::node::Node;
    use crate::params::ParamIdx;

    #[test]
    fn an_output_past_the_declared_state_count_is_rejected() {
        let mut arena = Arena::new();
        let narrow = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let rhs = arena.alloc(Node::Neg(narrow));
        let wide = arena.alloc(Node::StateVector { start: 0, end: 8 });

        let mut model = compile(&arena, rhs, identity_mass_matrix(2), 2, 0);
        let err = model
            .set_observables(ObservableKind::Outputs, &mut arena, &[narrow, wide])
            .expect_err("an output past n_states must be refused");
        assert!(
            matches!(
                err,
                CoreError::StateCountTooSmall {
                    declared: 2,
                    extent: 8
                }
            ),
            "unexpected error: {err}"
        );
        assert_eq!(model.n_outputs(), 0, "no root of a refused set compiles");

        let events = model.set_observables(ObservableKind::Events, &mut arena, &[wide]);
        assert!(events.is_err());
        model
            .set_observables(ObservableKind::Outputs, &mut arena, &[narrow])
            .expect("an output inside n_states is accepted");
        assert_eq!(model.n_outputs(), 1);
    }

    #[test]
    fn an_output_sens_action_seeds_its_columns_in_parameter_space() {
        // dH/dk1 = y1 = 4; a column-space read of the seed would give dH/dk0 = y0 = 3.
        let (model, y) = build_trailing_param_sens_model();
        let mut ws = model.create_workspace();
        let mut got = vec![0.0; model.total_output_len()];
        model.observable_sens_action(
            &mut ws,
            ObservableKind::Outputs,
            0.0,
            &y,
            &[5.0, 9.0],
            &[1.0],
            &mut got,
        );
        assert!((got[0] - 4.0).abs() < 1e-10, "expected 4, got {}", got[0]);
    }

    #[test]
    fn output_sens_project_seeds_the_requested_parameter() {
        // dH/dk1 = y[1] = 4, while dH/dk0 = y[0] = 3.
        let (model, y) = build_trailing_param_sens_model();
        let mut ws = model.create_workspace();
        let y_sens = vec![0.0; model.n_states];
        let mut got = vec![0.0; model.total_output_len()];
        model.output_sens_project(&mut ws, 0.0, &y, &[5.0, 9.0], &y_sens, &mut got);
        assert_eq!(got.len(), 1);
        assert!((got[0] - 4.0).abs() < 1e-10, "expected 4, got {}", got[0]);
    }

    #[test]
    fn set_outputs_compile_and_evaluate() {
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = element(&mut arena, y_full, 0);
        let two = arena.alloc(Node::Scalar(2.0));
        let var0 = arena.alloc(Node::Mul(two, y0));

        let mut model = compile(&arena, y_full, identity_mass_matrix(2), 2, 0);
        assert_eq!((model.n_outputs(), model.n_events()), (0, 0));
        assert_eq!((model.total_output_len(), model.total_event_len()), (0, 0));
        model
            .set_observables(ObservableKind::Outputs, &mut arena, &[var0])
            .expect("a valid output");

        assert_eq!(model.n_outputs(), 1);
        assert_eq!(model.output_len_at(0), 1);

        let mut out = [0.0; 1];
        let mut ws = model.create_workspace();
        model.eval_observables(
            &mut ws,
            ObservableKind::Outputs,
            0.0,
            &[3.0, 4.0],
            &[],
            &mut out,
        );
        assert!((out[0] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_outputs_compose_with_sensitivities() {
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let k = input(&mut arena, "k", 0, 0, 1);
        let rhs = arena.alloc(Node::Mul(k, y_full));
        let y0 = element(&mut arena, y_full, 0);

        let mut model = compile_with_sens(
            &arena,
            rhs,
            identity_mass_matrix(2),
            2,
            1,
            &[ParamIdx::new(0)],
        );
        model
            .set_observables(ObservableKind::Outputs, &mut arena, &[y0])
            .expect("a valid output");

        assert_eq!(model.n_sens_params(), 1);
        assert_eq!(model.n_outputs(), 1);
        assert_eq!(model.output_len_at(0), 1);

        let y_vals = [3.0, 4.0];
        let inputs = [2.0];
        let mut ws = model.create_workspace();

        let mut sens_out = [0.0; 2];
        model.eval_sens_all(&mut ws, 0.0, &y_vals, &inputs, &mut sens_out);
        assert!((sens_out[0] - 3.0).abs() < 1e-12);
        assert!((sens_out[1] - 4.0).abs() < 1e-12);

        let mut out = [0.0; 1];
        model.eval_observables(
            &mut ws,
            ObservableKind::Outputs,
            0.0,
            &y_vals,
            &inputs,
            &mut out,
        );
        assert!((out[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn set_events_compile_and_evaluate() {
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = element(&mut arena, y_full, 0);
        let y1 = element(&mut arena, y_full, 1);
        let thresh0 = arena.alloc(Node::Scalar(0.5));
        let thresh1 = arena.alloc(Node::Scalar(1.0));
        let event0 = arena.alloc(Node::Sub(y0, thresh0));
        let event1 = arena.alloc(Node::Sub(y1, thresh1));

        let mut model = compile(&arena, y_full, identity_mass_matrix(2), 2, 0);
        model
            .set_observables(ObservableKind::Events, &mut arena, &[event0])
            .expect("a valid event");
        assert_eq!(model.n_events(), 1);
        assert_eq!(model.total_event_len(), 1);

        let events = ObservableKind::Events;
        let mut ws = model.create_workspace();
        let mut out = [0.0; 1];
        for (y, want) in [([0.7, 1.0], 0.2), ([0.5, 1.0], 0.0), ([0.3, 1.0], -0.2)] {
            model.eval_observables(&mut ws, events, 0.0, &y, &[], &mut out);
            assert!((out[0] - want).abs() < 1e-12, "{y:?}: got {}", out[0]);
        }

        model
            .set_observables(events, &mut arena, &[event0, event1])
            .expect("valid events");
        assert_eq!((model.n_events(), model.total_event_len()), (2, 2));
        let mut ws = model.create_workspace();
        let mut out = [0.0; 2];
        model.eval_observables(&mut ws, events, 0.0, &[0.7, 1.5], &[], &mut out);
        assert!((out[0] - 0.2).abs() < 1e-12);
        assert!((out[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_events_compose_with_outputs() {
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = element(&mut arena, y_full, 0);
        let two = arena.alloc(Node::Scalar(2.0));
        let output_expr = arena.alloc(Node::Mul(two, y0));
        let thresh = arena.alloc(Node::Scalar(0.5));
        let event_expr = arena.alloc(Node::Sub(y0, thresh));

        let mut model = compile(&arena, y_full, identity_mass_matrix(2), 2, 0);
        model
            .set_observables(ObservableKind::Outputs, &mut arena, &[output_expr])
            .expect("a valid output");
        model
            .set_observables(ObservableKind::Events, &mut arena, &[event_expr])
            .expect("a valid event");

        assert_eq!(model.n_outputs(), 1);
        assert_eq!(model.n_events(), 1);

        let mut ws = model.create_workspace();
        let (mut out_val, mut event_val) = ([0.0; 1], [0.0; 1]);
        let y = [3.0, 4.0];
        model.eval_observables(&mut ws, ObservableKind::Outputs, 0.0, &y, &[], &mut out_val);
        model.eval_observables(
            &mut ws,
            ObservableKind::Events,
            0.0,
            &y,
            &[],
            &mut event_val,
        );

        assert!((out_val[0] - 6.0).abs() < 1e-12);
        assert!((event_val[0] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn two_or_more_observables_fuse_bitwise_what_their_own_tapes_do() {
        // Two observables sharing the subexpression `y[0]*y[1] + y[0]`.
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = element(&mut arena, y_full, 0);
        let y1 = element(&mut arena, y_full, 1);
        let prod = arena.alloc(Node::Mul(y0, y1));
        let shared = arena.alloc(Node::Add(prod, y0));
        let half = arena.alloc(Node::Scalar(0.5));
        let first = arena.alloc(Node::Sub(shared, half));
        let two = arena.alloc(Node::Scalar(2.0));
        let second = arena.alloc(Node::Mul(shared, two));

        let mut model = compile(&arena, y_full, identity_mass_matrix(2), 2, 0);
        for kind in [ObservableKind::Outputs, ObservableKind::Events] {
            model
                .set_observables(kind, &mut arena, &[first, second])
                .expect("valid observables");
            assert!(model.observables(kind).fused_expr().is_some());
        }
        assert_eq!(model.total_output_len(), 2);

        // (n_states, k) F-contiguous states, and the k times they pair with.
        let ts = [0.0, 0.5, 1.0];
        let y_cols = [0.7, 1.5, 0.3, -2.0, 1.25, 0.0];
        let k = ts.len();

        let mut expected = [0.0; 6];
        for (lane, &t) in ts.iter().enumerate() {
            let y = &y_cols[lane * 2..(lane + 1) * 2];
            for (row, expr) in model.observable_exprs(ObservableKind::Outputs).enumerate() {
                let mut scratch = vec![0.0; expr.scratch_len()];
                expected[lane * 2 + row] = expr.eval(&mut scratch, t, y, &[], &[])[0];
            }
        }

        let mut ws = model.create_workspace();
        for kind in [ObservableKind::Outputs, ObservableKind::Events] {
            for (lane, &t) in ts.iter().enumerate() {
                let mut fused = [0.0; 2];
                let y = &y_cols[lane * 2..(lane + 1) * 2];
                model.eval_observables(&mut ws, kind, t, y, &[], &mut fused);
                assert_eq!(fused[0].to_bits(), expected[lane * 2].to_bits());
                assert_eq!(fused[1].to_bits(), expected[lane * 2 + 1].to_bits());
            }
        }

        let mut batched = [0.0; 6];
        model.eval_outputs_batch(&mut ws, k, &ts, &y_cols, &[], &mut batched);
        for (got, want) in batched.iter().zip(&expected) {
            assert_eq!(got.to_bits(), want.to_bits());
        }
    }

    #[test]
    fn a_lone_observable_compiles_no_fused_tape() {
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = element(&mut arena, y_full, 0);
        let half = arena.alloc(Node::Scalar(0.5));
        let event0 = arena.alloc(Node::Sub(y0, half));

        let mut model = compile(&arena, y_full, identity_mass_matrix(2), 2, 0);
        let n_nodes = arena.len();
        model
            .set_observables(ObservableKind::Events, &mut arena, &[event0])
            .expect("a valid event");

        assert!(
            model
                .observables(ObservableKind::Events)
                .fused_expr()
                .is_none()
        );
        assert_eq!(
            arena.len(),
            n_nodes,
            "nothing to fuse, so nothing allocated"
        );
    }

    #[test]
    fn an_observable_from_a_graph_that_places_a_parameter_elsewhere_is_rejected() {
        // `k` is index 0 at offset 0 in the model, but at offset 1 in the observable's graph.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let k = input(&mut arena, "k", 0, 0, 1);
        let rhs = arena.alloc(Node::Mul(k, y));
        let mut model = compile(&arena, rhs, identity_mass_matrix(2), 2, 2);

        let mut foreign = Arena::new();
        let y = foreign.alloc(Node::StateVector { start: 0, end: 2 });
        let _first = input(&mut foreign, "first", 1, 0, 1);
        let k = input(&mut foreign, "k", 0, 1, 1);
        let output = foreign.alloc(Node::Mul(k, y));

        let err = model
            .set_observables(ObservableKind::Outputs, &mut foreign, &[output])
            .expect_err("the observable reads k from another packed slot");
        assert!(
            matches!(err, CoreError::ObservableParamSlot { index: 0 }),
            "unexpected error: {err}"
        );
        assert_eq!(model.n_outputs(), 0);
    }

    /// `f(y) = k * y` with `H = k * y[0]^2 - offset`, as `kind`, and sensitivities on `k`.
    fn build_test_model_with_observable_and_sens(
        kind: ObservableKind,
        offset: f64,
    ) -> CompiledModel {
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = element(&mut arena, y_full, 0);
        let k = input(&mut arena, "k", 0, 0, 1);
        let two = arena.alloc(Node::Scalar(2.0));
        let offset = arena.alloc(Node::Scalar(offset));
        let rhs = arena.alloc(Node::Mul(k, y_full));
        let y0_sq = arena.alloc(Node::Pow(y0, two));
        let ky0_sq = arena.alloc(Node::Mul(k, y0_sq));
        let observable = arena.alloc(Node::Sub(ky0_sq, offset));

        let mut model = compile_with_sens(
            &arena,
            rhs,
            identity_mass_matrix(2),
            2,
            1,
            &[ParamIdx::new(0)],
        );
        model
            .set_observables(kind, &mut arena, &[observable])
            .expect("a valid observable");
        model
    }

    /// Both tangent actions of `kind` against central finite differences.
    fn check_observable_actions_against_finite_differences(kind: ObservableKind) {
        let model = build_test_model_with_observable_and_sens(kind, 1.0);
        let mut ws = model.create_workspace();
        let (t, y, inputs) = (0.3, [3.0; 2], [2.0]);
        let len = model.observables(kind).total_len();
        let h = 1e-6;
        let central = |ws: &mut Workspace, plus: (&[f64], &[f64]), minus: (&[f64], &[f64])| {
            let (mut at_plus, mut at_minus) = (vec![0.0; len], vec![0.0; len]);
            model.eval_observables(ws, kind, t, plus.0, plus.1, &mut at_plus);
            model.eval_observables(ws, kind, t, minus.0, minus.1, &mut at_minus);
            let quotient = |(p, m): (&f64, &f64)| (p - m) / (2.0 * h);
            at_plus
                .iter()
                .zip(&at_minus)
                .map(quotient)
                .collect::<Vec<f64>>()
        };

        let mut by_p = vec![0.0; len];
        model.observable_sens_action(&mut ws, kind, t, &y, &inputs, &[1.0], &mut by_p);
        let fd_p = central(&mut ws, (&y, &[2.0 + h]), (&y, &[2.0 - h]));

        let mut by_y = vec![0.0; len];
        model.observable_jac_action(&mut ws, kind, t, &y, &inputs, &[1.0, 0.0], &mut by_y);
        let fd_y = central(
            &mut ws,
            (&[3.0 + h, 3.0], &inputs),
            (&[3.0 - h, 3.0], &inputs),
        );

        for (got, fd) in by_p.iter().zip(&fd_p).chain(by_y.iter().zip(&fd_y)) {
            assert!((got - fd).abs() < 1e-5, "{kind:?}: got {got}, fd {fd}");
        }
    }

    #[test]
    fn output_and_event_actions_match_finite_differences() {
        check_observable_actions_against_finite_differences(ObservableKind::Outputs);
        check_observable_actions_against_finite_differences(ObservableKind::Events);
    }

    #[test]
    fn output_sens_project_equals_action_sum() {
        // projection[k] must equal dH/dp . e_k + dH/dy . y_sens_k
        let model = build_test_model_with_observable_and_sens(ObservableKind::Outputs, 0.0);
        let mut ws = model.create_workspace();
        let (t, y, inputs) = (0.3, [3.0; 2], [2.0]);
        let n_s = model.n_sens_params();
        let n_states = model.n_states;
        let n_out = model.total_output_len();

        let y_sens: Vec<f64> = (0..n_s * n_states)
            .map(|i| 0.1 * (i as f64 + 1.0))
            .collect();
        let mut got = vec![0.0; n_s * n_out];
        model.output_sens_project(&mut ws, t, &y, &inputs, &y_sens, &mut got);

        for k in 0..n_s {
            let mut e_k = vec![0.0; n_s];
            e_k[k] = 1.0;
            let mut sens_term = vec![0.0; n_out];
            let outputs = ObservableKind::Outputs;
            model.observable_sens_action(&mut ws, outputs, t, &y, &inputs, &e_k, &mut sens_term);
            let mut jac_term = vec![0.0; n_out];
            let direction = &y_sens[k * n_states..(k + 1) * n_states];
            model.observable_jac_action(&mut ws, outputs, t, &y, &inputs, direction, &mut jac_term);
            for o in 0..n_out {
                let expected = sens_term[o] + jac_term[o];
                assert!(
                    (got[k * n_out + o] - expected).abs() < 1e-12,
                    "k={k} o={o}: got {} want {}",
                    got[k * n_out + o],
                    expected
                );
            }
        }
    }

    #[test]
    fn an_output_reading_an_input_past_the_declared_width_is_rejected() {
        // Registered after the build and past `n_inputs`: evaluating it would over-read `inputs`.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let k = input(&mut arena, "k", 0, 0, 1);
        let rhs = arena.alloc(Node::Mul(k, y));
        let mut model = compile(&arena, rhs, identity_mass_matrix(2), 2, 1);

        let late = input(&mut arena, "late", 1, 1, 1);
        let observable = arena.alloc(Node::Mul(late, y));
        let err = model
            .set_observables(ObservableKind::Outputs, &mut arena, &[observable])
            .expect_err("an input past n_inputs must be refused");
        assert!(
            matches!(
                err,
                CoreError::InputWidthTooSmall {
                    declared: 1,
                    extent: 2
                }
            ),
            "unexpected error: {err}"
        );
        assert_eq!(model.n_outputs(), 0, "the rejected output was not compiled");
        let events = model.set_observables(ObservableKind::Events, &mut arena, &[observable]);
        assert!(events.is_err());
    }

    #[test]
    fn an_output_reading_an_input_registered_after_the_build_is_seeded() {
        // Inside `n_inputs` but past the build's registrations, which used to size `dp`.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let k = input(&mut arena, "k", 0, 0, 1);
        let rhs = arena.alloc(Node::Mul(k, y));
        let mut model = compile_with_sens(
            &arena,
            rhs,
            identity_mass_matrix(2),
            2,
            2,
            &[ParamIdx::new(0)],
        );

        let late = input(&mut arena, "late", 1, 1, 1);
        let y0 = element(&mut arena, y, 0);
        let output = arena.alloc(Node::Mul(late, y0));
        model
            .set_observables(ObservableKind::Outputs, &mut arena, &[output])
            .expect("an input inside n_inputs is accepted");

        let mut ws = model.create_workspace();
        let (y_vals, inputs) = ([3.0, 4.0], [2.0, 5.0]);
        let mut sens = [f64::NAN];
        model.observable_sens_action(
            &mut ws,
            ObservableKind::Outputs,
            0.0,
            &y_vals,
            &inputs,
            &[1.0],
            &mut sens,
        );
        assert!(sens[0].abs() < 1e-15, "H does not read k, got {}", sens[0]);

        // dH/dk + dH/dy . s = 0 + late * s_0.
        let mut projected = [f64::NAN];
        model.output_sens_project(&mut ws, 0.0, &y_vals, &inputs, &[0.5, 0.0], &mut projected);
        assert!((projected[0] - 2.5).abs() < 1e-12, "got {}", projected[0]);
    }

    #[test]
    #[should_panic(expected = "compiled only with sensitivities")]
    fn observables_skip_their_tangent_tapes_without_sensitivities() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let mut model = compile(&arena, y, identity_mass_matrix(2), 2, 0);
        model
            .set_observables(ObservableKind::Outputs, &mut arena, &[y])
            .expect("a valid output");

        let mut out = [0.0; 2];
        model.observable_jac_action(
            &mut model.create_workspace(),
            ObservableKind::Outputs,
            0.0,
            &[1.0, 2.0],
            &[],
            &[1.0, 0.0],
            &mut out,
        );
    }
}
