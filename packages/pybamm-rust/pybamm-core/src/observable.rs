//! Observables: what a solve reports rather than integrates.
//!
//! Output variables and events are one structure — a primal tape, a `dH/dp`
//! tape, a `dH/dy` tape, a component count, a scratch buffer per tape — so it is
//! written once here and held twice. [`ObservableSet`] owns a family's
//! concatenated layout, its optional fused primal tape and its scratch sizing;
//! [`ObservableKind`] names a family, so a `CompiledModel` method takes one as
//! an argument instead of existing per family.

use std::sync::Arc;

use crate::arena::{Arena, NodeId};
use crate::eval::{CompiledExpr, TangentInputs};
use crate::ir::TypedIr;
use crate::node::Node;
use crate::simplify::simplify_pipeline;
use crate::tangent::{tangent_wrt_params, tangent_wrt_states};

/// Which family of observables a caller means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservableKind {
    /// Output variables `H(t, y; p)`, reported to the caller with their
    /// sensitivities.
    Outputs,
    /// Event functions `g(t, y; p)`, which the integrator roots on.
    Events,
}

/// Write a parameter-space tangent into a global `dp` buffer.
///
/// `sens_params[i]` is the global index of `v[i]`. The compiled tangent tapes are
/// generic over every parameter, so a seed at any global index is valid; this is
/// the one home of that convention, shared by the rhs and observable actions.
pub fn seed_param_tangent(dp: &mut [f64], sens_params: &[usize], v: &[f64]) {
    debug_assert_eq!(
        v.len(),
        sens_params.len(),
        "tangent length must match the parameter mapping",
    );
    dp.fill(0.0);
    for (&global, &vi) in sens_params.iter().zip(v) {
        dp[global] = vi;
    }
}

/// Seed the unit tangent for sensitivity column `column`.
///
/// [`seed_param_tangent`] with `v = e_column`, without a buffer to hold `e`.
pub fn seed_param_tangent_unit(dp: &mut [f64], sens_params: &[usize], column: usize) {
    dp.fill(0.0);
    dp[sens_params[column]] = 1.0;
}

/// One compiled observable: the primal tape plus its two tangent tapes.
#[derive(Debug, Clone)]
pub struct CompiledObservable {
    expr: Arc<CompiledExpr>,
    /// Components this observable writes, captured at compile time.
    len: usize,
    /// `dH/dp`, the param-tangent tape.
    sens_expr: Arc<CompiledExpr>,
    /// `dH/dy`, the state-tangent tape.
    jac_expr: Arc<CompiledExpr>,
}

impl CompiledObservable {
    /// Compile `node` and its two tangent graphs off `arena`.
    ///
    /// `node` must already exist in `arena`.
    pub fn new(arena: &Arena, node: NodeId) -> Self {
        let ir = TypedIr::from_arena(arena, node);
        Self {
            len: ir.output_len(),
            expr: Arc::new(CompiledExpr::from_ir(ir)),
            sens_expr: compile_tangent(arena, node, TangentTarget::Params),
            jac_expr: compile_tangent(arena, node, TangentTarget::States),
        }
    }

    /// Components this observable writes.
    #[inline]
    pub const fn output_len(&self) -> usize {
        self.len
    }

    /// Shared handle to the primal tape (no recompilation).
    #[inline]
    pub const fn expr(&self) -> &Arc<CompiledExpr> {
        &self.expr
    }

    /// The `dH/dp` tape, as a selector for [`ObservableSet::sens_action`].
    #[inline]
    const fn sens_tape(&self) -> &Arc<CompiledExpr> {
        &self.sens_expr
    }

    /// The `dH/dy` tape, as a selector for [`ObservableSet::jac_action`].
    #[inline]
    const fn jac_tape(&self) -> &Arc<CompiledExpr> {
        &self.jac_expr
    }
}

/// Which variable a tangent graph differentiates with respect to.
#[derive(Debug, Clone, Copy)]
enum TangentTarget {
    Params,
    States,
}

/// Compile a tangent graph for `node` over a clone of the model's arena.
///
/// Derivative tapes are simplified; the primal is not (see
/// `CompiledModel::new`), so this entry point is derivative-only.
fn compile_tangent(arena: &Arena, node: NodeId, target: TangentTarget) -> Arc<CompiledExpr> {
    let mut diff_arena = arena.clone();
    let root = match target {
        TangentTarget::Params => tangent_wrt_params(&mut diff_arena, node),
        TangentTarget::States => tangent_wrt_states(&mut diff_arena, node),
    };
    let (final_arena, root) = simplify_pipeline(diff_arena, root);
    Arc::new(CompiledExpr::from_ir(TypedIr::from_arena(
        &final_arena,
        root,
    )))
}

/// A family of observables evaluated together, and the concatenated layout they
/// are reported in.
///
/// Element `i` occupies `[sum(len_at(..i)), sum(len_at(..=i)))` of every buffer
/// the family-wide methods write, which is the layout `PyBaMM` reads output
/// variables and event values back in.
#[derive(Debug, Clone, Default)]
pub struct ObservableSet {
    items: Vec<CompiledObservable>,
    /// One `Concat` tape over every root, evaluated by
    /// [`eval_all`](Self::eval_all) when present. `None` until
    /// [`fuse`](Self::fuse) is called with at least two roots.
    fused: Option<Arc<CompiledExpr>>,
    /// `items`' component counts summed, maintained on push.
    total_len: usize,
}

impl ObservableSet {
    /// An empty set.
    #[inline]
    pub const fn new() -> Self {
        Self {
            items: Vec::new(),
            fused: None,
            total_len: 0,
        }
    }

    /// Compile `node` and append it to the family.
    ///
    /// `node` must already exist in `arena`. Invalidates any fused tape, so
    /// [`fuse`](Self::fuse) belongs after the last push.
    pub fn push(&mut self, arena: &Arena, node: NodeId) {
        let observable = CompiledObservable::new(arena, node);
        self.total_len += observable.len;
        self.items.push(observable);
        self.fused = None;
    }

    /// Build one fused primal tape from a `Concat` of `roots`.
    ///
    /// Per-observable tapes re-evaluate whatever subexpressions the family
    /// shares; lowering a `Concat` of the roots emits each shared arena node
    /// once. No-op for fewer than two observables, which have nothing to share.
    /// `roots` must be the same nodes, in the same order, passed to
    /// [`push`](Self::push), from `arena`.
    pub fn fuse(&mut self, arena: &mut Arena, roots: &[NodeId]) {
        if self.items.len() < 2 {
            return;
        }
        debug_assert_eq!(
            roots.len(),
            self.items.len(),
            "fuse: roots must match the pushed observables",
        );
        let concat = arena.alloc(Node::Concat(roots.to_vec()));
        // Same lowering entry point as `push`, so simplification and instruction
        // semantics are identical to the per-observable tapes.
        let ir = TypedIr::from_arena(arena, concat);
        self.fused = Some(Arc::new(CompiledExpr::from_ir(ir)));
    }

    /// Observables in the family.
    #[inline]
    pub const fn count(&self) -> usize {
        self.items.len()
    }

    /// Whether the family is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Components observable `i` writes.
    ///
    /// Panics if `i >= count()`.
    #[inline]
    pub fn len_at(&self, i: usize) -> usize {
        self.items[i].len
    }

    /// Components the whole family writes, which is the buffer size every
    /// family-wide method needs.
    #[inline]
    pub const fn total_len(&self) -> usize {
        self.total_len
    }

    /// Shared handle to observable `i`'s primal tape (no recompilation).
    ///
    /// Panics if `i >= count()`.
    #[inline]
    pub fn expr_arc(&self, i: usize) -> Arc<CompiledExpr> {
        Arc::clone(&self.items[i].expr)
    }

    /// The fused primal tape, if [`fuse`](Self::fuse) built one.
    #[inline]
    pub const fn fused_expr(&self) -> Option<&Arc<CompiledExpr>> {
        self.fused.as_ref()
    }

    /// Allocate per-solve buffers sized for this family.
    pub fn create_scratch(&self) -> ObservableScratch {
        ObservableScratch {
            primal: self.scratches(CompiledObservable::expr),
            sens: self.scratches(CompiledObservable::sens_tape),
            jac: self.scratches(CompiledObservable::jac_tape),
            fused: self
                .fused
                .as_ref()
                .map_or_else(Vec::new, |e| vec![0.0; e.scratch_len()]),
            batch: Vec::new(),
            jac_term: vec![0.0; self.total_len],
        }
    }

    /// One zeroed buffer per observable, sized for the tape `tape` selects.
    fn scratches(&self, tape: fn(&CompiledObservable) -> &Arc<CompiledExpr>) -> Vec<Vec<f64>> {
        self.items
            .iter()
            .map(|item| vec![0.0; tape(item).scratch_len()])
            .collect()
    }

    /// Evaluate observable `i` into `out`, returning the count written.
    ///
    /// Panics if `i >= count()` or `out.len() < len_at(i)`.
    pub fn eval_at(
        &self,
        scratch: &mut ObservableScratch,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        i: usize,
        out: &mut [f64],
    ) -> usize {
        let result = self.items[i]
            .expr
            .eval(&mut scratch.primal[i], t, y, &[], inputs);
        out[..result.len()].copy_from_slice(result);
        result.len()
    }

    /// Evaluate the whole family into `out`, in the concatenated layout.
    ///
    /// Takes the fused tape when the family carries one. Panics if
    /// `out.len() < total_len()`.
    pub fn eval_all(
        &self,
        scratch: &mut ObservableScratch,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        out: &mut [f64],
    ) {
        if let Some(fused) = &self.fused {
            let result = fused.eval(&mut scratch.fused, t, y, &[], inputs);
            out[..self.total_len].copy_from_slice(&result[..self.total_len]);
            return;
        }
        let mut offset = 0;
        for (item, buffer) in self.items.iter().zip(&mut scratch.primal) {
            let result = item.expr.eval(buffer, t, y, &[], inputs);
            out[offset..offset + item.len].copy_from_slice(&result[..item.len]);
            offset += item.len;
        }
    }

    /// Batch-evaluate the whole family over `k` trajectory points.
    ///
    /// `ts` holds the `k` times, `y_cols` the `(n_states, k)` F-contiguous state
    /// matrix, and `out` receives the `(total_len, k)` F-contiguous matrix. Tapes
    /// the batch evaluator rejects (e.g. a `y_dot` reference) fall back to
    /// per-point scalar evaluation, so results always match `k`
    /// [`eval_all`](Self::eval_all) calls bitwise.
    // Times, states, inputs and the output matrix are all distinct arguments.
    #[allow(clippy::too_many_arguments)]
    pub fn eval_batch(
        &self,
        scratch: &mut ObservableScratch,
        k: usize,
        ts: &[f64],
        y_cols: &[f64],
        n_states: usize,
        inputs: &[f64],
        out: &mut [f64],
    ) {
        let ObservableScratch { primal, batch, .. } = scratch;
        let n_total = self.total_len;
        let mut offset = 0;
        for (item, fallback) in self.items.iter().zip(primal.iter_mut()) {
            let (expr, len) = (&item.expr, item.len);
            let needed = expr.scratch_len() * k;
            if batch.len() < needed {
                batch.resize(needed, 0.0);
            }
            match expr.eval_batch(&mut batch[..needed], k, ts, y_cols, inputs) {
                Ok(result) => {
                    // result is lane-minor (`result[row * k + lane]`); out is
                    // F-contiguous.
                    for row in 0..len {
                        for (lane, &v) in result[row * k..(row + 1) * k].iter().enumerate() {
                            out[lane * n_total + offset + row] = v;
                        }
                    }
                },
                Err(_) => {
                    for lane in 0..k {
                        let y = &y_cols[lane * n_states..(lane + 1) * n_states];
                        let result = expr.eval(fallback, ts[lane], y, &[], inputs);
                        out[lane * n_total + offset..lane * n_total + offset + len]
                            .copy_from_slice(&result[..len]);
                    }
                },
            }
            offset += len;
        }
    }

    /// Compute `dH/dp . v` over the whole family into `out`.
    ///
    /// `dp` is the global parameter-space tangent, seeded by
    /// [`seed_param_tangent`]. Panics if `out.len() < total_len()`.
    pub fn sens_action(
        &self,
        scratch: &mut ObservableScratch,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        dp: &[f64],
        out: &mut [f64],
    ) {
        let tangent = TangentInputs {
            dy: None,
            dp: Some(dp),
        };
        Self::tangent_action(
            &self.items,
            CompiledObservable::sens_tape,
            &mut scratch.sens,
            t,
            y,
            inputs,
            &tangent,
            out,
        );
    }

    /// Compute `dH/dy . v` over the whole family into `out`.
    ///
    /// `v` is a state-space tangent of length `n_states`. Panics if
    /// `out.len() < total_len()`.
    pub fn jac_action(
        &self,
        scratch: &mut ObservableScratch,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) {
        let tangent = TangentInputs {
            dy: Some(v),
            dp: None,
        };
        Self::tangent_action(
            &self.items,
            CompiledObservable::jac_tape,
            &mut scratch.jac,
            t,
            y,
            inputs,
            &tangent,
            out,
        );
    }

    /// Project state sensitivities onto observable sensitivities.
    ///
    /// `dH/dp_k = sens_action(e_k) + jac_action(y_sens_k)` for every sensitivity
    /// column, where `sens_params[k]` is column `k`'s global parameter index.
    ///   `y_sens` layout: `y_sens[k * n_states + j]`
    ///   `out` layout: `out[k * total_len() + o]`
    /// `dp` is the caller's global tangent buffer, reseeded per column.
    // Evaluation point, tangent buffers, mapping and output are all distinct.
    #[allow(clippy::too_many_arguments)]
    pub fn sens_project(
        &self,
        scratch: &mut ObservableScratch,
        dp: &mut [f64],
        t: f64,
        y: &[f64],
        inputs: &[f64],
        sens_params: &[usize],
        y_sens: &[f64],
        n_states: usize,
        out: &mut [f64],
    ) {
        let n_out = self.total_len;
        if n_out == 0 || sens_params.is_empty() {
            return;
        }
        // Destructured so the two tangent passes and the accumulator can borrow
        // disjoint buffers of one scratch.
        let ObservableScratch {
            sens,
            jac,
            jac_term,
            ..
        } = scratch;
        for k in 0..sens_params.len() {
            seed_param_tangent_unit(dp, sens_params, k);
            let dst = &mut out[k * n_out..(k + 1) * n_out];
            Self::tangent_action(
                &self.items,
                CompiledObservable::sens_tape,
                sens,
                t,
                y,
                inputs,
                &TangentInputs {
                    dy: None,
                    dp: Some(dp),
                },
                dst,
            );
            Self::tangent_action(
                &self.items,
                CompiledObservable::jac_tape,
                jac,
                t,
                y,
                inputs,
                &TangentInputs {
                    dy: Some(&y_sens[k * n_states..(k + 1) * n_states]),
                    dp: None,
                },
                jac_term,
            );
            for (dst_o, &term) in dst.iter_mut().zip(jac_term.iter()) {
                *dst_o += term; // + dH/dy . y_sens_k
            }
        }
    }

    /// The one tangent-action loop, over whichever tape `tape` selects.
    ///
    /// Walks observables and their buffers in lockstep and writes each result
    /// into its slice of the concatenated layout.
    // Tapes, buffers, evaluation point, tangent and output are distinct groups.
    #[allow(clippy::too_many_arguments)]
    fn tangent_action(
        items: &[CompiledObservable],
        tape: fn(&CompiledObservable) -> &Arc<CompiledExpr>,
        scratches: &mut [Vec<f64>],
        t: f64,
        y: &[f64],
        inputs: &[f64],
        tangent: &TangentInputs<'_>,
        out: &mut [f64],
    ) {
        let mut offset = 0;
        for (item, buffer) in items.iter().zip(scratches.iter_mut()) {
            let result = tape(item).eval_with_tangent(buffer, t, y, &[], inputs, tangent);
            out[offset..offset + item.len].copy_from_slice(&result[..item.len]);
            offset += item.len;
        }
    }
}

/// Per-solve mutable buffers for one [`ObservableSet`].
///
/// Sized from the set, one buffer per tape, so a caller can neither mis-size a
/// buffer nor pair a scratch with the wrong family. Create one per solve; two
/// concurrent evaluations of the same set need two.
#[derive(Debug, Clone, Default)]
pub struct ObservableScratch {
    /// One buffer per observable's primal tape.
    primal: Vec<Vec<f64>>,
    /// One buffer per observable's `dH/dp` tape.
    sens: Vec<Vec<f64>>,
    /// One buffer per observable's `dH/dy` tape.
    jac: Vec<Vec<f64>>,
    /// The fused tape's buffer; empty unless the set carries one.
    fused: Vec<f64>,
    /// Lane-scaled buffer for [`ObservableSet::eval_batch`], grown on demand and
    /// reused across windows.
    batch: Vec<f64>,
    /// `dH/dy . y_sens_k` for one column of
    /// [`ObservableSet::sens_project`].
    jac_term: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;

    /// Assert values, at a tolerance these small-integer fixtures never need.
    fn assert_close(got: &[f64], want: &[f64]) {
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-12, "got {got:?}, want {want:?}");
        }
    }

    /// `H = [y0 * y1 + p0, y0 * y1 - 1]` as two observables over one shared
    /// product, so fusing has something to share and both tangents are nonzero.
    fn build_pair() -> (Arena, ObservableSet, Vec<NodeId>) {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = arena.alloc(Node::Index {
            child: y,
            start: 0,
            end: 1,
        });
        let y1 = arena.alloc(Node::Index {
            child: y,
            start: 1,
            end: 2,
        });
        let product = arena.alloc(Node::Mul(y0, y1));
        let p0 = arena.alloc(Node::InputParameter {
            name: "p0".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let first = arena.alloc(Node::Add(product, p0));
        let one = arena.alloc(Node::Scalar(1.0));
        let second = arena.alloc(Node::Sub(product, one));

        let mut set = ObservableSet::new();
        set.push(&arena, first);
        set.push(&arena, second);
        (arena, set, vec![first, second])
    }

    #[test]
    fn a_set_reports_its_concatenated_layout() {
        let (_arena, set, _roots) = build_pair();
        assert_eq!(set.count(), 2);
        assert_eq!(set.len_at(0), 1);
        assert_eq!(set.len_at(1), 1);
        assert_eq!(set.total_len(), 2);
        assert!(!set.is_empty());
        assert!(ObservableSet::new().is_empty());
    }

    #[test]
    fn eval_all_concatenates_what_eval_at_writes() {
        let (_arena, set, _roots) = build_pair();
        let mut scratch = set.create_scratch();
        let (y, inputs) = ([3.0, 5.0], [7.0]);

        let mut all = [0.0; 2];
        set.eval_all(&mut scratch, 0.0, &y, &inputs, &mut all);
        assert_close(&all, &[22.0, 14.0]);

        for (i, &concatenated) in all.iter().enumerate() {
            let mut one = [0.0; 1];
            assert_eq!(set.eval_at(&mut scratch, 0.0, &y, &inputs, i, &mut one), 1);
            assert_eq!(one[0].to_bits(), concatenated.to_bits());
        }
    }

    #[test]
    fn a_fused_set_evaluates_bitwise_what_the_per_item_tapes_do() {
        let (mut arena, mut set, roots) = build_pair();
        let mut per_item_scratch = set.create_scratch();
        let (y, inputs) = ([0.75, -1.5], [0.25]);
        let mut per_item = [0.0; 2];
        set.eval_all(&mut per_item_scratch, 0.0, &y, &inputs, &mut per_item);

        set.fuse(&mut arena, &roots);
        assert!(set.fused_expr().is_some());
        let mut scratch = set.create_scratch();
        let mut fused = [0.0; 2];
        set.eval_all(&mut scratch, 0.0, &y, &inputs, &mut fused);

        assert_eq!(fused.map(f64::to_bits), per_item.map(f64::to_bits));
        // CSE across the roots: the shared product is emitted once.
        let fused_instructions = set.fused_expr().unwrap().ir().instructions().len();
        let per_item_instructions: usize = (0..set.count())
            .map(|i| set.expr_arc(i).ir().instructions().len())
            .sum();
        assert!(
            fused_instructions < per_item_instructions,
            "fused tape ({fused_instructions}) should be shorter than the \
             per-item sum ({per_item_instructions})",
        );
    }

    #[test]
    fn fusing_fewer_than_two_observables_is_a_noop() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let mut set = ObservableSet::new();
        set.push(&arena, y);
        set.fuse(&mut arena, &[y]);
        assert!(set.fused_expr().is_none());
    }

    #[test]
    fn a_push_after_fusing_drops_the_stale_tape() {
        let (mut arena, mut set, roots) = build_pair();
        set.fuse(&mut arena, &roots);
        assert!(set.fused_expr().is_some());
        set.push(&arena, roots[0]);
        assert!(
            set.fused_expr().is_none(),
            "a fused tape that predates an observable would under-report the family",
        );
    }

    #[test]
    fn the_tangent_actions_differentiate_the_family() {
        let (_arena, set, _roots) = build_pair();
        let mut scratch = set.create_scratch();
        let (y, inputs) = ([3.0, 5.0], [7.0]);

        // dH/dy . v with v = [1, 0] is [y1, y1].
        let mut jac = [0.0; 2];
        set.jac_action(&mut scratch, 0.0, &y, &inputs, &[1.0, 0.0], &mut jac);
        assert_close(&jac, &[5.0, 5.0]);

        // dH/dp . e_0 is [1, 0]: only the first observable reads p0.
        let mut dp = [0.0];
        seed_param_tangent(&mut dp, &[0], &[1.0]);
        let mut sens = [0.0; 2];
        set.sens_action(&mut scratch, 0.0, &y, &inputs, &dp, &mut sens);
        assert_close(&sens, &[1.0, 0.0]);
    }

    #[test]
    fn sens_project_sums_the_two_actions() {
        let (_arena, set, _roots) = build_pair();
        let mut scratch = set.create_scratch();
        let (y, inputs) = ([3.0, 5.0], [7.0]);
        let sens_params = [0];
        let y_sens = [1.0, 0.0]; // dy/dp0

        let mut projected = [0.0; 2];
        let mut dp = [0.0];
        set.sens_project(
            &mut scratch,
            &mut dp,
            0.0,
            &y,
            &inputs,
            &sens_params,
            &y_sens,
            2,
            &mut projected,
        );

        // dH/dp0 + dH/dy . dy/dp0 = [1, 0] + [y1, y1].
        assert_close(&projected, &[6.0, 5.0]);
    }

    #[test]
    fn eval_batch_matches_a_loop_of_eval_all() {
        let (_arena, set, _roots) = build_pair();
        let mut scratch = set.create_scratch();
        let inputs = [7.0];
        let ts = [0.0, 1.0, 2.0];
        let y_cols = [3.0, 5.0, 1.0, 2.0, -4.0, 0.5];

        let mut batched = [0.0; 6];
        set.eval_batch(&mut scratch, 3, &ts, &y_cols, 2, &inputs, &mut batched);

        for lane in 0..3 {
            let mut expected = [0.0; 2];
            set.eval_all(
                &mut scratch,
                ts[lane],
                &y_cols[lane * 2..(lane + 1) * 2],
                &inputs,
                &mut expected,
            );
            assert_eq!(
                batched[lane * 2..(lane + 1) * 2]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected.map(f64::to_bits).to_vec(),
            );
        }
    }
}
