//! Observables (output variables and events): what a solve reports rather than integrates.

use std::sync::Arc;

use crate::arena::{Arena, DagRef, NodeId};
use crate::eval::{CompiledExpr, SplitCompiledExpr, TangentInputs};
use crate::jacobian::tangent_dag;
use crate::node::Node;
use crate::params::{ParamIdx, SensIdx};
use crate::simplify::simplify_pipeline;
use crate::tangent::{tangent_wrt_params, tangent_wrt_states};
use crate::tangent_batch::{self, LanePlan};

/// Which family of observables a caller means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservableKind {
    /// Output variables `H(t, y; p)`.
    Outputs,
    /// Event functions `g(t, y; p)`.
    Events,
}

/// Write a parameter-space tangent into `dp`; `sens_params[i]` is the registration index of `v[i]`.
pub fn seed_param_tangent(dp: &mut [f64], sens_params: &[ParamIdx], v: &[f64]) {
    // In release too: `zip` would silently drop the surplus entries.
    assert_eq!(
        v.len(),
        sens_params.len(),
        "tangent length must match the parameter mapping",
    );
    dp.fill(0.0);
    for (&param, &vi) in sens_params.iter().zip(v) {
        dp[param.index()] = vi;
    }
}

/// Seed the unit tangent for sensitivity column `column`.
fn seed_param_tangent_unit(dp: &mut [f64], sens_params: &[ParamIdx], column: SensIdx) {
    dp.fill(0.0);
    dp[sens_params[column.get()].index()] = 1.0;
}

/// Panic message for a tangent action on observables compiled without sensitivities.
const NO_TANGENTS: &str = "observable tangents are compiled only with sensitivities";

/// A family's derivative tapes, each split so one primal pass serves every direction.
#[derive(Debug)]
struct ObservableTangents {
    /// `dH/dp`.
    sens: SplitCompiledExpr,
    /// `dH/dy`.
    jac: SplitCompiledExpr,
    /// How sensitivity directions walk `jac`.
    jac_lanes: LanePlan,
}

impl ObservableTangents {
    /// Both tangent tapes of `root`, the whole family's root.
    fn compile(arena: &Arena, root: NodeId) -> Self {
        let jac = compile_tangent(arena, root, tangent_wrt_states);
        Self {
            sens: compile_tangent(arena, root, tangent_wrt_params),
            jac_lanes: LanePlan::new(&jac),
            jac,
        }
    }
}

/// Compile the simplified `differentiate` graph of `node`.
fn compile_tangent(
    arena: &Arena,
    node: NodeId,
    differentiate: fn(&mut Arena, NodeId) -> NodeId,
) -> SplitCompiledExpr {
    let tangent = simplify_pipeline(tangent_dag(arena, node, differentiate));
    SplitCompiledExpr::compile(tangent.as_ref())
}

/// A family of observables evaluated together.
///
/// Family-wide buffers concatenate items, in root order.
#[derive(Debug, Default)]
pub struct ObservableSet {
    /// Each observable's own tape, in root order.
    items: Vec<Arc<CompiledExpr>>,
    /// One `Concat` tape over every root, when there are two or more.
    fused: Option<CompiledExpr>,
    /// Tangent tapes of the family root, when sensitivities need them.
    tangents: Option<ObservableTangents>,
    total_len: usize,
}

impl ObservableSet {
    /// Compile each of `roots`, fuse two or more, and differentiate the family when
    /// `with_tangents`.
    ///
    /// The family root is a `Concat` of the roots, which this allocates in `arena`, so shared
    /// subexpressions evaluate once in the primal and in both tangent tapes.
    pub fn compile(arena: &mut Arena, roots: &[NodeId], with_tangents: bool) -> Self {
        let items: Vec<Arc<CompiledExpr>> = roots
            .iter()
            .map(|&root| Arc::new(CompiledExpr::new(DagRef::new(arena, root))))
            .collect();
        let (family, fused) = match roots {
            [] => (None, None),
            &[root] => (Some(root), None),
            _ => {
                let concat = arena.alloc(Node::Concat(roots.to_vec()));
                let fused = CompiledExpr::new(DagRef::new(arena, concat));
                (Some(concat), Some(fused))
            },
        };
        Self {
            total_len: items.iter().map(|expr| expr.output_len()).sum(),
            tangents: family
                .filter(|_| with_tangents)
                .map(|root| ObservableTangents::compile(arena, root)),
            items,
            fused,
        }
    }

    #[inline]
    pub const fn count(&self) -> usize {
        self.items.len()
    }

    /// Components observable `i` writes.
    #[inline]
    pub fn len_at(&self, i: usize) -> usize {
        self.items[i].output_len()
    }

    /// Components the whole family writes (the family-wide buffer size).
    #[inline]
    pub const fn total_len(&self) -> usize {
        self.total_len
    }

    /// Each observable's own tape, in root order.
    pub fn exprs(&self) -> impl ExactSizeIterator<Item = &Arc<CompiledExpr>> {
        self.items.iter()
    }

    #[cfg(test)]
    pub const fn fused_expr(&self) -> Option<&CompiledExpr> {
        self.fused.as_ref()
    }

    /// Most directions one `dH/dy` walk carries.
    #[cfg(test)]
    pub fn widest_jac_walk(&self) -> usize {
        self.tangents().jac_lanes.widest()
    }

    #[inline]
    const fn tangents(&self) -> &ObservableTangents {
        self.tangents.as_ref().expect(NO_TANGENTS)
    }

    /// Allocate per-solve buffers sized for this family.
    pub fn create_scratch(&self) -> ObservableScratch {
        let tangents = self.tangents.iter();
        let tape_len = self
            .items
            .iter()
            .map(|expr| expr.scratch_len())
            .chain(self.fused.as_ref().map(CompiledExpr::scratch_len))
            .chain(tangents.flat_map(|t| [t.sens.scratch_len(), t.jac.scratch_len()]))
            .max()
            .unwrap_or(0);
        let batched = self.tangents.as_ref().filter(|t| t.jac_lanes.widest() > 1);
        ObservableScratch {
            tape: vec![0.0; tape_len],
            batch: Vec::new(),
            tangent_lanes: batched.map_or_else(Vec::new, |t| {
                vec![0.0; tangent_batch::tangent_scratch_len(&t.jac, t.jac_lanes.widest())]
            }),
            seed_lanes: Vec::new(),
        }
    }

    /// Evaluate the whole family into `out`, in the concatenated layout.
    pub fn eval_all(
        &self,
        scratch: &mut ObservableScratch,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        out: &mut [f64],
    ) {
        let mut offset = 0;
        for (expr, len) in self.primal_tapes() {
            let result = expr.eval(&mut scratch.tape, t, y, &[], inputs);
            out[offset..offset + len].copy_from_slice(&result[..len]);
            offset += len;
        }
    }

    /// Batch-evaluate the whole family over `k` trajectory points.
    ///
    /// `y_cols` is `(n_states, k)` and `out` `(total_len, k)`, both F-contiguous.
    /// Results match `k` `eval_all` calls bitwise.
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
        let ObservableScratch { tape, batch, .. } = scratch;
        let y_cols = &y_cols[..k * n_states];
        let n_total = self.total_len;
        let mut offset = 0;
        for (expr, len) in self.primal_tapes() {
            let needed = expr.scratch_len() * k;
            if batch.len() < needed {
                batch.resize(needed, 0.0);
            }
            match expr.eval_batch(&mut batch[..needed], k, ts, y_cols, n_states, inputs) {
                Ok(result) => scatter_lane_minor(result, k, n_total, offset, len, out),
                Err(_) => {
                    for lane in 0..k {
                        let y = &y_cols[lane * n_states..(lane + 1) * n_states];
                        let result = expr.eval(tape, ts[lane], y, &[], inputs);
                        out[lane * n_total + offset..lane * n_total + offset + len]
                            .copy_from_slice(&result[..len]);
                    }
                },
            }
            offset += len;
        }
    }

    /// The tapes that evaluate the family, each with its output length: the fused tape alone,
    /// or one per item in root order.
    fn primal_tapes(&self) -> impl Iterator<Item = (&CompiledExpr, usize)> {
        let fused = self.fused.as_ref().map(|expr| (expr, self.total_len));
        let items: &[Arc<CompiledExpr>] = if fused.is_some() { &[] } else { &self.items };
        fused
            .into_iter()
            .chain(items.iter().map(|expr| (&**expr, expr.output_len())))
    }

    /// Compute `dH/dp . v` into `out`; `dp` is `v` scattered to registration indices.
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
        self.tangent_action(|tapes| &tapes.sens, scratch, t, y, inputs, &tangent, out);
    }

    /// Compute `dH/dy . v` over the whole family into `out`.
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
        self.tangent_action(|tapes| &tapes.jac, scratch, t, y, inputs, &tangent, out);
    }

    #[allow(clippy::too_many_arguments)]
    fn tangent_action(
        &self,
        tape: fn(&ObservableTangents) -> &SplitCompiledExpr,
        scratch: &mut ObservableScratch,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        tangent: &TangentInputs<'_>,
        out: &mut [f64],
    ) {
        if self.items.is_empty() {
            return;
        }
        let result =
            tape(self.tangents()).eval_with_tangent(&mut scratch.tape, t, y, &[], inputs, tangent);
        out[..self.total_len].copy_from_slice(result);
    }

    /// Project state sensitivities: `dH/dp_k = sens_action(e_k) + jac_action(y_sens_k)`.
    ///
    /// Layouts are `y_sens[k * n_states + j]` and `out[k * total_len() + o]`.
    #[allow(clippy::too_many_arguments)]
    pub fn sens_project(
        &self,
        scratch: &mut ObservableScratch,
        dp: &mut [f64],
        t: f64,
        y: &[f64],
        inputs: &[f64],
        sens_params: &[ParamIdx],
        y_sens: &[f64],
        n_states: usize,
        out: &mut [f64],
    ) {
        let (n_out, n_sens) = (self.total_len, sens_params.len());
        if n_out == 0 || n_sens == 0 {
            return;
        }
        let ObservableTangents {
            sens,
            jac,
            jac_lanes,
        } = self.tangents();
        let ObservableScratch {
            tape,
            tangent_lanes,
            seed_lanes,
            ..
        } = scratch;
        let columns = &mut out[..n_sens * n_out];

        let mut primal = sens.eval_primal(tape, t, y, &[], inputs);
        for (k, column) in columns.chunks_exact_mut(n_out).enumerate() {
            seed_param_tangent_unit(dp, sens_params, SensIdx::new(k));
            let tangent = TangentInputs {
                dy: None,
                dp: Some(dp),
            };
            column.copy_from_slice(primal.eval_tangent(&tangent));
        }

        let mut primal = jac.eval_primal(tape, t, y, &[], inputs);
        let mut k = 0;
        while k < n_sens {
            let lanes = jac_lanes.lanes(n_sens - k);
            if lanes == 1 {
                let tangent = TangentInputs {
                    dy: Some(&y_sens[k * n_states..(k + 1) * n_states]),
                    dp: None,
                };
                let result = primal.eval_tangent(&tangent);
                add_lane(&mut columns[k * n_out..(k + 1) * n_out], result, 1, 0);
                k += 1;
                continue;
            }
            // Surplus lanes of a part-empty walk keep zero seeds, and nothing reads them.
            let carried = lanes.min(n_sens - k);
            seed_lanes.clear();
            seed_lanes.resize(n_states * lanes, 0.0);
            for lane in 0..carried {
                let direction = &y_sens[(k + lane) * n_states..(k + lane + 1) * n_states];
                for (state, &value) in direction.iter().enumerate() {
                    seed_lanes[state * lanes + lane] = value;
                }
            }
            let result = primal.eval_tangent_batch(lanes, tangent_lanes, seed_lanes);
            for lane in 0..carried {
                let column = &mut columns[(k + lane) * n_out..(k + lane + 1) * n_out];
                add_lane(column, result, lanes, lane);
            }
            k += carried;
        }
    }
}

/// Add lane `lane` of lane-minor `result`, `lanes` wide, into `column`.
fn add_lane(column: &mut [f64], result: &[f64], lanes: usize, lane: usize) {
    for (dst, &term) in column
        .iter_mut()
        .zip(result.iter().skip(lane).step_by(lanes))
    {
        *dst += term;
    }
}

/// Scatter lane-minor `result[row * k + lane]` into rows `offset..offset + len` of
/// F-contiguous `out`.
fn scatter_lane_minor(
    result: &[f64],
    k: usize,
    n_total: usize,
    offset: usize,
    len: usize,
    out: &mut [f64],
) {
    for row in 0..len {
        for (lane, &v) in result[row * k..(row + 1) * k].iter().enumerate() {
            out[lane * n_total + offset + row] = v;
        }
    }
}

/// Per-solve mutable buffers for one [`ObservableSet`]; concurrent evaluations need one each.
#[derive(Debug, Default)]
pub struct ObservableScratch {
    /// Sized for the family's widest tape; each evaluation runs one whole tape, so all share it.
    tape: Vec<f64>,
    /// Lane-scaled buffer for [`ObservableSet::eval_batch`], grown on demand.
    batch: Vec<f64>,
    /// Lane-minor tangent region of the batched `dH/dy` walks; empty when they never batch.
    tangent_lanes: Vec<f64>,
    /// Lane-minor `dH/dy` seeds, grown on demand.
    seed_lanes: Vec<f64>,
}

#[cfg(test)]
impl ObservableScratch {
    /// Overwrite every buffer with `value`, as an unwound call could leave them.
    pub fn poison(&mut self, value: f64) {
        self.tape.fill(value);
        self.tangent_lanes.fill(value);
        crate::jacobian::poison_grown(&mut self.batch, value);
        crate::jacobian::poison_grown(&mut self.seed_lanes, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;

    fn assert_close(got: &[f64], want: &[f64]) {
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-12, "got {got:?}, want {want:?}");
        }
    }

    /// `H = [y0 * y1 + p0, y0 * y1 - 1]`: two observables sharing one product.
    fn build_pair() -> (Arena, ObservableSet) {
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

        let set = ObservableSet::compile(&mut arena, &[first, second], true);
        (arena, set)
    }

    #[test]
    fn a_set_reports_its_concatenated_layout() {
        let (_arena, set) = build_pair();
        assert_eq!(set.count(), 2);
        assert_eq!(set.len_at(0), 1);
        assert_eq!(set.len_at(1), 1);
        assert_eq!(set.total_len(), 2);
        assert_eq!(set.exprs().len(), 2);
        assert_eq!(ObservableSet::default().count(), 0);
    }

    #[test]
    fn a_fused_set_evaluates_bitwise_what_the_per_item_tapes_do() {
        let (_arena, set) = build_pair();
        let fused = set.fused_expr().expect("two observables fuse");
        let mut scratch = set.create_scratch();
        let (y, inputs) = ([0.75, -1.5], [0.25]);

        let mut all = [0.0; 2];
        set.eval_all(&mut scratch, 0.0, &y, &inputs, &mut all);
        assert_close(&all, &[-0.875, -2.125]);

        let per_item = set
            .exprs()
            .map(|expr| expr.eval(&mut scratch.tape, 0.0, &y, &[], &inputs)[0].to_bits());
        assert!(per_item.eq(all.map(f64::to_bits)));

        let fused_instructions = fused.ir().instructions().len();
        let per_item_instructions: usize =
            set.exprs().map(|expr| expr.ir().instructions().len()).sum();
        assert!(
            fused_instructions < per_item_instructions,
            "fused tape ({fused_instructions}) should be shorter than the \
             per-item sum ({per_item_instructions})",
        );
    }

    #[test]
    fn a_lone_observable_fuses_nothing() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let n_nodes = arena.len();
        let set = ObservableSet::compile(&mut arena, &[y], false);
        assert!(set.fused_expr().is_none());
        assert_eq!(arena.len(), n_nodes);
    }

    #[test]
    fn the_tangent_actions_differentiate_the_family() {
        let (_arena, set) = build_pair();
        let mut scratch = set.create_scratch();
        let (y, inputs) = ([3.0, 5.0], [7.0]);

        let mut jac = [0.0; 2];
        set.jac_action(&mut scratch, 0.0, &y, &inputs, &[1.0, 0.0], &mut jac);
        assert_close(&jac, &[5.0, 5.0]);

        let mut dp = [0.0];
        seed_param_tangent(&mut dp, &[ParamIdx::new(0)], &[1.0]);
        let mut sens = [0.0; 2];
        set.sens_action(&mut scratch, 0.0, &y, &inputs, &dp, &mut sens);
        assert_close(&sens, &[1.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "tangent length must match")]
    fn a_tangent_longer_than_its_parameter_mapping_is_refused() {
        // Before, release builds zipped the surplus entry away and seeded a partial tangent.
        let mut dp = [0.0; 2];
        seed_param_tangent(&mut dp, &[ParamIdx::new(1)], &[1.0, 2.0]);
    }

    /// Every tape writes each slot it reads, so one shared buffer needs no clearing between tapes.
    #[test]
    fn the_shared_tape_buffer_carries_nothing_between_evaluations() {
        let (_arena, set) = build_pair();
        let (y, inputs, v) = ([3.0, 5.0], [7.0], [0.25, -1.5]);
        let mut dp = [0.0];
        seed_param_tangent(&mut dp, &[ParamIdx::new(0)], &[1.0]);
        let run = |scratch: &mut ObservableScratch| {
            let (mut all, mut jac, mut sens) = ([0.0; 2], [0.0; 2], [0.0; 2]);
            set.eval_all(scratch, 0.0, &y, &inputs, &mut all);
            set.jac_action(scratch, 0.0, &y, &inputs, &v, &mut jac);
            set.sens_action(scratch, 0.0, &y, &inputs, &dp, &mut sens);
            [all, jac, sens].map(|values| values.map(f64::to_bits))
        };

        let clean = run(&mut set.create_scratch());
        let mut dirty = set.create_scratch();
        dirty.tape.fill(f64::NAN);
        assert_eq!(run(&mut dirty), clean);
    }

    #[test]
    fn sens_project_sums_the_two_actions() {
        let (_arena, set) = build_pair();
        let mut scratch = set.create_scratch();
        let (y, inputs) = ([3.0, 5.0], [7.0]);
        let sens_params = [ParamIdx::new(0)];
        let y_sens = [1.0, 0.0];

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

        assert_close(&projected, &[6.0, 5.0]);
    }

    /// One primal pass per tape must project each column as a full per-column sweep does.
    #[test]
    fn sens_project_matches_the_actions_bitwise_in_every_column() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let [p, q] = ["p", "q"].map(|name| {
            let index = usize::from(name == "q");
            arena.alloc(Node::InputParameter {
                name: name.to_string(),
                index,
                offset: index,
                width: 1,
            })
        });
        let py = arena.alloc(Node::Mul(p, y));
        let sin_py = arena.alloc(Node::Sin(py));
        let first = arena.alloc(Node::Mul(q, sin_py));
        let qy = arena.alloc(Node::Mul(q, y));
        let exp_qy = arena.alloc(Node::Exp(qy));
        let second = arena.alloc(Node::Add(exp_qy, py));
        let set = ObservableSet::compile(&mut arena, &[first, second], true);

        let mut scratch = set.create_scratch();
        let n_out = set.total_len();
        let sens_params = [ParamIdx::new(1), ParamIdx::new(0)];
        let (y, inputs, y_sens) = ([0.3, -0.8], [1.7, 0.6], [0.1, 0.2, -0.3, 0.4]);
        let mut dp = [0.0; 2];
        let mut projected = vec![0.0; sens_params.len() * n_out];
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

        for (k, column) in projected.chunks(n_out).enumerate() {
            let mut unit = [0.0; 2];
            unit[k] = 1.0;
            seed_param_tangent(&mut dp, &sens_params, &unit);
            let (mut sens, mut jac) = (vec![0.0; n_out], vec![0.0; n_out]);
            set.sens_action(&mut scratch, 0.0, &y, &inputs, &dp, &mut sens);
            let direction = &y_sens[k * 2..(k + 1) * 2];
            set.jac_action(&mut scratch, 0.0, &y, &inputs, direction, &mut jac);
            let want: Vec<u64> = sens
                .iter()
                .zip(&jac)
                .map(|(a, b)| (a + b).to_bits())
                .collect();
            let got: Vec<u64> = column.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, want, "column {k}");
        }
    }

    /// The tangent tapes a family once compiled per item, each with its item's length.
    struct PerItemTapes(Vec<(SplitCompiledExpr, SplitCompiledExpr, usize)>);

    impl PerItemTapes {
        fn compile(arena: &Arena, roots: &[NodeId]) -> Self {
            Self(
                roots
                    .iter()
                    .map(|&root| {
                        let len = CompiledExpr::new(DagRef::new(arena, root)).output_len();
                        let sens = compile_tangent(arena, root, tangent_wrt_params);
                        (sens, compile_tangent(arena, root, tangent_wrt_states), len)
                    })
                    .collect(),
            )
        }

        /// Each item's `tangent` action, concatenated.
        fn action(&self, wrt_states: bool, y: &[f64], inputs: &[f64], v: &[f64]) -> Vec<u64> {
            let tangent = if wrt_states {
                TangentInputs {
                    dy: Some(v),
                    dp: None,
                }
            } else {
                TangentInputs {
                    dy: None,
                    dp: Some(v),
                }
            };
            let mut out = Vec::new();
            for (sens, jac, len) in &self.0 {
                let tape = if wrt_states { jac } else { sens };
                let mut scratch = vec![0.0; tape.scratch_len()];
                let result = tape.eval_with_tangent(&mut scratch, 0.5, y, &[], inputs, &tangent);
                out.extend(result[..*len].iter().map(|v| v.to_bits()));
            }
            out
        }

        /// The projection as one primal pass per item tape and one sweep per column computed it.
        fn sens_project(
            &self,
            y: &[f64],
            inputs: &[f64],
            sens_params: &[ParamIdx],
            y_sens: &[f64],
        ) -> Vec<u64> {
            let (n_states, n_out) = (y.len(), self.0.iter().map(|item| item.2).sum::<usize>());
            let mut out = vec![0.0; sens_params.len() * n_out];
            let mut dp = vec![0.0; inputs.len()];
            let mut offset = 0;
            for (sens, jac, len) in &self.0 {
                let mut tape = vec![0.0; sens.scratch_len().max(jac.scratch_len())];
                let rows = |k: usize| k * n_out + offset..k * n_out + offset + len;
                let mut primal = sens.eval_primal(&mut tape, 0.5, y, &[], inputs);
                for k in 0..sens_params.len() {
                    seed_param_tangent_unit(&mut dp, sens_params, SensIdx::new(k));
                    let tangent = TangentInputs {
                        dy: None,
                        dp: Some(&dp),
                    };
                    out[rows(k)].copy_from_slice(&primal.eval_tangent(&tangent)[..*len]);
                }
                let mut primal = jac.eval_primal(&mut tape, 0.5, y, &[], inputs);
                for k in 0..sens_params.len() {
                    let tangent = TangentInputs {
                        dy: Some(&y_sens[k * n_states..(k + 1) * n_states]),
                        dp: None,
                    };
                    let result = primal.eval_tangent(&tangent);
                    for (dst, &term) in out[rows(k)].iter_mut().zip(result) {
                        *dst += term;
                    }
                }
                offset += len;
            }
            out.iter().map(|v| v.to_bits()).collect()
        }
    }

    /// The family's one tangent-tape pair reproduces per-item tapes bit for bit, its batched
    /// `dH/dy` walks included, on a scratch left dirty by earlier calls.
    fn assert_family_matches_per_item_tapes(
        arena: &mut Arena,
        roots: &[NodeId],
        y: &[f64],
        inputs: &[f64],
    ) {
        let reference = PerItemTapes::compile(arena, roots);
        let set = ObservableSet::compile(arena, roots, true);
        let first_walks = [1, 3, 6].map(|columns| set.tangents().jac_lanes.lanes(columns));
        assert_eq!(first_walks, [1, 4, 8], "the family's dH/dy must batch");
        let n_out = set.total_len();
        let mut scratch = set.create_scratch();
        let dirty = |scratch: &mut ObservableScratch| {
            scratch.tape.fill(f64::NAN);
            scratch.tangent_lanes.fill(f64::NAN);
        };
        let bits = |values: &[f64]| -> Vec<u64> { values.iter().map(|v| v.to_bits()).collect() };
        let v: Vec<f64> = (0..y.len()).map(|i| 0.25 * i as f64 - 0.6).collect();
        let mut out = vec![f64::NAN; n_out];
        dirty(&mut scratch);
        set.jac_action(&mut scratch, 0.5, y, inputs, &v, &mut out);
        assert_eq!(bits(&out), reference.action(true, y, inputs, &v));
        let dp: Vec<f64> = (0..inputs.len()).map(|i| 0.5 - 0.3 * i as f64).collect();
        dirty(&mut scratch);
        set.sens_action(&mut scratch, 0.5, y, inputs, &dp, &mut out);
        assert_eq!(bits(&out), reference.action(false, y, inputs, &dp));

        // One scalar sweep, then padded 4- and 8-lane walks, then an 8-lane walk and a scalar.
        for n_sens in [1, 3, 6, 9] {
            let sens_params: Vec<ParamIdx> = (0..n_sens)
                .map(|k| ParamIdx::new(u32::try_from(k % inputs.len()).expect("few inputs")))
                .collect();
            let y_sens: Vec<f64> = (0..n_sens * y.len())
                .map(|k| ((k * 37) % 23) as f64 / 23.0 - 0.4)
                .collect();
            let mut projected = vec![f64::NAN; n_sens * n_out];
            let mut dp = vec![0.0; inputs.len()];
            dirty(&mut scratch);
            set.sens_project(
                &mut scratch,
                &mut dp,
                0.5,
                y,
                inputs,
                &sens_params,
                &y_sens,
                y.len(),
                &mut projected,
            );
            let want = reference.sens_project(y, inputs, &sens_params, &y_sens);
            assert_eq!(bits(&projected), want, "{n_sens} sensitivity columns");
        }
    }

    #[test]
    fn a_family_tangent_pair_matches_per_item_tapes_bitwise() {
        let n = 6;
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: n });
        let [p0, p1, p2] = [0, 1, 2].map(|index| {
            arena.alloc(Node::InputParameter {
                name: format!("p{index}"),
                index,
                offset: index,
                width: 1,
            })
        });
        let p0_y = arena.alloc(Node::Mul(p0, y));
        let p1_y = arena.alloc(Node::Mul(p1, y));
        let sin = arena.alloc(Node::Sin(p0_y));
        let cos = arena.alloc(Node::Cos(p1_y));
        let shared = arena.alloc(Node::Add(sin, cos));
        let weighted = arena.alloc(Node::Mul(shared, p2));
        let ones = arena.alloc(Node::SparseMatrix(Box::new(
            crate::node::CsrData::try_new(
                vec![0, n],
                (0..n).collect(),
                vec![1.0; n],
                crate::node::Shape::matrix(1, n),
            )
            .expect("valid row matrix"),
        )));
        let total = arena.alloc(Node::MatMul(ones, weighted));
        let middle = arena.alloc(Node::Index {
            child: shared,
            start: 1,
            end: 4,
        });
        let scaled_middle = arena.alloc(Node::Mul(middle, p0));
        // States only, parameters only, a signed-zero literal, and a scalar-broadcast `Hypot`.
        let slice = arena.alloc(Node::StateVector { start: 2, end: 5 });
        let product = arena.alloc(Node::Mul(p1, p2));
        let zero = arena.alloc(Node::Scalar(-0.0));
        let last = arena.alloc(Node::Index {
            child: shared,
            start: n - 1,
            end: n,
        });
        let negated = arena.alloc(Node::Neg(last));
        let hypot = arena.alloc(Node::Hypot(negated, p0));
        let roots = [total, scaled_middle, slice, product, zero, hypot];

        let y_values: Vec<f64> = (0..n).map(|i| 0.3f64.mul_add(i as f64, -0.7)).collect();
        assert_family_matches_per_item_tapes(&mut arena, &roots, &y_values, &[0.8, -1.3, 2.1]);
    }

    /// Each real fixture's rhs `f` under four parameters, with outputs sharing its cone.
    #[cfg(feature = "serialize")]
    #[test]
    fn a_real_model_family_matches_per_item_tapes_bitwise() {
        use crate::snapshot::DagSnapshot;

        let fixtures: [&[u8]; 2] = [
            include_bytes!("../benches/fixtures/spme.bin"),
            include_bytes!("../benches/fixtures/dfn.bin"),
        ];
        for bytes in fixtures {
            let snapshot = DagSnapshot::from_bytes(bytes).expect("fixture decodes");
            let n = snapshot.n_states;
            let mut arena = snapshot.dag().arena().clone();
            let f = snapshot.dag().root();
            let params: Vec<NodeId> = (0..4)
                .map(|index| {
                    arena.alloc(Node::InputParameter {
                        name: format!("p{index}"),
                        index,
                        offset: index,
                        width: 1,
                    })
                })
                .collect();
            let scaled = arena.alloc(Node::Mul(params[0], f));
            let sin = arena.alloc(Node::Sin(f));
            let sin_term = arena.alloc(Node::Mul(params[1], sin));
            let shifted = arena.alloc(Node::Add(scaled, sin_term));
            let rhs = arena.alloc(Node::Add(shifted, params[3]));
            let head = arena.alloc(Node::Index {
                child: rhs,
                start: 0,
                end: n / 3,
            });
            let scaled_head = arena.alloc(Node::Mul(head, params[2]));
            let tail = arena.alloc(Node::Index {
                child: rhs,
                start: n / 3,
                end: n,
            });
            let sin_tail = arena.alloc(Node::Sin(tail));
            let last = arena.alloc(Node::Index {
                child: rhs,
                start: n - 1,
                end: n,
            });
            let hypot = arena.alloc(Node::Hypot(last, params[0]));
            let roots = [scaled_head, sin_tail, hypot];

            let y: Vec<f64> = (0..n)
                .map(|i| 0.01f64.mul_add(i as f64 / n as f64, 0.5))
                .collect();
            assert_family_matches_per_item_tapes(&mut arena, &roots, &y, &[0.3, 0.4, 0.5, 0.6]);
        }
    }

    #[test]
    fn tangent_tapes_compile_only_when_asked_for() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let without = ObservableSet::compile(&mut arena, &[y], false);
        let with = ObservableSet::compile(&mut arena, &[y], true);
        assert!(without.tangents.is_none());
        assert!(with.tangents.is_some());
    }

    #[test]
    #[should_panic(expected = "compiled only with sensitivities")]
    fn a_tangent_action_without_tangent_tapes_is_refused() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let set = ObservableSet::compile(&mut arena, &[y], false);
        let mut out = [0.0; 2];
        set.jac_action(
            &mut set.create_scratch(),
            0.0,
            &[1.0, 2.0],
            &[],
            &[1.0, 0.0],
            &mut out,
        );
    }

    #[test]
    fn eval_batch_reads_only_the_states_it_is_told_of() {
        // Trailing values past `k * n_states` used to shift the batched path's lanes.
        let (_arena, set) = build_pair();
        let mut scratch = set.create_scratch();
        let (ts, inputs) = ([0.0, 1.0], [7.0]);
        let exact = [3.0, 5.0, 1.0, 2.0];
        let padded = [3.0, 5.0, 1.0, 2.0, -4.0, 0.5];
        let (mut want, mut got) = ([0.0; 4], [0.0; 4]);
        set.eval_batch(&mut scratch, 2, &ts, &exact, 2, &inputs, &mut want);
        set.eval_batch(&mut scratch, 2, &ts, &padded, 2, &inputs, &mut got);
        assert_eq!(got.map(f64::to_bits), want.map(f64::to_bits));
    }

    #[test]
    fn eval_batch_matches_a_loop_of_eval_all() {
        let (_arena, set) = build_pair();
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
