//! Batched tangent sweeps: one walk of the tangent tape for `K` colour seeds.
//!
//! Primal slots are shared across lanes and tangent slots are lane-minor; each lane
//! accumulates in the scalar sweep's order, so results are bitwise identical.

use crate::eval::{
    SplitCompiledExpr, csr_rows, split_read_write, split_two_read_write, unary_apply,
    with_binary_op, with_unary_op,
};
use crate::eval_batch::broadcast_lanes;
use crate::ir::ids::BufSlot;
use crate::ir::{BinaryOp, BroadcastKind, ConstPool, Instruction, TypedIr, UnaryOp};

/// Lane counts the batched sweep is instantiated for, widest first.
const SUPPORTED_LANES: [usize; 2] = [8, 4];

/// The widest lane count [`run_tangent_batch`] supports.
pub(crate) const MAX_LANES: usize = SUPPORTED_LANES[0];

/// Scratch ceiling for one batched tangent region.
const LANE_SCRATCH_BUDGET: usize = 32 << 20;

/// How many lanes each walk of one tangent tape carries, chosen once from the tape's shape.
///
/// Per-lane arithmetic is width-independent, so a plan moves no bits, only time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LanePlan {
    /// Per [`SUPPORTED_LANES`] entry, the fewest seeds left at which a walk that wide pays.
    min_seeds: [Option<usize>; 2],
}

impl LanePlan {
    /// One scalar walk per seed.
    pub(crate) const SCALAR: Self = Self {
        min_seeds: [None; 2],
    };

    /// The cheapest plan for `expr` by [`walk_costs`]; scalar when it is not batchable.
    pub(crate) fn new(expr: &SplitCompiledExpr) -> Self {
        if !is_batchable(expr) {
            return Self::SCALAR;
        }
        let (scalar, batched) = walk_costs(expr);
        let fits =
            |lanes| tangent_scratch_len(expr, lanes) * size_of::<f64>() <= LANE_SCRATCH_BUDGET;
        let mut wide = [None; 2];
        for ((slot, cost), lanes) in wide.iter_mut().zip(batched).zip(SUPPORTED_LANES) {
            *slot = fits(lanes).then_some(cost);
        }
        Self::from_costs(scalar, wide)
    }

    /// The plan for a scalar walk costing `scalar` and, per [`SUPPORTED_LANES`] entry, a walk
    /// that wide costing `wide` (`None` where it may not run).
    fn from_costs(scalar: usize, wide: [Option<usize>; 2]) -> Self {
        let mut plan = Self::SCALAR;
        // Narrowest first, so each width is weighed against the plan of the narrower ones.
        for (i, &lanes) in SUPPORTED_LANES.iter().enumerate().rev() {
            let Some(cost) = wide[i] else { continue };
            plan.min_seeds[i] = (1..=lanes).find(|&seeds| cost < plan.cost(seeds, scalar, wide));
        }
        plan
    }

    /// Lanes for the next walk with `seeds` seeds left: 1, or a [`SUPPORTED_LANES`] entry.
    ///
    /// Never fewer than for fewer seeds, so a scratch sized for the first walk fits every walk.
    pub(crate) fn lanes(self, seeds: usize) -> usize {
        self.width_index(seeds).map_or(1, |i| SUPPORTED_LANES[i])
    }

    /// The most lanes any walk carries.
    pub(crate) fn widest(self) -> usize {
        self.lanes(usize::MAX)
    }

    fn width_index(self, seeds: usize) -> Option<usize> {
        self.min_seeds
            .iter()
            .position(|min| min.is_some_and(|min| seeds >= min))
    }

    /// What walking `seeds` seeds by this plan costs.
    fn cost(self, mut seeds: usize, scalar: usize, wide: [Option<usize>; 2]) -> usize {
        let mut total = 0;
        while seeds > 0 {
            let walk = self.width_index(seeds);
            total += walk.and_then(|i| wide[i]).unwrap_or(scalar);
            seeds -= walk.map_or(1, |i| SUPPORTED_LANES[i]).min(seeds);
        }
        total
    }
}

/// Estimated cost of one walk of `expr`'s tangent section, scalar and at each
/// [`SUPPORTED_LANES`] entry, in scalar element-wise operations (fitted on Apple M4).
///
/// A batched walk dispatches each instruction once for all its lanes and runs a sparse
/// product's latency-bound row sums side by side, but repeats element-wise work per lane.
fn walk_costs(expr: &SplitCompiledExpr) -> (usize, [usize; 2]) {
    /// Per instruction, a scalar and a batched walk's dispatch.
    const DISPATCH: (usize, usize) = (15, 32);
    /// Per stored matrix entry of a sparse product, a scalar and a batched walk's row sums.
    const ENTRY: (usize, usize) = (5, 7);
    /// Per element of an element-wise instruction, a walk at each [`SUPPORTED_LANES`] entry.
    const ELEMENT: [usize; 2] = [7, 4];
    /// Lane-minor values past which an operand outgrows L1, costing half again per element.
    const CACHED_VALUES: usize = 2048;
    let (ir, split) = (expr.ir(), expr.split_info());
    let consts = ir.consts();
    let mut scalar = 0;
    let mut batched = [0; 2];
    for instr in &ir.instructions()[split.primal_end..] {
        let (elements, entries) = match *instr {
            Instruction::LoadTangentState { width: len, .. }
            | Instruction::Binary { len, .. }
            | Instruction::Unary { len, .. }
            | Instruction::Index { len, .. } => (len.index(), 0),
            Instruction::Concat {
                sources_idx,
                sources_len,
                ..
            } => {
                let sources = &consts.concat_sources[sources_idx.index()..][..sources_len.index()];
                (sources.iter().map(|slot| slot.len_usize()).sum(), 0)
            },
            Instruction::MatMul { csr_idx, .. } => (0, consts.csr_data[csr_idx.index()].data.len()),
            _ => (0, 0),
        };
        scalar += DISPATCH.0 + elements + ENTRY.0 * entries;
        for ((cost, lanes), per_element) in batched.iter_mut().zip(SUPPORTED_LANES).zip(ELEMENT) {
            let mut work = elements * per_element;
            if elements * lanes > CACHED_VALUES {
                work += work / 2;
            }
            *cost += DISPATCH.1 + work + ENTRY.1 * entries;
        }
    }
    (scalar, batched)
}

#[derive(Clone, Copy)]
enum Operand {
    Primal(usize),
    /// Lane-minor tangent index, already scaled by `K`.
    Tangent(usize),
}

#[inline]
const fn operand<const K: usize>(offset: usize, primal_len: usize) -> Operand {
    if offset < primal_len {
        Operand::Primal(offset)
    } else {
        Operand::Tangent((offset - primal_len) * K)
    }
}

/// Whether every instruction in the tangent section has a batched form.
///
/// Only a `Concat` may read nothing but primal slots; any other reader of only primal
/// slots would be seed-independent, so a tape holding one sweeps scalar instead.
pub fn is_batchable(expr: &SplitCompiledExpr) -> bool {
    let split = expr.split_info();
    let tangent = |slot: BufSlot| slot.index() >= split.primal_buffer_size;
    expr.ir().instructions()[split.primal_end..]
        .iter()
        .all(|instr| match *instr {
            Instruction::LoadTangentState { .. } | Instruction::Concat { .. } => true,
            Instruction::Unary { src, .. } | Instruction::Index { src, .. } => tangent(src),
            Instruction::MatMul { vec_src, .. } => tangent(vec_src),
            Instruction::Binary { a, b, .. } => tangent(a) || tangent(b),
            _ => false,
        })
}

/// Whether the root is primal, i.e. the tangent folded to a seed-independent constant.
const fn root_is_primal(ir: &TypedIr, primal_len: usize) -> bool {
    ir.root_slot().offset_usize() < primal_len
}

/// Scratch length the tangent region needs for `lanes` lanes.
///
/// A primal root gets an extra tail that the sweep broadcasts it into.
pub const fn tangent_scratch_len(expr: &SplitCompiledExpr, lanes: usize) -> usize {
    let (ir, primal_len) = (expr.ir(), expr.split_info().primal_buffer_size);
    let spill = if root_is_primal(ir, primal_len) {
        ir.root_slot().len_usize() * lanes
    } else {
        0
    };
    (ir.buffer_size() - primal_len) * lanes + spill
}

/// Run the tangent section of `expr` for `lanes` seed vectors at once.
///
/// `primal` holds the primal section's values; `seeds` is `(n_states, lanes)` lane-minor and
/// the result `(out_len, lanes)` lane-minor. `lanes` must be 8 or 4, and [`is_batchable`] must
/// hold.
pub fn run_tangent_batch<'t>(
    expr: &SplitCompiledExpr,
    lanes: usize,
    primal: &[f64],
    tan: &'t mut [f64],
    seeds: &[f64],
) -> &'t [f64] {
    match lanes {
        8 => sweep::<8>(expr, primal, tan, seeds),
        4 => sweep::<4>(expr, primal, tan, seeds),
        other => panic!("no batched tangent sweep has {other} lanes"),
    }
}

fn sweep<'t, const K: usize>(
    expr: &SplitCompiledExpr,
    primal: &[f64],
    tan: &'t mut [f64],
    seeds: &[f64],
) -> &'t [f64] {
    let (ir, split) = (expr.ir(), expr.split_info());
    let primal_len = split.primal_buffer_size;
    let consts = ir.consts();

    for instr in &ir.instructions()[split.primal_end..] {
        exec::<K>(*instr, primal, tan, seeds, primal_len, consts);
    }

    let root = ir.root_slot();
    if root_is_primal(ir, primal_len) {
        let spill = (ir.buffer_size() - primal_len) * K;
        broadcast_lanes(&mut tan[spill..], K, &primal[root.bounds()]);
        return &tan[spill..spill + root.len_usize() * K];
    }
    let base = (root.offset_usize() - primal_len) * K;
    &tan[base..base + root.len_usize() * K]
}

fn exec<const K: usize>(
    instr: Instruction,
    primal: &[f64],
    tan: &mut [f64],
    seeds: &[f64],
    primal_len: usize,
    consts: &ConstPool,
) {
    match instr {
        Instruction::LoadTangentState { start, width, dst } => {
            let dst = tangent_base::<K>(dst.index(), primal_len);
            let len = width.index() * K;
            tan[dst..dst + len].copy_from_slice(&seeds[start.index() * K..][..len]);
        },

        Instruction::Index {
            src,
            start,
            dst,
            len,
        } => {
            let src = tangent_base::<K>(src.offset_by(start), primal_len);
            let dst = tangent_base::<K>(dst.index(), primal_len);
            tan.copy_within(src..src + len.index() * K, dst);
        },

        Instruction::Concat {
            sources_idx,
            sources_len,
            dst,
        } => {
            let mut write = tangent_base::<K>(dst.index(), primal_len);
            for i in 0..sources_len.index() {
                let slot = consts.concat_sources[sources_idx.index() + i];
                let len = slot.len_usize();
                match operand::<K>(slot.offset_usize(), primal_len) {
                    Operand::Tangent(src) => tan.copy_within(src..src + len * K, write),
                    Operand::Primal(src) => {
                        broadcast_lanes(&mut tan[write..], K, &primal[src..src + len]);
                    },
                }
                write += len * K;
            }
        },

        Instruction::Unary { op, src, dst, len } => unary::<K>(
            op,
            tangent_base::<K>(src.index(), primal_len),
            tangent_base::<K>(dst.index(), primal_len),
            len.index(),
            tan,
        ),

        Instruction::Binary {
            op,
            a,
            b,
            dst,
            len,
            kind,
        } => binary_op::<K>(
            op,
            operand::<K>(a.index(), primal_len),
            operand::<K>(b.index(), primal_len),
            tangent_base::<K>(dst.index(), primal_len),
            len.index(),
            kind,
            primal,
            tan,
        ),

        Instruction::MatMul {
            csr_idx,
            vec_src,
            dst,
        } => {
            let csr = &consts.csr_data[csr_idx.index()];
            let (x, out) = split_read_write(
                tan,
                tangent_base::<K>(vec_src.index(), primal_len),
                csr.shape.cols * K,
                tangent_base::<K>(dst.index(), primal_len),
                csr.shape.rows * K,
            );
            for (out, (cols, vals)) in out.chunks_exact_mut(K).zip(csr_rows(csr)) {
                let mut acc = [0.0_f64; K];
                for (&col, &value) in cols.iter().zip(vals) {
                    for (a, &v) in acc.iter_mut().zip(&x[col * K..][..K]) {
                        *a += value * v;
                    }
                }
                out.copy_from_slice(&acc);
            }
        },

        other => unreachable!("instruction {other:?} is not batchable; check is_batchable first"),
    }
}

/// The lane-minor index of tangent slot `offset`.
#[inline]
fn tangent_base<const K: usize>(offset: usize, primal_len: usize) -> usize {
    debug_assert!(
        offset >= primal_len,
        "is_batchable admits only tangent slots here"
    );
    (offset - primal_len) * K
}

#[allow(clippy::too_many_arguments)]
fn binary_op<const K: usize>(
    op: BinaryOp,
    a: Operand,
    b: Operand,
    dst: usize,
    len: usize,
    kind: BroadcastKind,
    primal: &[f64],
    tan: &mut [f64],
) {
    with_binary_op!(op, f => binary::<K, _>(f, a, b, dst, len, kind, primal, tan));
}

fn unary<const K: usize>(op: UnaryOp, src: usize, dst: usize, len: usize, tan: &mut [f64]) {
    // Lane-minor, an element-wise op over `len` elements is one over `len * K` values.
    with_unary_op!(op, f => unary_apply(tan, f, src, dst, len * K));
}

/// `out[e] = f(a(e), b(e))` lane by lane, for each element `e` of `out`.
// Like `eval`'s op helpers, it must inline so the operand closures fuse into one loop.
#[allow(clippy::inline_always)]
#[inline(always)]
fn zip_lanes<const K: usize>(
    out: &mut [[f64; K]],
    f: &impl Fn(f64, f64) -> f64,
    a: impl Fn(usize) -> [f64; K],
    b: impl Fn(usize) -> [f64; K],
) {
    for (e, out) in out.iter_mut().enumerate() {
        for ((o, x), y) in out.iter_mut().zip(a(e)).zip(b(e)) {
            *o = f(x, y);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn binary<const K: usize, F: Fn(f64, f64) -> f64>(
    f: F,
    a: Operand,
    b: Operand,
    dst: usize,
    len: usize,
    kind: BroadcastKind,
    primal: &[f64],
    tan: &mut [f64],
) {
    use BroadcastKind::{ScalarScalar, ScalarVector, VectorScalar, VectorVector};
    if len == 1 {
        // Splitting windows costs more than one element's op, which dominates scalar-row tapes.
        let mut out = [0.0_f64; K];
        match (a, b) {
            (Operand::Tangent(a), Operand::Tangent(b)) => {
                for ((o, &x), &y) in out.iter_mut().zip(&tan[a..][..K]).zip(&tan[b..][..K]) {
                    *o = f(x, y);
                }
            },
            (Operand::Primal(a), Operand::Tangent(b)) => {
                let x = primal[a];
                for (o, &y) in out.iter_mut().zip(&tan[b..][..K]) {
                    *o = f(x, y);
                }
            },
            (Operand::Tangent(a), Operand::Primal(b)) => {
                let y = primal[b];
                for (o, &x) in out.iter_mut().zip(&tan[a..][..K]) {
                    *o = f(x, y);
                }
            },
            (Operand::Primal(_), Operand::Primal(_)) => {
                unreachable!("is_batchable admits no binary over two primal operands")
            },
        }
        tan[dst..][..K].copy_from_slice(&out);
        return;
    }
    // Lane-minor, a vector operand is `len` chunks of `K` lanes and a broadcast one a single chunk.
    let a_len = if matches!(kind, VectorScalar | VectorVector) {
        len
    } else {
        1
    };
    let b_len = if matches!(kind, ScalarVector | VectorVector) {
        len
    } else {
        1
    };
    match (a, b) {
        (Operand::Tangent(a), Operand::Tangent(b)) => {
            let (x, y, out) =
                split_two_read_write(tan, (a, a_len * K), (b, b_len * K), (dst, len * K));
            let (x, y, out) = (
                x.as_chunks::<K>().0,
                y.as_chunks::<K>().0,
                out.as_chunks_mut::<K>().0,
            );
            match kind {
                ScalarScalar | VectorVector => zip_lanes(out, &f, |e| x[e], |e| y[e]),
                ScalarVector => zip_lanes(out, &f, |_| x[0], |e| y[e]),
                VectorScalar => zip_lanes(out, &f, |e| x[e], |_| y[0]),
            }
        },
        (Operand::Primal(a), Operand::Tangent(b)) => {
            let (y, out) = split_read_write(tan, b, b_len * K, dst, len * K);
            let (y, out) = (y.as_chunks::<K>().0, out.as_chunks_mut::<K>().0);
            let x = &primal[a..a + a_len];
            match kind {
                ScalarScalar | VectorVector => zip_lanes(out, &f, |e| [x[e]; K], |e| y[e]),
                ScalarVector => zip_lanes(out, &f, |_| [x[0]; K], |e| y[e]),
                VectorScalar => zip_lanes(out, &f, |e| [x[e]; K], |_| y[0]),
            }
        },
        (Operand::Tangent(a), Operand::Primal(b)) => {
            let (x, out) = split_read_write(tan, a, a_len * K, dst, len * K);
            let (x, out) = (x.as_chunks::<K>().0, out.as_chunks_mut::<K>().0);
            let y = &primal[b..b + b_len];
            match kind {
                ScalarScalar | VectorVector => zip_lanes(out, &f, |e| x[e], |e| [y[e]; K]),
                ScalarVector => zip_lanes(out, &f, |_| x[0], |e| [y[e]; K]),
                VectorScalar => zip_lanes(out, &f, |e| x[e], |_| [y[0]; K]),
            }
        },
        (Operand::Primal(_), Operand::Primal(_)) => {
            unreachable!("is_batchable admits no binary over two primal operands")
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lanes of each walk that `seeds` seeds take under `plan`.
    fn walks(plan: LanePlan, seeds: usize) -> Vec<usize> {
        let mut left = seeds;
        std::iter::from_fn(|| {
            let lanes = (left > 0).then(|| plan.lanes(left))?;
            left -= lanes.min(left);
            Some(lanes)
        })
        .collect()
    }

    #[test]
    fn a_wider_walk_runs_only_where_it_is_cheaper() {
        // (scalar, 8-lane, 4-lane) walk costs, then the walks for 2, 3, 5, 7, 9 and 11 seeds.
        type Case = (usize, [Option<usize>; 2], [&'static [usize]; 6]);
        let cases: [Case; 4] = [
            // Dispatch-bound: a part-empty walk beats three or more scalar ones.
            (
                100,
                [Some(260), Some(230)],
                [&[1, 1], &[4], &[8], &[8], &[8, 1], &[8, 4]],
            ),
            // Long element-wise vectors: even a full walk loses.
            (
                100,
                [Some(1000), Some(420)],
                [&[1, 1], &[1; 3], &[1; 5], &[1; 7], &[1; 9], &[1; 11]],
            ),
            // In between: only (nearly) full walks pay.
            (
                100,
                [Some(640), Some(350)],
                [&[1, 1], &[1; 3], &[4, 1], &[8], &[8, 1], &[8, 1, 1, 1]],
            ),
            // 8 lanes over the scratch budget.
            (
                100,
                [None, Some(230)],
                [&[1, 1], &[4], &[4, 1], &[4, 4], &[4, 4, 1], &[4, 4, 4]],
            ),
        ];
        for (scalar, wide, plans) in cases {
            let plan = LanePlan::from_costs(scalar, wide);
            for (seeds, want) in [2, 3, 5, 7, 9, 11].into_iter().zip(plans) {
                assert_eq!(
                    walks(plan, seeds),
                    want,
                    "{seeds} seeds at costs {scalar}, {wide:?}"
                );
            }
            let lanes: Vec<usize> = (1..=20).map(|seeds| plan.lanes(seeds)).collect();
            assert!(
                lanes.is_sorted(),
                "lanes must not shrink with more seeds: {lanes:?}"
            );
        }
    }
}
