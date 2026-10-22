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
        if !writes_nothing(instr, consts) {
            exec::<K>(*instr, primal, tan, seeds, primal_len, consts);
        }
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

/// Whether `instr` writes no element: its zero-width slot sits at offset 0, in the primal region.
fn writes_nothing(instr: &Instruction, consts: &ConstPool) -> bool {
    match *instr {
        Instruction::LoadTangentState { width: len, .. }
        | Instruction::Index { len, .. }
        | Instruction::Unary { len, .. }
        | Instruction::Binary { len, .. } => len.index() == 0,
        Instruction::Concat {
            sources_idx,
            sources_len,
            ..
        } => consts.concat_sources[sources_idx.index()..][..sources_len.index()]
            .iter()
            .all(|slot| slot.len_usize() == 0),
        Instruction::MatMul { csr_idx, .. } => consts.csr_data[csr_idx.index()].shape.rows == 0,
        _ => false,
    }
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
            for (out, (cols, vals)) in out.as_chunks_mut::<K>().0.iter_mut().zip(csr_rows(csr)) {
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
