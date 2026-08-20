//! Batched tangent sweeps: one walk of the tangent tape for `K` colour seeds.
//!
//! Coloured Jacobian assembly runs the tangent section once per colour, and on
//! larger models nearly all of that time sits in the CSR gather inside
//! [`Instruction::MatMul`] — a scattered read whose cache line serves a single
//! lane. Widening the tangent region to `K` lanes per element turns that gather
//! into one contiguous `K`-wide load reused by every lane, so the operator is
//! streamed once per block instead of once per colour.
//!
//! The split-eval layout makes the split cheap to exploit: primal slots occupy
//! `[0, primal_buffer_size)` and are shared across lanes unchanged, while
//! tangent slots live above it and carry one value per lane. Each lane
//! accumulates in the same order as the scalar sweep, so results are bitwise
//! identical to [`CompiledExpr::run_tangent_section`].
//!
//! [`CompiledExpr::run_tangent_section`]: crate::eval::CompiledExpr

use crate::eval::{EQUALITY_EPS, erf_approx, sign};
use crate::ir::{BinaryOp, BroadcastKind, ConstPool, Instruction, TypedIr, UnaryOp};

/// Lane counts the batched sweep is instantiated for, widest first.
pub const SUPPORTED_LANES: [usize; 2] = [8, 4];

/// An operand's home: primal slots are shared across lanes, tangent slots are
/// lane-minor.
#[derive(Clone, Copy)]
enum Operand {
    /// Base index into the primal region.
    Primal(usize),
    /// Base index into the lane-minor tangent region, already scaled by `K`.
    Tangent(usize),
}

/// Classify a slot offset against the primal/tangent boundary.
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
/// Tapes carrying anything else (reductions, interpolants, branch dispatch)
/// fall back to the per-colour scalar sweep.
pub fn is_batchable(ir: &TypedIr) -> bool {
    let Some(split) = ir.split_eval_info() else {
        return false;
    };
    ir.instructions()[split.primal_end..].iter().all(|instr| {
        matches!(
            instr,
            Instruction::LoadTangentState { .. }
                | Instruction::LoadScalar { .. }
                | Instruction::LoadArray { .. }
                | Instruction::FillZero { .. }
                | Instruction::Binary { .. }
                | Instruction::Unary { .. }
                | Instruction::Index { .. }
                | Instruction::Concat { .. }
                | Instruction::MatMul { .. }
        )
    })
}

/// Whether the root slot sits in the primal pool, which happens when the
/// tangent folds to a constant and so depends on no seed.
const fn root_is_primal(ir: &TypedIr, primal_len: usize) -> bool {
    ir.root_slot().offset_usize() < primal_len
}

/// Scratch length the tangent region needs for `lanes` lanes.
///
/// A primal root has no lane-minor home of its own, so the buffer carries a
/// tail the sweep broadcasts it into, keeping one return shape for callers.
pub fn tangent_scratch_len(ir: &TypedIr, lanes: usize) -> usize {
    ir.split_eval_info().map_or(0, |s| {
        let tangent = (ir.buffer_size() - s.primal_buffer_size) * lanes;
        let spill = if root_is_primal(ir, s.primal_buffer_size) {
            ir.root_slot().len_usize() * lanes
        } else {
            0
        };
        tangent + spill
    })
}

/// Run the tangent section for `K` seed vectors at once.
///
/// `primal` is the primal region a prior primal sweep filled, `tan` the
/// lane-minor tangent region, and `seeds` the `(n_states, K)` lane-minor seed
/// matrix. Returns the root slot as `(out_len, K)` lane-minor.
///
/// # Panics
///
/// Panics if the tape is not split-eval or carries an instruction outside the
/// batchable set; call [`is_batchable`] first.
pub fn run_tangent_batch<'t, const K: usize>(
    ir: &TypedIr,
    primal: &[f64],
    tan: &'t mut [f64],
    seeds: &[f64],
) -> &'t [f64] {
    let split = ir
        .split_eval_info()
        .expect("run_tangent_batch requires a split-eval IR");
    let primal_len = split.primal_buffer_size;
    let consts = ir.consts();

    for instr in &ir.instructions()[split.primal_end..] {
        exec::<K>(*instr, primal, tan, seeds, primal_len, consts);
    }

    let root = ir.root_slot();
    if root_is_primal(ir, primal_len) {
        let spill = (ir.buffer_size() - primal_len) * K;
        for e in 0..root.len_usize() {
            tan[spill + e * K..spill + (e + 1) * K].fill(primal[root.offset_usize() + e]);
        }
        return &tan[spill..spill + root.len_usize() * K];
    }
    let base = (root.offset_usize() - primal_len) * K;
    &tan[base..base + root.len_usize() * K]
}

#[allow(clippy::too_many_lines)]
fn exec<const K: usize>(
    instr: Instruction,
    primal: &[f64],
    tan: &mut [f64],
    seeds: &[f64],
    primal_len: usize,
    consts: &ConstPool,
) {
    match instr {
        Instruction::LoadTangentState { start, end, dst } => {
            let dst = tangent_base::<K>(dst as usize, primal_len);
            let len = (end - start) as usize * K;
            tan[dst..dst + len].copy_from_slice(&seeds[start as usize * K..][..len]);
        },

        Instruction::LoadScalar { value, dst } => {
            let dst = tangent_base::<K>(dst as usize, primal_len);
            tan[dst..dst + K].fill(value);
        },

        Instruction::FillZero { dst, len } => {
            let dst = tangent_base::<K>(dst as usize, primal_len);
            tan[dst..dst + len as usize * K].fill(0.0);
        },

        Instruction::LoadArray { data_idx, len, dst } => {
            let src = consts.get_array(data_idx, len);
            let dst = tangent_base::<K>(dst as usize, primal_len);
            for (e, &value) in src.iter().enumerate() {
                tan[dst + e * K..dst + (e + 1) * K].fill(value);
            }
        },

        Instruction::Index {
            src,
            start,
            dst,
            len,
        } => {
            let dst = tangent_base::<K>(dst as usize, primal_len);
            let len = len as usize;
            match operand::<K>(src as usize + start as usize, primal_len) {
                Operand::Tangent(src) => tan.copy_within(src..src + len * K, dst),
                Operand::Primal(src) => {
                    for e in 0..len {
                        tan[dst + e * K..dst + (e + 1) * K].fill(primal[src + e]);
                    }
                },
            }
        },

        Instruction::Concat {
            sources_idx,
            sources_len,
            dst,
        } => {
            let mut write = tangent_base::<K>(dst as usize, primal_len);
            for i in 0..sources_len as usize {
                let (offset, len) = consts.concat_sources[sources_idx as usize + i];
                let len = len as usize;
                match operand::<K>(offset as usize, primal_len) {
                    Operand::Tangent(src) => tan.copy_within(src..src + len * K, write),
                    Operand::Primal(src) => {
                        for e in 0..len {
                            tan[write + e * K..write + (e + 1) * K].fill(primal[src + e]);
                        }
                    },
                }
                write += len * K;
            }
        },

        Instruction::Unary { op, src, dst, len } => unary::<K>(
            op,
            operand::<K>(src as usize, primal_len),
            tangent_base::<K>(dst as usize, primal_len),
            len as usize,
            primal,
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
            operand::<K>(a as usize, primal_len),
            operand::<K>(b as usize, primal_len),
            tangent_base::<K>(dst as usize, primal_len),
            len as usize,
            kind,
            primal,
            tan,
        ),

        Instruction::MatMul {
            csr_idx,
            vec_src,
            dst,
        } => {
            let csr = &consts.csr_data[csr_idx as usize];
            let dst = tangent_base::<K>(dst as usize, primal_len);
            match operand::<K>(vec_src as usize, primal_len) {
                Operand::Tangent(vec) => {
                    for row in 0..csr.shape.rows {
                        let span = csr.indptr[row]..csr.indptr[row + 1];
                        let mut acc = [0.0_f64; K];
                        for (&col, &value) in csr.indices[span.clone()].iter().zip(&csr.data[span])
                        {
                            let lanes = &tan[vec + col * K..][..K];
                            for (a, &v) in acc.iter_mut().zip(lanes) {
                                *a += value * v;
                            }
                        }
                        tan[dst + row * K..][..K].copy_from_slice(&acc);
                    }
                },
                Operand::Primal(vec) => {
                    for row in 0..csr.shape.rows {
                        let span = csr.indptr[row]..csr.indptr[row + 1];
                        let mut sum = 0.0;
                        for (&col, &value) in csr.indices[span.clone()].iter().zip(&csr.data[span])
                        {
                            sum += value * primal[vec + col];
                        }
                        tan[dst + row * K..][..K].fill(sum);
                    }
                },
            }
        },

        other => unreachable!("instruction {other:?} is not batchable; check is_batchable first"),
    }
}

/// Tangent-region base index for a slot that must live above the primal pool.
#[inline]
fn tangent_base<const K: usize>(offset: usize, primal_len: usize) -> usize {
    debug_assert!(
        offset >= primal_len,
        "tangent-section writes target the tangent pool"
    );
    (offset - primal_len) * K
}

/// Dispatch a binary op to a monomorphised [`binary`] loop, mirroring
/// `eval::eval_binary_op` so both paths share one definition per op.
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
    macro_rules! apply {
        ($f:expr) => {
            binary::<K, _>($f, a, b, dst, len, kind, primal, tan)
        };
    }
    match op {
        BinaryOp::Add => apply!(|x, y| x + y),
        BinaryOp::Sub => apply!(|x, y| x - y),
        BinaryOp::Mul => apply!(|x, y| x * y),
        BinaryOp::Div => apply!(|x, y| x / y),
        BinaryOp::Pow => apply!(f64::powf),
        BinaryOp::Minimum => apply!(f64::min),
        BinaryOp::Maximum => apply!(f64::max),
        BinaryOp::Modulo => apply!(|x, y| x % y),
        BinaryOp::Hypot => apply!(f64::hypot),
        BinaryOp::EqualHeaviside => apply!(|x, y| if x <= y { 1.0 } else { 0.0 }),
        BinaryOp::NotEqualHeaviside => apply!(|x, y| if x < y { 1.0 } else { 0.0 }),
        BinaryOp::Equality => {
            apply!(|x: f64, y: f64| if (x - y).abs() < EQUALITY_EPS {
                1.0
            } else {
                0.0
            });
        },
    }
}

/// Dispatch a unary op to a monomorphised [`unary_apply_lanes`] loop, mirroring
/// `eval::eval_unary_op`.
fn unary<const K: usize>(
    op: UnaryOp,
    src: Operand,
    dst: usize,
    len: usize,
    primal: &[f64],
    tan: &mut [f64],
) {
    macro_rules! apply {
        ($f:expr) => {
            unary_apply_lanes::<K, _>($f, src, dst, len, primal, tan)
        };
    }
    match op {
        UnaryOp::Neg => apply!(|x: f64| -x),
        UnaryOp::Abs => apply!(f64::abs),
        UnaryOp::Sqrt => apply!(f64::sqrt),
        UnaryOp::Exp => apply!(f64::exp),
        UnaryOp::Log => apply!(f64::ln),
        UnaryOp::Sin => apply!(f64::sin),
        UnaryOp::Cos => apply!(f64::cos),
        UnaryOp::Tanh => apply!(f64::tanh),
        UnaryOp::Sinh => apply!(f64::sinh),
        UnaryOp::Cosh => apply!(f64::cosh),
        UnaryOp::Arcsinh => apply!(f64::asinh),
        UnaryOp::Arctan => apply!(f64::atan),
        UnaryOp::Erf => apply!(erf_approx),
        UnaryOp::Sign => apply!(sign),
        UnaryOp::Floor => apply!(f64::floor),
        UnaryOp::Ceiling => apply!(f64::ceil),
    }
}

fn unary_apply_lanes<const K: usize, F: Fn(f64) -> f64>(
    f: F,
    src: Operand,
    dst: usize,
    len: usize,
    primal: &[f64],
    tan: &mut [f64],
) {
    match src {
        Operand::Tangent(src) => {
            for e in 0..len {
                let mut out = [0.0_f64; K];
                for (o, &x) in out.iter_mut().zip(&tan[src + e * K..][..K]) {
                    *o = f(x);
                }
                tan[dst + e * K..][..K].copy_from_slice(&out);
            }
        },
        Operand::Primal(src) => {
            for e in 0..len {
                tan[dst + e * K..][..K].fill(f(primal[src + e]));
            }
        },
    }
}

/// Element strides `(a, b)` for a broadcast kind: a broadcast operand holds
/// one element and never advances.
#[inline]
const fn broadcast_strides(kind: BroadcastKind) -> (usize, usize) {
    match kind {
        BroadcastKind::ScalarScalar => (0, 0),
        BroadcastKind::ScalarVector => (0, 1),
        BroadcastKind::VectorScalar => (1, 0),
        BroadcastKind::VectorVector => (1, 1),
    }
}

/// Every lane loop lands each element in a stack array before storing it.
/// Operand and destination slots are disjoint, but they share one `&mut [f64]`,
/// so reading and writing it in the same expression leaves the compiler unable
/// to rule out aliasing and it emits a scalar, ordered loop.
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
    let (sa, sb) = broadcast_strides(kind);
    match (a, b) {
        (Operand::Tangent(a), Operand::Tangent(b)) => {
            for e in 0..len {
                let (x, y) = (&tan[a + e * sa * K..][..K], &tan[b + e * sb * K..][..K]);
                let mut out = [0.0_f64; K];
                for (o, (&x, &y)) in out.iter_mut().zip(x.iter().zip(y)) {
                    *o = f(x, y);
                }
                tan[dst + e * K..][..K].copy_from_slice(&out);
            }
        },
        (Operand::Primal(a), Operand::Tangent(b)) => {
            for e in 0..len {
                let x = primal[a + e * sa];
                let mut out = [0.0_f64; K];
                for (o, &y) in out.iter_mut().zip(&tan[b + e * sb * K..][..K]) {
                    *o = f(x, y);
                }
                tan[dst + e * K..][..K].copy_from_slice(&out);
            }
        },
        (Operand::Tangent(a), Operand::Primal(b)) => {
            for e in 0..len {
                let y = primal[b + e * sb];
                let mut out = [0.0_f64; K];
                for (o, &x) in out.iter_mut().zip(&tan[a + e * sa * K..][..K]) {
                    *o = f(x, y);
                }
                tan[dst + e * K..][..K].copy_from_slice(&out);
            }
        },
        (Operand::Primal(a), Operand::Primal(b)) => {
            for e in 0..len {
                let value = f(primal[a + e * sa], primal[b + e * sb]);
                tan[dst + e * K..][..K].fill(value);
            }
        },
    }
}
