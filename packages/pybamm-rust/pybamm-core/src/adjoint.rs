//! Reverse-mode (VJP) assembly of dense Jacobian rows.
//!
//! Tapes use a no-reuse (SSA) layout so every operand slot is a stable value-tape
//! offset; seeding element `r` of a wide root recovers row `r`.

use std::ops::Range;

use crate::accumulate::{Accumulate, Product, dispatch};
use crate::arena::DagRef;
use crate::branch_regions::active_branch;
use crate::eval::{
    CompiledExpr, NdCell, arg_reduce, csr_rows, interp_cubic_1d_deriv, interp_linear_1d_deriv,
    nd_coords, nd_source_offset, sign, split_read_write,
};
use crate::ir::{BinaryOp, BroadcastKind, ConstPool, Instruction, TypedIr, UnaryOp};
use crate::node::CsrData;

/// Adjoint (reverse-mode AD) tape for one expression, one seeded backward pass per row.
#[derive(Debug, Clone)]
pub struct AdjointTape {
    expr: CompiledExpr,
    n_states: usize,
    /// The merged state ranges the tape loads: the only `grad` entries a backward pass writes.
    state_reads: Vec<Range<usize>>,
    /// Per instruction, the `Dispatch` whose block span ends there, or `u32::MAX`.
    span_owner: Vec<u32>,
}

impl AdjointTape {
    /// Compile `dag` with a no-reuse (SSA) layout so the primal scratch is the value tape.
    pub fn new(dag: DagRef<'_>, n_states: usize) -> Self {
        let expr = CompiledExpr::new_pinned(dag);
        for instr in expr.ir().instructions() {
            assert!(
                !matches!(
                    instr,
                    Instruction::LoadTangentState { .. }
                        | Instruction::LoadTangentParameter { .. }
                        | Instruction::ReduceArgSelect { .. }
                ),
                "adjoint tape requires a primal row: found derivative-only {instr:?}"
            );
        }
        let span_owner = build_span_owner(expr.ir());
        Self {
            state_reads: merged_state_reads(expr.ir()),
            expr,
            n_states,
            span_owner,
        }
    }

    /// Tape length in instructions.
    #[inline]
    pub(crate) fn instruction_count(&self) -> usize {
        self.expr.ir().instructions().len()
    }

    /// Scratch length for the value tape and the parallel `bar` buffer.
    #[inline]
    pub const fn scratch_len(&self) -> usize {
        self.expr.scratch_len()
    }

    /// Rows this tape can recover (the root's width).
    #[inline]
    pub const fn n_rows(&self) -> usize {
        self.expr.output_len()
    }

    /// Fill the value tape shared by every row's backward pass.
    pub fn eval_forward(
        &self,
        scratch: &mut [f64],
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
    ) {
        self.expr.eval(scratch, t, y, y_dot, inputs);
    }

    /// Assemble row `row` into `grad[..n_states]` from a `scratch` filled by
    /// [`eval_forward`](Self::eval_forward); returns the instructions walked.
    ///
    /// Only the states the tape loads are written, re-zeroed first, so every other `grad`
    /// entry must already be zero; [`clear_state_reads`](Self::clear_state_reads) restores that.
    pub fn assemble_row(
        &self,
        scratch: &[f64],
        bar: &mut [f64],
        grad: &mut [f64],
        row: usize,
    ) -> usize {
        self.seed_row(bar, grad, row);
        backward(self.expr.ir(), &self.span_owner, scratch, bar, grad)
    }

    /// Zero the `grad` entries a row writes, leaving all of `grad` zero after a row.
    pub fn clear_state_reads(&self, grad: &mut [f64]) {
        for states in &self.state_reads {
            grad[states.clone()].fill(0.0);
        }
    }

    fn seed_row(&self, bar: &mut [f64], grad: &mut [f64], row: usize) {
        let ir = self.expr.ir();
        let root = ir.root_slot();
        assert!(
            row < root.len_usize(),
            "row {row} is outside the tape's {} rows",
            root.len_usize()
        );
        debug_assert!(
            self.only_state_reads_are_dirty(grad),
            "`grad` must be zero outside the states the tape reads"
        );
        self.clear_state_reads(grad);
        bar[..ir.buffer_size()].fill(0.0);
        bar[root.offset_usize() + row] = 1.0;
    }

    /// Forward pass then row 0; returns the instructions walked, excluding skipped blocks.
    ///
    /// `scratch` and `bar` need [`scratch_len`](Self::scratch_len) and `grad` `n_states`.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble(
        &self,
        scratch: &mut [f64],
        bar: &mut [f64],
        grad: &mut [f64],
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
    ) -> usize {
        self.expr.eval(scratch, t, y, y_dot, inputs);
        grad[..self.n_states].fill(0.0);
        self.assemble_row(scratch, bar, grad, 0)
    }

    /// Whether every `grad` entry outside the tape's state reads is zero.
    fn only_state_reads_are_dirty(&self, grad: &[f64]) -> bool {
        let mut written = vec![false; self.n_states];
        for states in &self.state_reads {
            written[states.clone()].fill(true);
        }
        grad[..self.n_states]
            .iter()
            .zip(written)
            .all(|(&g, written)| written || g == 0.0)
    }

    #[cfg(test)]
    pub(crate) const fn expr_for_test(&self) -> &CompiledExpr {
        &self.expr
    }
}

/// The state ranges `ir`'s `LoadStateVector`s read, ascending with overlaps merged.
fn merged_state_reads(ir: &TypedIr) -> Vec<Range<usize>> {
    let mut reads: Vec<Range<usize>> = ir
        .instructions()
        .iter()
        .filter_map(|instr| match *instr {
            Instruction::LoadStateVector { start, width, .. } => {
                Some(start.index()..start.index() + width.index())
            },
            _ => None,
        })
        .collect();
    reads.sort_unstable_by_key(|states| states.start);
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(reads.len());
    for states in reads {
        match merged.last_mut() {
            Some(last) if states.start <= last.end => last.end = last.end.max(states.end),
            _ => merged.push(states),
        }
    }
    merged
}

/// Reverse of `broadcast_apply` given partials `pa = ∂f/∂x` and `pb = ∂f/∂y`.
///
/// A zero output adjoint skips the partial, which may be undefined on an inactive branch.
#[allow(clippy::too_many_arguments)]
#[inline]
fn reverse_broadcast<Fa, Fb>(
    scratch: &[f64],
    bar: &mut [f64],
    a: usize,
    b: usize,
    dst: usize,
    len: usize,
    kind: BroadcastKind,
    pa: Fa,
    pb: Fb,
) where
    Fa: Fn(f64, f64) -> f64,
    Fb: Fn(f64, f64) -> f64,
{
    match kind {
        BroadcastKind::ScalarScalar => {
            let bd = bar[dst];
            if bd == 0.0 {
                return;
            }
            let (x, y) = (scratch[a], scratch[b]);
            bar[a] += pa(x, y) * bd;
            bar[b] += pb(x, y) * bd;
        },
        BroadcastKind::ScalarVector => {
            let x = scratch[a];
            let mut acc = 0.0;
            for i in 0..len {
                let bd = bar[dst + i];
                if bd == 0.0 {
                    continue;
                }
                let y = scratch[b + i];
                acc += pa(x, y) * bd;
                bar[b + i] += pb(x, y) * bd;
            }
            bar[a] += acc;
        },
        BroadcastKind::VectorScalar => {
            let y = scratch[b];
            let mut acc = 0.0;
            for i in 0..len {
                let bd = bar[dst + i];
                if bd == 0.0 {
                    continue;
                }
                let x = scratch[a + i];
                bar[a + i] += pa(x, y) * bd;
                acc += pb(x, y) * bd;
            }
            bar[b] += acc;
        },
        BroadcastKind::VectorVector => {
            for i in 0..len {
                let bd = bar[dst + i];
                if bd == 0.0 {
                    continue;
                }
                let (x, y) = (scratch[a + i], scratch[b + i]);
                bar[a + i] += pa(x, y) * bd;
                bar[b + i] += pb(x, y) * bd;
            }
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn reverse_binary(
    op: BinaryOp,
    a: usize,
    b: usize,
    dst: usize,
    len: usize,
    kind: BroadcastKind,
    scratch: &[f64],
    bar: &mut [f64],
) {
    match op {
        BinaryOp::Add => {
            reverse_broadcast(scratch, bar, a, b, dst, len, kind, |_, _| 1.0, |_, _| 1.0);
        },
        BinaryOp::Sub => {
            reverse_broadcast(scratch, bar, a, b, dst, len, kind, |_, _| 1.0, |_, _| -1.0);
        },
        BinaryOp::Mul => reverse_broadcast(scratch, bar, a, b, dst, len, kind, |_, y| y, |x, _| x),
        BinaryOp::Div => {
            reverse_broadcast(
                scratch,
                bar,
                a,
                b,
                dst,
                len,
                kind,
                |_, y| 1.0 / y,
                |x, y| -x / (y * y),
            );
        },
        BinaryOp::Pow => reverse_broadcast(
            scratch,
            bar,
            a,
            b,
            dst,
            len,
            kind,
            |x, y| y * BinaryOp::Pow.apply(x, y - 1.0),
            // 0 where x <= 0, as forward mode's `exponent_partial` takes it.
            |x, y| {
                if x > 0.0 {
                    BinaryOp::Pow.apply(x, y) * UnaryOp::Log.apply(x)
                } else {
                    0.0
                }
            },
        ),
        BinaryOp::Minimum => reverse_broadcast(
            scratch,
            bar,
            a,
            b,
            dst,
            len,
            kind,
            |x, y| if x <= y { 1.0 } else { 0.0 },
            |x, y| if x <= y { 0.0 } else { 1.0 },
        ),
        BinaryOp::Maximum => reverse_broadcast(
            scratch,
            bar,
            a,
            b,
            dst,
            len,
            kind,
            |x, y| if x >= y { 1.0 } else { 0.0 },
            |x, y| if x >= y { 0.0 } else { 1.0 },
        ),
        BinaryOp::Modulo => reverse_broadcast(
            scratch,
            bar,
            a,
            b,
            dst,
            len,
            kind,
            |_, _| 1.0,
            // Forward mode's `-trunc(x / y)`: `%` truncates the exact quotient, not `fl(x / y)`.
            |x, y| (x % y - x) / y,
        ),
        BinaryOp::Hypot => reverse_broadcast(
            scratch,
            bar,
            a,
            b,
            dst,
            len,
            kind,
            |x, y| x / BinaryOp::Hypot.apply(x, y),
            |x, y| y / BinaryOp::Hypot.apply(x, y),
        ),
        BinaryOp::EqualHeaviside | BinaryOp::NotEqualHeaviside | BinaryOp::Equality => {
            reverse_broadcast(scratch, bar, a, b, dst, len, kind, |_, _| 0.0, |_, _| 0.0);
        },
    }
}

/// Elementwise unary adjoint; zero adjoints skip potentially undefined partials.
fn reverse_unary(
    op: UnaryOp,
    src: usize,
    dst: usize,
    len: usize,
    scratch: &[f64],
    bar: &mut [f64],
) {
    for i in 0..len {
        let bd = bar[dst + i];
        if bd == 0.0 {
            continue;
        }
        let x = scratch[src + i];
        let out = scratch[dst + i];
        let d = match op {
            UnaryOp::Neg => -1.0,
            UnaryOp::Abs => sign(x),
            UnaryOp::Sqrt => 0.5 / out,
            UnaryOp::Exp => out,
            UnaryOp::Log => 1.0 / x,
            UnaryOp::Sin => UnaryOp::Cos.apply(x),
            UnaryOp::Cos => -UnaryOp::Sin.apply(x),
            UnaryOp::Tanh => 1.0 - out * out,
            UnaryOp::Sinh => UnaryOp::Cosh.apply(x),
            UnaryOp::Cosh => UnaryOp::Sinh.apply(x),
            UnaryOp::Arcsinh => 1.0 / UnaryOp::Sqrt.apply(1.0 + x * x),
            UnaryOp::Arctan => 1.0 / (1.0 + x * x),
            UnaryOp::Erf => std::f64::consts::FRAC_2_SQRT_PI * UnaryOp::Exp.apply(-x * x),
            UnaryOp::Sign | UnaryOp::Floor | UnaryOp::Ceiling => 0.0,
        };
        bar[src + i] += d * bd;
    }
}

/// Map each `Dispatch` span's last instruction index to the `Dispatch` itself.
///
/// The release-mode asserts guard against overlapping spans, which would give wrong gradients.
fn build_span_owner(ir: &TypedIr) -> Vec<u32> {
    let instrs = ir.instructions();
    let mut span_owner = vec![u32::MAX; instrs.len()];
    for (i, instr) in instrs.iter().enumerate() {
        if let Instruction::Dispatch { span_len, .. } = *instr {
            let end = i + span_len.index();
            assert!(end > i, "a dispatch span must cover at least itself");
            assert_eq!(
                span_owner[end - 1],
                u32::MAX,
                "overlapping dispatch spans: blocks must be flat and disjoint"
            );
            span_owner[end - 1] =
                u32::try_from(i).expect("instruction indices are u32 throughout the IR");
        }
    }
    span_owner
}

/// Backward replay over a pinned primal stream from a seeded `bar`, skipping inactive
/// branch blocks; returns the instructions walked.
///
/// Skipping is sound because `Conditional` pushes adjoint into the active branch only.
fn backward(
    ir: &TypedIr,
    span_owner: &[u32],
    scratch: &[f64],
    bar: &mut [f64],
    grad: &mut [f64],
) -> usize {
    let consts = ir.consts();
    let instructions = ir.instructions();
    let mut walked = 0_usize;
    let mut i = instructions.len();
    while i > 0 {
        i -= 1;
        let owner = span_owner[i];
        if owner != u32::MAX {
            let d = owner as usize;
            let Instruction::Dispatch {
                selector,
                blocks_idx,
                blocks_len,
                ..
            } = instructions[d]
            else {
                unreachable!("span_owner points at a Dispatch")
            };
            walked += 1;
            if let Some(active) = active_branch(scratch[selector.index()], blocks_len.index()) {
                let block = consts.branch_blocks[blocks_idx.index() + active].bounds(d);
                walked += block.len();
                for k in block.rev() {
                    backward_instruction(consts, instructions[k], scratch, bar, grad);
                }
            }
            i = d; // the Dispatch itself has no adjoint
            continue;
        }
        walked += 1;
        backward_instruction(consts, instructions[i], scratch, bar, grad);
    }
    walked
}

/// Accumulate one instruction's adjoint into `bar` (and `grad` for a state load).
#[inline]
fn backward_instruction(
    consts: &ConstPool,
    instr: Instruction,
    scratch: &[f64],
    bar: &mut [f64],
    grad: &mut [f64],
) {
    match instr {
        // Interpolant derivatives are treated as constant w.r.t. their input.
        Instruction::LoadScalar { .. }
        | Instruction::LoadTime { .. }
        | Instruction::LoadArray { .. }
        | Instruction::FillZero { .. }
        | Instruction::LoadStateVectorDot { .. }
        | Instruction::LoadInputParameter { .. }
        | Instruction::Interp1DLinearDeriv { .. }
        | Instruction::Interp1DCubicDeriv { .. }
        | Instruction::InterpNdPartial { .. } => {},

        Instruction::LoadStateVector { start, width, dst } => {
            let (start, dst) = (start.index(), dst.index());
            for k in 0..width.index() {
                grad[start + k] += bar[dst + k];
            }
        },

        // Rejected in `AdjointTape::new`.
        Instruction::LoadTangentState { .. }
        | Instruction::LoadTangentParameter { .. }
        | Instruction::ReduceArgSelect { .. } => {
            unreachable!("derivative-only instruction in primal adjoint tape")
        },

        Instruction::Binary {
            op,
            a,
            b,
            dst,
            len,
            kind,
        } => {
            reverse_binary(
                op,
                a.index(),
                b.index(),
                dst.index(),
                len.index(),
                kind,
                scratch,
                bar,
            );
        },
        Instruction::Unary { op, src, dst, len } => {
            reverse_unary(op, src.index(), dst.index(), len.index(), scratch, bar);
        },

        Instruction::MaxReduce { src, src_len, dst } => {
            let bd = bar[dst.index()];
            if bd == 0.0 {
                return;
            }
            let src = src.index();
            let k = arg_reduce(scratch[src..src + src_len.index()].iter().copied(), true);
            bar[src + k] += bd;
        },
        Instruction::MinReduce { src, src_len, dst } => {
            let bd = bar[dst.index()];
            if bd == 0.0 {
                return;
            }
            let src = src.index();
            let k = arg_reduce(scratch[src..src + src_len.index()].iter().copied(), false);
            bar[src + k] += bd;
        },

        Instruction::Index {
            src,
            start,
            dst,
            len,
        } => {
            let src = src.offset_by(start);
            let dst = dst.index();
            for i in 0..len.index() {
                bar[src + i] += bar[dst + i];
            }
        },
        Instruction::Concat {
            sources_idx,
            sources_len,
            dst,
        } => {
            let dst = dst.index();
            let mut o = 0usize;
            for s in 0..sources_len.index() {
                let slot = consts.concat_sources[sources_idx.index() + s];
                for i in 0..slot.len_usize() {
                    bar[slot.offset_usize() + i] += bar[dst + o + i];
                }
                o += slot.len_usize();
            }
        },

        Instruction::MatMul {
            csr_idx,
            vec_src,
            dst,
        } => {
            // bar_vec += Aᵀ @ bar_dst.
            let mat = &consts.csr_data[csr_idx.index()];
            let (bar_dst, bar_vec) = split_read_write(
                bar,
                dst.index(),
                mat.shape.rows,
                vec_src.index(),
                mat.shape.cols,
            );
            dispatch(CsrTransposed {
                csr: mat,
                bar_dst,
                bar_vec,
            });
        },
        Instruction::DenseMatMul {
            mat_src,
            rows,
            cols,
            vec_src,
            dst,
        } => {
            // A is row-major; bar_vec += Aᵀ @ bar_dst.
            dispatch(DenseTransposed {
                scratch,
                bar,
                mat: mat_src.index(),
                vec: vec_src.index(),
                dst: dst.index(),
                rows: rows.index(),
                cols: cols.index(),
            });
        },

        Instruction::Interp1DLinear {
            interp_idx,
            src,
            dst,
            len,
        } => {
            let interp = &consts.interpolants[interp_idx.index()];
            let src = src.index();
            let dst = dst.index();
            for i in 0..len.index() {
                let bd = bar[dst + i];
                if bd == 0.0 {
                    continue;
                }
                let slope = interp_linear_1d_deriv(interp, scratch[src + i]);
                bar[src + i] += slope * bd;
            }
        },
        Instruction::Interp1DCubic {
            interp_idx,
            src,
            dst,
            len,
        } => {
            let interp = &consts.cubic_interpolants[interp_idx.index()];
            let src = src.index();
            let dst = dst.index();
            for i in 0..len.index() {
                let bd = bar[dst + i];
                if bd == 0.0 {
                    continue;
                }
                let d =
                    interp_cubic_1d_deriv(&interp.breakpoints, &interp.coeffs, scratch[src + i]);
                bar[src + i] += d * bd;
            }
        },
        Instruction::InterpNd {
            interp_idx,
            sources_idx,
            dst,
            len,
        } => {
            let interp = &consts.nd_interpolants[interp_idx.index()];
            let cell = NdCell::of(interp);
            let sources = &consts.interp_nd_sources[sources_idx.index()..][..cell.ndim()];
            let dst = dst.index();
            let mut dxs = [0.0_f64; 3];
            for i in 0..len.index() {
                let bd = bar[dst + i];
                if bd == 0.0 {
                    continue;
                }
                let coords = nd_coords(sources, i, |offset| scratch[offset]);
                let coeffs = cell.locate(interp, &coords, &mut dxs);
                for (a, &source) in sources.iter().enumerate() {
                    let partial = cell.partial(coeffs, &dxs, a);
                    bar[nd_source_offset(source, i)] += partial * bd;
                }
            }
        },

        Instruction::Conditional {
            selector,
            branches_idx,
            branches_len,
            dst,
            out_len,
        } => {
            let dst = dst.index();
            let out_len = out_len.index();
            if let Some(i) = active_branch(scratch[selector.index()], branches_len.index()) {
                let slot = consts.branch_offsets[branches_idx.index() + i];
                let off = slot.offset_usize();
                for k in 0..out_len {
                    bar[off + k] += bar[dst + k];
                }
            }
        },

        Instruction::Dispatch { .. } => unreachable!("handled via the span table"),
    }
}

/// `bar_vec += Aᵀ @ bar_dst` for a CSR `A`, skipping rows with no adjoint.
struct CsrTransposed<'a> {
    csr: &'a CsrData,
    bar_dst: &'a [f64],
    bar_vec: &'a mut [f64],
}

impl Product for CsrTransposed<'_> {
    type Output = ();

    // Must inline into `dispatch`'s FMA-enabled wrapper.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn run<A: Accumulate>(self) {
        let Self {
            csr,
            bar_dst,
            bar_vec,
        } = self;
        for (&bd, (cols, vals)) in bar_dst.iter().zip(csr_rows(csr)) {
            if bd == 0.0 {
                continue;
            }
            for (&col, &value) in cols.iter().zip(vals) {
                bar_vec[col] = A::mul_acc(bar_vec[col], value, bd);
            }
        }
    }
}

/// `bar[vec..] += Mᵀ @ bar[dst..]` for the row-major `rows x cols` matrix at `scratch[mat..]`.
struct DenseTransposed<'a> {
    scratch: &'a [f64],
    bar: &'a mut [f64],
    mat: usize,
    vec: usize,
    dst: usize,
    rows: usize,
    cols: usize,
}

impl Product for DenseTransposed<'_> {
    type Output = ();

    // Must inline into `dispatch`'s FMA-enabled wrapper.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn run<A: Accumulate>(self) {
        let Self {
            scratch,
            bar,
            mat,
            vec,
            dst,
            rows,
            cols,
        } = self;
        for row in 0..rows {
            let bd = bar[dst + row];
            if bd == 0.0 {
                continue;
            }
            let base = mat + row * cols;
            for j in 0..cols {
                bar[vec + j] = A::mul_acc(bar[vec + j], scratch[base + j], bd);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::adjoint::AdjointTape;
    use crate::arena::{Arena, DagRef};
    use crate::node::{CsrData, Node, Shape};

    fn assemble(tape: &AdjointTape, y: &[f64]) -> Vec<f64> {
        let mut scratch = vec![0.0; tape.scratch_len()];
        let mut bar = vec![0.0; tape.scratch_len()];
        let mut grad = vec![0.0; tape.n_states];
        tape.assemble(&mut scratch, &mut bar, &mut grad, 0.0, y, &[], &[]);
        grad
    }

    #[test]
    fn test_reverse_mul_scalar_scalar() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
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
        let f = arena.alloc(Node::Mul(y0, y1));
        let tape = AdjointTape::new(DagRef::new(&arena, f), 3);
        let grad = assemble(&tape, &[2.0, 3.0, 5.0]);
        assert_eq!(grad, vec![3.0, 2.0, 0.0]);
    }

    #[test]
    fn test_reverse_matmul_and_broadcast_sum() {
        // f = [1,1,1] @ (y0 * y[1..4]); grad = [y1+y2+y3, y0, y0, y0].
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 4 });
        let s = arena.alloc(Node::Index {
            child: y,
            start: 0,
            end: 1,
        });
        let v = arena.alloc(Node::Index {
            child: y,
            start: 1,
            end: 4,
        });
        let p = arena.alloc(Node::Mul(s, v));
        let ones = arena.alloc(Node::SparseMatrix(Box::new(
            CsrData::try_new(
                vec![0, 3],
                vec![0, 1, 2],
                vec![1.0, 1.0, 1.0],
                Shape::matrix(1, 3),
            )
            .unwrap(),
        )));
        let f = arena.alloc(Node::MatMul(ones, p));
        let tape = AdjointTape::new(DagRef::new(&arena, f), 4);
        let grad = assemble(&tape, &[2.0, 3.0, 4.0, 5.0]);
        assert_eq!(grad, vec![12.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_reverse_exp_index() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let y2 = arena.alloc(Node::Index {
            child: y,
            start: 2,
            end: 3,
        });
        let f = arena.alloc(Node::Exp(y2));
        let tape = AdjointTape::new(DagRef::new(&arena, f), 3);
        let grad = assemble(&tape, &[1.0, 1.0, 0.0]);
        assert!((grad[2] - 1.0).abs() < 1e-14 && grad[0] == 0.0 && grad[1] == 0.0);
    }

    #[test]
    fn state_reads_merge_overlapping_loads() {
        let mut arena = Arena::new();
        let loads = [0..2, 5..6, 1..3].map(|states| {
            arena.alloc(Node::StateVector {
                start: states.start,
                end: states.end,
            })
        });
        let root = arena.alloc(Node::Concat(loads.to_vec()));
        let tape = AdjointTape::new(DagRef::new(&arena, root), 6);
        assert_eq!(tape.state_reads, [0..3, 5..6]);
    }

    #[test]
    fn rows_rezero_only_the_states_the_tape_reads() {
        // Each row re-zeroes y0 and y2, the tape's reads, and never touches y1 or y3.
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y2 = arena.alloc(Node::StateVector { start: 2, end: 3 });
        let root = arena.alloc(Node::Concat(vec![y0, y2]));
        let tape = AdjointTape::new(DagRef::new(&arena, root), 4);

        let mut scratch = vec![0.0; tape.scratch_len()];
        let mut bar = vec![0.0; tape.scratch_len()];
        let mut grad = vec![7.0, 0.0, 7.0, 0.0];
        tape.eval_forward(&mut scratch, 0.0, &[1.0, 2.0, 3.0, 4.0], &[], &[]);
        tape.assemble_row(&scratch, &mut bar, &mut grad, 0);
        assert_eq!(grad, [1.0, 0.0, 0.0, 0.0]);
        tape.assemble_row(&scratch, &mut bar, &mut grad, 1);
        assert_eq!(grad, [0.0, 0.0, 1.0, 0.0]);
        tape.clear_state_reads(&mut grad);
        assert_eq!(grad, [0.0; 4]);
    }

    #[test]
    fn test_reverse_modulo_differentiates_the_divisor() {
        // d(y0 mod y1)/dy1 is -trunc(y0 / y1), a positive step for a negative dividend.
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
        let f = arena.alloc(Node::Modulo(y0, y1));
        let tape = AdjointTape::new(DagRef::new(&arena, f), 2);
        for (dividend, expected) in [(3.0, -2.0), (0.5, 0.0), (-2.0, 1.0)] {
            let grad = assemble(&tape, &[dividend, 1.3]);
            assert!((grad[0] - 1.0).abs() < 1e-14, "d/dy0 at y0 = {dividend}");
            assert!(
                (grad[1] - expected).abs() < 1e-12,
                "d/dy1 at y0 = {dividend}"
            );
        }
    }

    #[test]
    fn test_inactive_conditional_unary_does_not_poison_gradient() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let invalid = arena.alloc(Node::Sqrt(y));
        let selector = arena.alloc(Node::Scalar(1.0));
        let root = arena.alloc(Node::Conditional {
            selector,
            branches: vec![y, invalid],
        });

        let tape = AdjointTape::new(DagRef::new(&arena, root), 1);
        let grad = assemble(&tape, &[-1.0]);
        assert_eq!(grad, vec![1.0]);
    }

    #[test]
    fn test_inactive_conditional_binary_does_not_poison_gradient() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let invalid = arena.alloc(Node::Div(y, zero));
        let selector = arena.alloc(Node::Scalar(1.0));
        let root = arena.alloc(Node::Conditional {
            selector,
            branches: vec![y, invalid],
        });

        let tape = AdjointTape::new(DagRef::new(&arena, root), 1);
        let grad = assemble(&tape, &[2.0]);
        assert_eq!(grad, vec![1.0]);
    }

    #[test]
    fn test_pinned_reverse_tape_has_no_self_aliasing_binary() {
        // Regression: the no-reuse layout must never alias a binary operand with its dst.
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
        let f = arena.alloc(Node::Mul(y0, y1));
        let tape = AdjointTape::new(DagRef::new(&arena, f), 2);
        for instr in tape.expr_for_test().ir().instructions() {
            if let crate::ir::Instruction::Binary { a, b, dst, len, .. } = *instr {
                let (a, b, dst, len) = (a.index(), b.index(), dst.index(), len.index());
                assert!(a + len <= dst || dst + len <= a, "operand a aliases dst");
                assert!(b + len <= dst || dst + len <= b, "operand b aliases dst");
            }
        }
    }

    #[test]
    #[should_panic(expected = "primal row")]
    fn test_reverse_tape_rejects_derivative_only_node() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let basis = arena.alloc(Node::Index {
            child: y,
            start: 0,
            end: 1,
        });
        let picker = arena.alloc(Node::Index {
            child: y,
            start: 1,
            end: 2,
        });
        let ras = arena.alloc(Node::ReduceArgSelect {
            basis,
            picker,
            is_max: true,
        });
        let _ = AdjointTape::new(DagRef::new(&arena, ras), 2);
    }
}
