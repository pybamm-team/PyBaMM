//! Reverse-mode (VJP) assembly of dense Jacobian rows.
//!
//! Wide rows split out of the column coloring are filled by backward passes
//! over their primal sub-expression instead of one forward JVP sweep per
//! column. The sub-expression is compiled with a no-reuse (SSA) slot layout
//! ([`CompiledExpr::new_pinned`]), so after one primal evaluation the scratch
//! buffer holds every intermediate; each instruction's operand slots are then
//! stable value-tape offsets the adjoint reads directly. The backward `match`
//! over [`Instruction`] is exhaustive: every op has an adjoint, so there is no
//! runtime fallback.
//!
//! A tape's root may be wider than one element, in which case seeding element
//! `r` recovers row `r`, so a group of rows shares one forward pass and one
//! compiled tape. See [`crate::row_extract`] for how many rows share one.
//!
//! The pinned layout forms branch blocks like every other layout, so the
//! backward walk jumps over the blocks of inactive conditional branches instead
//! of replaying adjoints that are all no-ops.

// `NodeId` must come from `arena`: `node.rs` only re-imports it privately, so
// `crate::node::NodeId` is not reachable from another module.
use crate::arena::{Arena, NodeId};
use crate::branch_regions::{active_branch, dispatch_span_end};
use crate::eval::{
    CompiledExpr, interp_cubic_1d_deriv, interp_linear_1d_deriv, locate_nd_cell, sign,
    tensor_horner_partial,
};
use crate::ir::{BinaryOp, BroadcastKind, ConstPool, Instruction, TypedIr, UnaryOp};

/// Prepared adjoint (reverse-mode AD) tape for one expression, whose rows are
/// recovered one seeded backward pass at a time.
#[derive(Debug, Clone)]
pub struct AdjointTape {
    expr: CompiledExpr,
    n_states: usize,
    /// For each instruction index, the index of the `Dispatch` whose block span
    /// *ends* there, or `u32::MAX`. Precomputed once so the hot backward walk
    /// allocates nothing.
    span_owner: Vec<u32>,
}

impl AdjointTape {
    /// Compile `root` with a no-reuse (SSA) layout so the primal scratch is the
    /// reverse value tape. Its width is how many rows the tape can recover.
    ///
    /// # Panics
    /// Panics if `root` contains a derivative-only instruction (a tangent load
    /// or `ReduceArgSelect`); a primal row lifted out of the residual never
    /// does.
    pub fn new(arena: &Arena, root: NodeId, n_states: usize) -> Self {
        let expr = CompiledExpr::new_pinned(arena, root);
        // Assert rather than assume: `ReduceArgSelect` and the tangent loads are
        // built only by `tangent.rs`, never from Python, so no primal row has them.
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
            expr,
            n_states,
            span_owner,
        }
    }

    /// Tape length, so what this artifact costs in compiled memory.
    #[inline]
    pub fn instruction_count(&self) -> usize {
        self.expr.ir().instructions().len()
    }

    /// Scratch length for the value tape (and the parallel `bar` buffer).
    #[inline]
    pub const fn scratch_len(&self) -> usize {
        self.expr.scratch_len()
    }

    /// State dimension the gradient row spans.
    #[inline]
    pub const fn n_states(&self) -> usize {
        self.n_states
    }

    /// Rows this tape can recover, so its root's width.
    #[inline]
    pub const fn n_rows(&self) -> usize {
        self.expr.output_len()
    }

    /// Fill the value tape, which every row's backward pass then reads.
    ///
    /// Split out from [`assemble_row`](Self::assemble_row) so a block of rows
    /// pays for the shared forward work once rather than per row.
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

    /// Assemble row `row` into `grad[..n_states]` from an already-filled
    /// `scratch`, as left by [`eval_forward`](Self::eval_forward), returning how
    /// many instructions the backward walk touched.
    ///
    /// # Panics
    /// Panics if `row` is outside the root's width.
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

    /// Clear the gradient and `bar` buffers and seed row `row`'s adjoint.
    ///
    /// # Panics
    /// Panics if `row` is outside the root's width.
    fn seed_row(&self, bar: &mut [f64], grad: &mut [f64], row: usize) {
        let ir = self.expr.ir();
        let root = ir.root_slot();
        assert!(
            row < root.len_usize(),
            "row {row} is outside the tape's {} rows",
            root.len_usize()
        );
        grad[..self.n_states].fill(0.0);
        bar[..ir.buffer_size()].fill(0.0);
        bar[root.offset_usize() + row] = 1.0;
    }

    /// Forward pass then row 0, so the whole gradient row `df/dy` in one call,
    /// returning how many instructions the backward walk touched.
    ///
    /// `scratch` and `bar` must each be at least [`scratch_len`](Self::scratch_len);
    /// `grad` at least `n_states`. All three are caller-provided and reset here,
    /// no allocation occurs. Skipped blocks are excluded from the count, so a
    /// test can assert an inactive branch's adjoint arms never ran.
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
        // Forward record: fill the value tape (unique slot per node).
        self.expr.eval(scratch, t, y, y_dot, inputs);
        self.assemble_row(scratch, bar, grad, 0)
    }

    #[cfg(test)]
    pub(crate) const fn expr_for_test(&self) -> &CompiledExpr {
        &self.expr
    }
}

/// Argmax (`is_max`) or argmin of `vals`, taking the earliest element under a
/// strict comparison to match the primal `MaxReduce`/`MinReduce` eval.
fn arg_reduce(vals: &[f64], is_max: bool) -> usize {
    let mut best = 0;
    for i in 1..vals.len() {
        let better = if is_max {
            vals[i] > vals[best]
        } else {
            vals[i] < vals[best]
        };
        if better {
            best = i;
        }
    }
    best
}

/// Reverse of `broadcast_apply`: accumulate operand adjoints given the two
/// per-element partials `pa = ∂f/∂x` and `pb = ∂f/∂y`. The scalar operand of a
/// broadcast has its `bar` sum-reduced (the transpose of broadcasting). A zero
/// output adjoint skips the local partial, which may be undefined on an inactive branch.
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

/// Dispatch the binary adjoint by op. Partials mirror `eval_binary_op`.
#[allow(clippy::suboptimal_flops, clippy::too_many_arguments)]
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
            |x, y| y * x.powf(y - 1.0),
            |x, y| if x > 0.0 { x.powf(y) * x.ln() } else { 0.0 },
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
        BinaryOp::Modulo => {
            reverse_broadcast(scratch, bar, a, b, dst, len, kind, |_, _| 1.0, |_, _| 0.0);
        },
        BinaryOp::Hypot => reverse_broadcast(
            scratch,
            bar,
            a,
            b,
            dst,
            len,
            kind,
            |x, y| x / x.hypot(y),
            |x, y| y / x.hypot(y),
        ),
        BinaryOp::EqualHeaviside | BinaryOp::NotEqualHeaviside | BinaryOp::Equality => {
            reverse_broadcast(scratch, bar, a, b, dst, len, kind, |_, _| 0.0, |_, _| 0.0);
        },
    }
}

/// Elementwise unary adjoint: `bar[src+i] += f'(x) * bar[dst+i]`. Derivatives
/// mirror `eval_unary_op`; zero adjoints skip potentially undefined partials.
#[allow(clippy::suboptimal_flops)]
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
            UnaryOp::Sin => x.cos(),
            UnaryOp::Cos => -x.sin(),
            UnaryOp::Tanh => 1.0 - out * out,
            UnaryOp::Sinh => x.cosh(),
            UnaryOp::Cosh => x.sinh(),
            UnaryOp::Arcsinh => 1.0 / (1.0 + x * x).sqrt(),
            UnaryOp::Arctan => 1.0 / (1.0 + x * x),
            UnaryOp::Erf => 2.0 / std::f64::consts::PI.sqrt() * (-x * x).exp(),
            UnaryOp::Sign | UnaryOp::Floor | UnaryOp::Ceiling => 0.0,
        };
        bar[src + i] += d * bd;
    }
}

/// Map each `Dispatch` span's last instruction index to the `Dispatch` itself.
///
/// Blocks are flat and pairwise disjoint (a nested conditional is never
/// blocked), so one slot per index is enough. Both assertions ship rather than
/// being debug-only, for the same reason as `ir.rs`'s block-safety guards: an
/// overlapping span would silently drop one owner, and the failure mode is wrong
/// gradients, not a crash. This runs once per compile, off the hot path.
///
/// # Panics
/// Panics if two `Dispatch` spans end at the same instruction.
fn build_span_owner(ir: &TypedIr) -> Vec<u32> {
    let instrs = ir.instructions();
    let consts = ir.consts();
    let mut span_owner = vec![u32::MAX; instrs.len()];
    for (i, instr) in instrs.iter().enumerate() {
        if let Instruction::Dispatch {
            blocks_idx,
            blocks_len,
            ..
        } = *instr
        {
            let end = dispatch_span_end(consts, i, blocks_idx, blocks_len);
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

/// Backward (adjoint) replay over a pinned primal instruction stream, jumping
/// over the blocks of inactive conditional branches. Returns the number of
/// instructions touched, and expects `bar[root] = 1` already seeded.
///
/// Walking backwards reaches a block before its `Dispatch`, so `span_owner` says
/// which indices end a span; there the active branch is resolved from the
/// recorded primal selector and only that block is replayed. Skipping is sound
/// because the `Conditional` arm pushes the output adjoint into the matched
/// branch alone, so an inactive branch's slots carry `bar == 0`.
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
            } = instructions[d]
            else {
                unreachable!("span_owner points at a Dispatch")
            };
            walked += 1;
            if let Some(active) = active_branch(scratch[selector as usize], blocks_len as usize) {
                let (rel, len) = consts.branch_blocks[blocks_idx as usize + active];
                let start = d + rel as usize;
                // A block never contains a `Dispatch`, so replaying one needs no
                // further span handling.
                for k in (start..start + len as usize).rev() {
                    backward_instruction(consts, instructions[k], scratch, bar, grad);
                }
                walked += len as usize;
            }
            i = d; // the Dispatch itself has no adjoint
            continue;
        }
        walked += 1;
        backward_instruction(consts, instructions[i], scratch, bar, grad);
    }
    walked
}

/// Accumulate one instruction's adjoint contribution into `bar` (and `grad` for
/// a state load). Exhaustive over `Instruction`, no fallback.
#[inline]
fn backward_instruction(
    consts: &ConstPool,
    instr: Instruction,
    scratch: &[f64],
    bar: &mut [f64],
    grad: &mut [f64],
) {
    match instr {
        // Terminal instructions: their adjoint stops here. Interpolant
        // derivatives are constant w.r.t. input (forward returns zero).
        Instruction::LoadScalar { .. }
        | Instruction::LoadTime { .. }
        | Instruction::LoadArray { .. }
        | Instruction::FillZero { .. }
        | Instruction::LoadStateVectorDot { .. }
        | Instruction::LoadInputParameter { .. }
        | Instruction::Interp1DLinearDeriv { .. }
        | Instruction::Interp1DCubicDeriv { .. }
        | Instruction::InterpNdPartial { .. } => {},

        // The gradient sink.
        Instruction::LoadStateVector { start, end, dst } => {
            let dst = dst as usize;
            for (k, s) in (start as usize..end as usize).enumerate() {
                grad[s] += bar[dst + k];
            }
        },

        // Derivative-only instructions are rejected in `AdjointTape::new`; this
        // arm only keeps the `match` exhaustive with no runtime fallback.
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
                a as usize,
                b as usize,
                dst as usize,
                len as usize,
                kind,
                scratch,
                bar,
            );
        },
        Instruction::Unary { op, src, dst, len } => {
            reverse_unary(op, src as usize, dst as usize, len as usize, scratch, bar);
        },

        Instruction::MaxReduce { src, src_len, dst } => {
            let bd = bar[dst as usize];
            if bd == 0.0 {
                return;
            }
            let src = src as usize;
            let k = arg_reduce(&scratch[src..src + src_len as usize], true);
            bar[src + k] += bd;
        },
        Instruction::MinReduce { src, src_len, dst } => {
            let bd = bar[dst as usize];
            if bd == 0.0 {
                return;
            }
            let src = src as usize;
            let k = arg_reduce(&scratch[src..src + src_len as usize], false);
            bar[src + k] += bd;
        },

        Instruction::Index {
            src,
            start,
            dst,
            len,
        } => {
            let src = src as usize + start as usize;
            let dst = dst as usize;
            for i in 0..len as usize {
                bar[src + i] += bar[dst + i];
            }
        },
        Instruction::Concat {
            sources_idx,
            sources_len,
            dst,
        } => {
            let dst = dst as usize;
            let mut o = 0usize;
            for s in 0..sources_len as usize {
                let (off, slen) = consts.concat_sources[sources_idx as usize + s];
                for i in 0..slen as usize {
                    bar[off as usize + i] += bar[dst + o + i];
                }
                o += slen as usize;
            }
        },

        Instruction::MatMul {
            csr_idx,
            vec_src,
            dst,
        } => {
            // dst = A @ vec ; A constant, so bar_vec += Aᵀ @ bar_dst.
            let mat = &consts.csr_data[csr_idx as usize];
            let vec_src = vec_src as usize;
            let dst = dst as usize;
            for row in 0..mat.shape.rows {
                let bd = bar[dst + row];
                if bd == 0.0 {
                    continue;
                }
                for k in mat.indptr[row]..mat.indptr[row + 1] {
                    bar[vec_src + mat.indices[k]] += mat.data[k] * bd;
                }
            }
        },
        Instruction::DenseMatMul {
            mat_src,
            rows,
            cols,
            vec_src,
            dst,
        } => {
            // dst = A @ vec, A row-major constant ; bar_vec += Aᵀ @ bar_dst.
            let mat_src = mat_src as usize;
            let vec_src = vec_src as usize;
            let dst = dst as usize;
            let cols = cols as usize;
            for row in 0..rows as usize {
                let bd = bar[dst + row];
                if bd == 0.0 {
                    continue;
                }
                let base = mat_src + row * cols;
                for j in 0..cols {
                    bar[vec_src + j] += scratch[base + j] * bd;
                }
            }
        },

        Instruction::Interp1DLinear {
            interp_idx,
            src,
            dst,
            len,
        } => {
            let interp = &consts.interpolants[interp_idx as usize];
            let src = src as usize;
            let dst = dst as usize;
            for i in 0..len as usize {
                let bd = bar[dst + i];
                if bd == 0.0 {
                    continue;
                }
                let slope =
                    interp_linear_1d_deriv(&interp.x_data, &interp.y_data, scratch[src + i]);
                bar[src + i] += slope * bd;
            }
        },
        Instruction::Interp1DCubic {
            interp_idx,
            src,
            dst,
            len,
        } => {
            let interp = &consts.cubic_interpolants[interp_idx as usize];
            let src = src as usize;
            let dst = dst as usize;
            for i in 0..len as usize {
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
            let interp = &consts.nd_interpolants[interp_idx as usize];
            let ndim = interp.breakpoints.len();
            let order = interp.order as usize;
            let dst = dst as usize;
            let mut coords = [0.0_f64; 3];
            let mut dxs = [0.0_f64; 3];
            for i in 0..len as usize {
                let bd = bar[dst + i];
                if bd == 0.0 {
                    continue;
                }
                for (a, coord) in coords.iter_mut().enumerate().take(ndim) {
                    let (off, slen) = consts.interp_nd_sources[sources_idx as usize + a];
                    let j = if slen == 1 { 0 } else { i };
                    *coord = scratch[off as usize + j];
                }
                let cell = locate_nd_cell(
                    &interp.breakpoints,
                    &interp.coeffs,
                    order,
                    &coords[..ndim],
                    &mut dxs,
                );
                for a in 0..ndim {
                    let (off, slen) = consts.interp_nd_sources[sources_idx as usize + a];
                    let j = if slen == 1 { 0 } else { i };
                    let partial = tensor_horner_partial(cell, &dxs[..ndim], order, a);
                    bar[off as usize + j] += partial * bd;
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
            let dst = dst as usize;
            let out_len = out_len as usize;
            if let Some(i) = active_branch(scratch[selector as usize], branches_len as usize) {
                let (off, _) = consts.branch_offsets[branches_idx as usize + i];
                let off = off as usize;
                for k in 0..out_len {
                    bar[off + k] += bar[dst + k];
                }
            }
        },

        Instruction::Dispatch { .. } => unreachable!("handled via the span table"),
    }
}

#[cfg(test)]
mod tests {
    use crate::adjoint::AdjointTape;
    use crate::arena::Arena;
    use crate::node::{CsrData, Node, Shape};

    fn assemble(tape: &AdjointTape, y: &[f64]) -> Vec<f64> {
        let mut scratch = vec![0.0; tape.scratch_len()];
        let mut bar = vec![0.0; tape.scratch_len()];
        let mut grad = vec![0.0; tape.n_states()];
        tape.assemble(&mut scratch, &mut bar, &mut grad, 0.0, y, &[], &[]);
        grad
    }

    #[test]
    fn test_reverse_mul_scalar_scalar() {
        // f = y0 * y1 ; grad = [y1, y0, 0]. y=[2,3,5] -> [3,2,0].
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
        let tape = AdjointTape::new(&arena, f, 3);
        let grad = assemble(&tape, &[2.0, 3.0, 5.0]);
        assert_eq!(grad, vec![3.0, 2.0, 0.0]);
    }

    #[test]
    fn test_reverse_matmul_and_broadcast_sum() {
        // s = y0, v = [y1,y2,y3], p = s*v (ScalarVector broadcast), f = [1,1,1] @ p.
        // grad = [y1+y2+y3, s, s, s], so y = [2,3,4,5] -> [12, 2, 2, 2].
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
        let tape = AdjointTape::new(&arena, f, 4);
        let grad = assemble(&tape, &[2.0, 3.0, 4.0, 5.0]);
        assert_eq!(grad, vec![12.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_reverse_exp_index() {
        // f = exp(y2) ; grad = [0, 0, exp(y2)]. y2=0 -> [0,0,1].
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let y2 = arena.alloc(Node::Index {
            child: y,
            start: 2,
            end: 3,
        });
        let f = arena.alloc(Node::Exp(y2));
        let tape = AdjointTape::new(&arena, f, 3);
        let grad = assemble(&tape, &[1.0, 1.0, 0.0]);
        assert!((grad[2] - 1.0).abs() < 1e-14 && grad[0] == 0.0 && grad[1] == 0.0);
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

        let tape = AdjointTape::new(&arena, root, 1);
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

        let tape = AdjointTape::new(&arena, root, 1);
        let grad = assemble(&tape, &[2.0]);
        assert_eq!(grad, vec![1.0]);
    }

    #[test]
    fn test_pinned_reverse_tape_has_no_self_aliasing_binary() {
        // Regression guard for the eval.rs split_dst_two_src alias panic: the
        // no-reuse layout must never place a binary operand on top of its dst.
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
        let tape = AdjointTape::new(&arena, f, 2);
        for instr in tape.expr_for_test().ir().instructions() {
            if let crate::ir::Instruction::Binary { a, b, dst, len, .. } = *instr {
                let (a, b, dst, len) = (a as usize, b as usize, dst as usize, len as usize);
                assert!(a + len <= dst || dst + len <= a, "operand a aliases dst");
                assert!(b + len <= dst || dst + len <= b, "operand b aliases dst");
            }
        }
    }

    #[test]
    #[should_panic(expected = "primal row")]
    fn test_reverse_tape_rejects_derivative_only_node() {
        // `ReduceArgSelect` is derivative-only: rejected at build time, not
        // mis-differentiated later. Width-one operands let the reject fire first.
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
        let _ = AdjointTape::new(&arena, ras, 2);
    }
}
