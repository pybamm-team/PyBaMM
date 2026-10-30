//! Lane-batched (K-wide) primal evaluator: slot `i` maps to lanes `[i*k, (i+1)*k)`,
//! with results bitwise identical to `k` independent `eval` calls.

// The op helpers must inline as one chain into the interpreter loop.
#![allow(clippy::inline_always)]

use crate::accumulate::{Accumulate, Product, dispatch};
use crate::branch_regions::active_branch;
use crate::eval::{
    CompiledExpr, arg_reduce, csr_rows, interp_cubic_1d, interp_cubic_1d_deriv, interp_linear_1d,
    interp_linear_1d_deriv, interp_nd_lanes, max_reduce, min_reduce, split_read_write,
    split_two_read_write, unary_apply, with_binary_op, with_unary_op, zip_apply,
};
use crate::ir::{BroadcastKind, ConstPool, Instruction, Slot};
use crate::node::CsrData;

/// Error returned when a tape cannot be evaluated by the primal batch path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BatchEvalError {
    #[error("eval_batch is primal-only but the tape contains a tangent load")]
    NonPrimalInstruction,
    #[error("eval_batch does not supply y_dot but the tape references state derivatives")]
    StateDotUnsupported,
}

/// Apply a binary broadcast op over `k` lanes with a monomorphic closure.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn batch_broadcast_apply<F: Fn(f64, f64) -> f64>(
    buf: &mut [f64],
    k: usize,
    f: F,
    a: usize,
    b: usize,
    dst: usize,
    len: usize,
    kind: BroadcastKind,
) {
    match kind {
        // A scalar-scalar op has `len == 1`.
        BroadcastKind::ScalarScalar | BroadcastKind::VectorVector => {
            zip_apply(buf, f, a * k, b * k, dst * k, len * k);
        },
        BroadcastKind::ScalarVector => {
            let (scalar, b_s, d_s) =
                split_two_read_write(buf, (a * k, k), (b * k, len * k), (dst * k, len * k));
            for (d_chunk, b_chunk) in d_s.chunks_exact_mut(k).zip(b_s.chunks_exact(k)) {
                for ((o, &s), &y) in d_chunk.iter_mut().zip(scalar).zip(b_chunk) {
                    *o = f(s, y);
                }
            }
        },
        BroadcastKind::VectorScalar => {
            let (a_s, scalar, d_s) =
                split_two_read_write(buf, (a * k, len * k), (b * k, k), (dst * k, len * k));
            for (d_chunk, a_chunk) in d_s.chunks_exact_mut(k).zip(a_s.chunks_exact(k)) {
                for ((o, &x), &s) in d_chunk.iter_mut().zip(a_chunk).zip(scalar) {
                    *o = f(x, s);
                }
            }
        },
    }
}

/// Fill the front of `dst` with `values` repeated across `lanes` lanes, lane-minor.
#[inline]
pub fn broadcast_lanes(dst: &mut [f64], lanes: usize, values: &[f64]) {
    for (chunk, &value) in dst[..values.len() * lanes]
        .chunks_exact_mut(lanes)
        .zip(values)
    {
        chunk.fill(value);
    }
}

/// Lane `l` of a lane-minor window `k` wide: its elements `l`, `l + k`, `l + 2k`, ...
#[inline(always)]
fn lane(window: &[f64], k: usize, l: usize) -> impl Iterator<Item = f64> {
    window.iter().skip(l).step_by(k).copied()
}

/// The evaluation point of every lane, shared by a whole batched walk.
#[derive(Clone, Copy)]
struct BatchInputs<'a> {
    /// Lane count.
    k: usize,
    ts: &'a [f64],
    /// The `(n_states, k)` column-major state matrix.
    y_cols: &'a [f64],
    n_states: usize,
    inputs: &'a [f64],
}

/// Execute a primal instruction slice against a lane-batched buffer.
///
/// `Dispatch` runs every block any lane selects; `Conditional` still selects per lane.
fn eval_batch_instructions(
    buf: &mut [f64],
    instructions: &[Instruction],
    consts: &ConstPool,
    lanes: BatchInputs<'_>,
) -> Result<(), BatchEvalError> {
    let BatchInputs {
        k,
        ts,
        y_cols,
        n_states,
        inputs,
    } = lanes;
    let mut pc = 0_usize;
    while pc < instructions.len() {
        match instructions[pc] {
            Instruction::Dispatch {
                selector,
                blocks_idx,
                blocks_len,
                span_len,
            } => {
                let n = blocks_len.index();
                let selectors = selector.index() * k..(selector.index() + 1) * k;
                for b in 0..n {
                    // No block writes the selector: the `Conditional` after them still reads it.
                    if buf[selectors.clone()]
                        .iter()
                        .any(|&s| active_branch(s, n) == Some(b))
                    {
                        let block = consts.branch_blocks[blocks_idx.index() + b].bounds(pc);
                        eval_batch_instructions(buf, &instructions[block], consts, lanes)?;
                    }
                }
                pc += span_len.index();
                continue;
            },
            Instruction::LoadScalar { value, dst } => {
                let base = dst.index() * k;
                buf[base..base + k].fill(value);
            },
            Instruction::LoadTime { dst } => {
                let base = dst.index() * k;
                buf[base..base + k].copy_from_slice(&ts[..k]);
            },
            Instruction::LoadArray { data_idx, len, dst } => {
                broadcast_lanes(
                    &mut buf[dst.index() * k..],
                    k,
                    consts.get_array(data_idx, len),
                );
            },
            Instruction::FillZero { dst, len } => {
                let base = dst.index() * k;
                buf[base..base + len.index() * k].fill(0.0);
            },
            Instruction::LoadStateVector { start, width, dst } => {
                let base = dst.index() * k;
                // `y_cols` is the `(n_states, k)` column-major state matrix.
                for e in 0..width.index() {
                    let o = base + e * k;
                    let row = start.index() + e;
                    for l in 0..k {
                        buf[o + l] = y_cols[l * n_states + row];
                    }
                }
            },
            Instruction::LoadStateVectorDot { .. } => {
                return Err(BatchEvalError::StateDotUnsupported);
            },
            Instruction::LoadInputParameter { offset, width, dst } => {
                let values = &inputs[offset.index()..offset.index() + width.index()];
                broadcast_lanes(&mut buf[dst.index() * k..], k, values);
            },
            Instruction::LoadTangentState { .. } | Instruction::LoadTangentParameter { .. } => {
                return Err(BatchEvalError::NonPrimalInstruction);
            },

            Instruction::Binary {
                op,
                a,
                b,
                dst,
                len,
                kind,
            } => {
                let (a, b, dst, len) = (a.index(), b.index(), dst.index(), len.index());
                with_binary_op!(op, f => batch_broadcast_apply(buf, k, f, a, b, dst, len, kind));
            },

            Instruction::Unary { op, src, dst, len } => {
                let (src, dst, len) = (src.index(), dst.index(), len.index());
                with_unary_op!(op, f => unary_apply(buf, f, src * k, dst * k, len * k));
            },

            Instruction::MaxReduce { src, src_len, dst } => {
                let (src, src_len, dst) = (src.index(), src_len.index(), dst.index());
                for l in 0..k {
                    buf[dst * k + l] = max_reduce(lane(&buf[src * k..(src + src_len) * k], k, l));
                }
            },

            Instruction::MinReduce { src, src_len, dst } => {
                let (src, src_len, dst) = (src.index(), src_len.index(), dst.index());
                for l in 0..k {
                    buf[dst * k + l] = min_reduce(lane(&buf[src * k..(src + src_len) * k], k, l));
                }
            },

            Instruction::ReduceArgSelect {
                basis_src,
                picker_src,
                len,
                is_max,
                dst,
            } => {
                let (basis, picker, len) = (basis_src.index(), picker_src.index(), len.index());
                let dbase = dst.index() * k;
                for l in 0..k {
                    let best = arg_reduce(lane(&buf[picker * k..(picker + len) * k], k, l), is_max);
                    buf[dbase + l] = buf[(basis + best) * k + l];
                }
            },

            Instruction::Index {
                src,
                start,
                dst,
                len,
            } => {
                let s = src.offset_by(start) * k;
                let n = len.index() * k;
                buf.copy_within(s..s + n, dst.index() * k);
            },

            Instruction::Concat {
                sources_idx,
                sources_len,
                dst,
            } => {
                let mut write_pos = dst.index() * k;
                for i in 0..sources_len.index() {
                    let slot = consts.concat_sources[sources_idx.index() + i];
                    let n = slot.len_usize() * k;
                    buf.copy_within(
                        slot.offset_usize() * k..slot.offset_usize() * k + n,
                        write_pos,
                    );
                    write_pos += n;
                }
            },

            Instruction::MatMul {
                csr_idx,
                vec_src,
                dst,
            } => {
                let csr = &consts.csr_data[csr_idx.index()];
                let (x, out) = split_read_write(
                    buf,
                    vec_src.index() * k,
                    csr.shape.cols * k,
                    dst.index() * k,
                    csr.shape.rows * k,
                );
                dispatch(CsrLanes { csr, x, out, k });
            },

            Instruction::DenseMatMul {
                mat_src,
                rows,
                cols,
                vec_src,
                dst,
            } => {
                let (rows, cols) = (rows.index(), cols.index());
                let (m, v, out) = split_two_read_write(
                    buf,
                    (mat_src.index() * k, rows * cols * k),
                    (vec_src.index() * k, cols * k),
                    (dst.index() * k, rows * k),
                );
                dispatch(DenseLanes { m, v, out, cols, k });
            },

            Instruction::Interp1DLinear {
                interp_idx,
                src,
                dst,
                len,
            } => {
                let interp = &consts.interpolants[interp_idx.index()];
                let f = |x| interp_linear_1d(interp, x);
                unary_apply(buf, f, src.index() * k, dst.index() * k, len.index() * k);
            },

            Instruction::Interp1DLinearDeriv {
                interp_idx,
                src,
                dst,
                len,
            } => {
                let interp = &consts.interpolants[interp_idx.index()];
                let f = |x| interp_linear_1d_deriv(interp, x);
                unary_apply(buf, f, src.index() * k, dst.index() * k, len.index() * k);
            },

            Instruction::Interp1DCubic {
                interp_idx,
                src,
                dst,
                len,
            } => {
                let interp = &consts.cubic_interpolants[interp_idx.index()];
                let f = |x| interp_cubic_1d(&interp.breakpoints, &interp.coeffs, x);
                unary_apply(buf, f, src.index() * k, dst.index() * k, len.index() * k);
            },

            Instruction::Interp1DCubicDeriv {
                interp_idx,
                src,
                dst,
                len,
            } => {
                let interp = &consts.cubic_interpolants[interp_idx.index()];
                let f = |x| interp_cubic_1d_deriv(&interp.breakpoints, &interp.coeffs, x);
                unary_apply(buf, f, src.index() * k, dst.index() * k, len.index() * k);
            },

            Instruction::InterpNd {
                interp_idx,
                sources_idx,
                dst,
                len,
            } => {
                let out = Slot::new(dst, len);
                interp_nd_lanes(buf, k, consts, interp_idx, sources_idx, None, out);
            },

            Instruction::InterpNdPartial {
                interp_idx,
                sources_idx,
                axis,
                dst,
                len,
            } => {
                let (axis, out) = (Some(axis.index()), Slot::new(dst, len));
                interp_nd_lanes(buf, k, consts, interp_idx, sources_idx, axis, out);
            },

            Instruction::Conditional {
                selector,
                branches_idx,
                branches_len,
                dst,
                out_len,
            } => {
                let sel_base = selector.index() * k;
                let dst = dst.index();
                let out_len = out_len.index();
                for l in 0..k {
                    match active_branch(buf[sel_base + l], branches_len.index()) {
                        Some(i) => {
                            let bbase =
                                consts.branch_offsets[branches_idx.index() + i].offset_usize();
                            for e in 0..out_len {
                                buf[(dst + e) * k + l] = buf[(bbase + e) * k + l];
                            }
                        },
                        None => {
                            for e in 0..out_len {
                                buf[(dst + e) * k + l] = 0.0;
                            }
                        },
                    }
                }
            },
        }
        pc += 1;
    }
    Ok(())
}

impl CompiledExpr {
    /// Evaluate the tape for `k` lanes (time points) at once.
    ///
    /// `scratch` needs `scratch_len() * k` elements; `y_cols` is the `(n_states, k)`
    /// column-major state matrix. Returns the root slot lane-minor (element `e`, lane
    /// `l` at `e*k + l`), bitwise identical to `k` [`eval`](Self::eval) calls.
    ///
    /// # Panics
    /// If `k` is zero or `y_cols` does not hold exactly `n_states * k` values.
    pub fn eval_batch<'s>(
        &self,
        scratch: &'s mut [f64],
        k: usize,
        ts: &[f64],
        y_cols: &[f64],
        n_states: usize,
        inputs: &[f64],
    ) -> Result<&'s [f64], BatchEvalError> {
        let ir = self.ir();
        assert!(k >= 1, "eval_batch needs at least one lane");
        assert_eq!(
            y_cols.len(),
            n_states * k,
            "eval_batch needs n_states = {n_states} values for each of {k} lanes"
        );
        debug_assert!(ts.len() >= k, "ts must have at least k time values");
        debug_assert!(
            scratch.len() >= ir.buffer_size() * k,
            "scratch too small for k lanes"
        );
        let lanes = BatchInputs {
            k,
            ts,
            y_cols,
            n_states,
            inputs,
        };
        eval_batch_instructions(scratch, ir.instructions(), ir.consts(), lanes)?;
        let root = ir.root_slot();
        Ok(&scratch[root.offset_usize() * k..(root.offset_usize() + root.len_usize()) * k])
    }
}

/// `out = A @ x` across `k` lanes, each lane accumulated as the scalar `CsrProduct` does.
struct CsrLanes<'a> {
    csr: &'a CsrData,
    x: &'a [f64],
    out: &'a mut [f64],
    k: usize,
}

impl Product for CsrLanes<'_> {
    type Output = ();

    #[inline(always)]
    fn run<A: Accumulate>(self) {
        let Self { csr, x, out, k } = self;
        for (acc, (cols, vals)) in out.chunks_exact_mut(k).zip(csr_rows(csr)) {
            acc.fill(0.0);
            for (&col, &val) in cols.iter().zip(vals) {
                for (a, &xi) in acc.iter_mut().zip(&x[col * k..][..k]) {
                    *a = A::mul_acc(*a, val, xi);
                }
            }
        }
    }
}

/// `out = M @ v` across `k` lanes for a row-major `M` with `cols` columns.
struct DenseLanes<'a> {
    m: &'a [f64],
    v: &'a [f64],
    out: &'a mut [f64],
    cols: usize,
    k: usize,
}

impl Product for DenseLanes<'_> {
    type Output = ();

    #[inline(always)]
    fn run<A: Accumulate>(self) {
        let Self { m, v, out, cols, k } = self;
        for (row, acc) in out.chunks_exact_mut(k).enumerate() {
            acc.fill(0.0);
            let m_row = &m[row * cols * k..][..cols * k];
            for (m_lanes, v_lanes) in m_row.chunks_exact(k).zip(v.chunks_exact(k)) {
                for ((a, &mi), &vi) in acc.iter_mut().zip(m_lanes).zip(v_lanes) {
                    *a = A::mul_acc(*a, mi, vi);
                }
            }
        }
    }
}
