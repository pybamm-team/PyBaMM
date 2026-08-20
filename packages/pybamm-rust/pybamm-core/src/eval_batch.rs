// Intentional u32 usage for compact instruction storage - expression graphs
// won't exceed 4B nodes in practice
#![allow(clippy::cast_possible_truncation)]

//! Lane-batched (K-wide) evaluator for the primal observe path.
//!
//! [`CompiledExpr::eval_batch`](crate::CompiledExpr::eval_batch) interprets a
//! tape for `k` time points ("lanes") in one pass. The scratch layout is the
//! scalar layout scaled by `k`: scalar buffer index `i` maps to the contiguous
//! lane block `[i*k, (i+1)*k)`. Every handler is then its scalar counterpart with
//! slot offsets scaled by `k` and an inner contiguous loop over lanes, so
//! per-element operations and their order are unchanged and results are
//! **bitwise identical** to `k` independent `eval` calls.
//!
//! Primal instructions only: tangent and state-derivative loads return
//! [`BatchEvalError`] rather than guessing.

use crate::branch_regions::{active_branch, dispatch_span_end};
use crate::eval::{
    CompiledExpr, erf_approx, interp_cubic_1d, interp_cubic_1d_deriv, interp_linear_1d,
    interp_linear_1d_slope_lookup, locate_nd_cell, sign, split_dst_two_src, split_src_dst,
    tensor_horner, tensor_horner_partial,
};
use crate::ir::{BinaryOp, BroadcastKind, ConstPool, Instruction, UnaryOp};

/// Error returned when a tape cannot be evaluated by the primal batch path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BatchEvalError {
    /// The tape contains a tangent load; `eval_batch` is primal-only.
    #[error("eval_batch is primal-only but the tape contains a tangent load")]
    NonPrimalInstruction,
    /// The tape references state derivatives, which the batch path does not supply.
    #[error("eval_batch does not supply y_dot but the tape references state derivatives")]
    StateDotUnsupported,
}

/// Apply a binary broadcast op over `k` lanes with a monomorphic closure.
///
/// Slot offsets are scaled by `k`; each broadcast kind resolves to contiguous
/// `k`-wide slices iterated with a bounds-check-free `zip`, so the closure
/// vectorises exactly as the scalar `broadcast_apply` does. `lane_tmp` (length
/// `k`) holds the broadcast scalar's per-lane values for the mixed kinds, whose
/// operand aliases the destination-disjoint region the borrow checker cannot
/// otherwise split.
#[allow(clippy::inline_always, clippy::too_many_arguments)]
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
    lane_tmp: &mut [f64],
) {
    match kind {
        BroadcastKind::ScalarScalar => {
            let (a_s, b_s, d_s) = split_dst_two_src(buf, a * k, b * k, dst * k, k);
            for ((o, &x), &y) in d_s.iter_mut().zip(a_s).zip(b_s) {
                *o = f(x, y);
            }
        },
        BroadcastKind::VectorVector => {
            let n = len * k;
            let (a_s, b_s, d_s) = split_dst_two_src(buf, a * k, b * k, dst * k, n);
            for ((o, &x), &y) in d_s.iter_mut().zip(a_s).zip(b_s) {
                *o = f(x, y);
            }
        },
        BroadcastKind::ScalarVector => {
            lane_tmp.copy_from_slice(&buf[a * k..a * k + k]);
            let (b_s, d_s) = split_src_dst(buf, b * k, dst * k, len * k);
            for (d_chunk, b_chunk) in d_s.chunks_exact_mut(k).zip(b_s.chunks_exact(k)) {
                for ((o, &s), &y) in d_chunk.iter_mut().zip(lane_tmp.iter()).zip(b_chunk) {
                    *o = f(s, y);
                }
            }
        },
        BroadcastKind::VectorScalar => {
            lane_tmp.copy_from_slice(&buf[b * k..b * k + k]);
            let (a_s, d_s) = split_src_dst(buf, a * k, dst * k, len * k);
            for (d_chunk, a_chunk) in d_s.chunks_exact_mut(k).zip(a_s.chunks_exact(k)) {
                for ((o, &x), &s) in d_chunk.iter_mut().zip(a_chunk).zip(lane_tmp.iter()) {
                    *o = f(x, s);
                }
            }
        },
    }
}

/// Dispatch a binary op to `batch_broadcast_apply` with a monomorphic closure.
/// Must mirror `eval_binary_op` in `eval.rs` closure-for-closure so results
/// stay bitwise identical to the scalar path.
#[allow(clippy::inline_always, clippy::too_many_arguments)]
#[inline(always)]
fn batch_binary(
    buf: &mut [f64],
    k: usize,
    op: BinaryOp,
    a: usize,
    b: usize,
    dst: usize,
    len: usize,
    kind: BroadcastKind,
    lane_tmp: &mut [f64],
) {
    match op {
        BinaryOp::Add => {
            batch_broadcast_apply(buf, k, |x, y| x + y, a, b, dst, len, kind, lane_tmp);
        },
        BinaryOp::Sub => {
            batch_broadcast_apply(buf, k, |x, y| x - y, a, b, dst, len, kind, lane_tmp);
        },
        BinaryOp::Mul => {
            batch_broadcast_apply(buf, k, |x, y| x * y, a, b, dst, len, kind, lane_tmp);
        },
        BinaryOp::Div => {
            batch_broadcast_apply(buf, k, |x, y| x / y, a, b, dst, len, kind, lane_tmp);
        },
        BinaryOp::Pow => batch_broadcast_apply(buf, k, f64::powf, a, b, dst, len, kind, lane_tmp),
        BinaryOp::Minimum => {
            batch_broadcast_apply(buf, k, f64::min, a, b, dst, len, kind, lane_tmp);
        },
        BinaryOp::Maximum => {
            batch_broadcast_apply(buf, k, f64::max, a, b, dst, len, kind, lane_tmp);
        },
        BinaryOp::Modulo => {
            batch_broadcast_apply(buf, k, |x, y| x % y, a, b, dst, len, kind, lane_tmp);
        },
        BinaryOp::Hypot => {
            batch_broadcast_apply(buf, k, f64::hypot, a, b, dst, len, kind, lane_tmp);
        },
        BinaryOp::EqualHeaviside => batch_broadcast_apply(
            buf,
            k,
            |x, y| if x <= y { 1.0 } else { 0.0 },
            a,
            b,
            dst,
            len,
            kind,
            lane_tmp,
        ),
        BinaryOp::NotEqualHeaviside => batch_broadcast_apply(
            buf,
            k,
            |x, y| if x < y { 1.0 } else { 0.0 },
            a,
            b,
            dst,
            len,
            kind,
            lane_tmp,
        ),
        BinaryOp::Equality => {
            const EPS: f64 = 1e-14;
            batch_broadcast_apply(
                buf,
                k,
                |x, y| if (x - y).abs() < EPS { 1.0 } else { 0.0 },
                a,
                b,
                dst,
                len,
                kind,
                lane_tmp,
            );
        },
    }
}

/// Apply a unary op over `len * k` contiguous lane values with a monomorphic
/// closure (vectorises like `unary_apply`); handles the in-place `src == dst`.
#[allow(clippy::inline_always)]
#[inline(always)]
fn batch_unary_apply<F: Fn(f64) -> f64>(
    buf: &mut [f64],
    k: usize,
    f: F,
    src: usize,
    dst: usize,
    len: usize,
) {
    let n = len * k;
    if src == dst {
        for x in &mut buf[dst * k..dst * k + n] {
            *x = f(*x);
        }
        return;
    }
    let (s_s, d_s) = split_src_dst(buf, src * k, dst * k, n);
    for (o, &x) in d_s.iter_mut().zip(s_s) {
        *o = f(x);
    }
}

/// Dispatch a unary op to `batch_unary_apply`. Must mirror `eval_unary_op` in
/// `eval.rs` closure-for-closure so results stay bitwise identical.
#[allow(clippy::inline_always)]
#[inline(always)]
fn batch_unary(buf: &mut [f64], k: usize, op: UnaryOp, src: usize, dst: usize, len: usize) {
    match op {
        UnaryOp::Neg => batch_unary_apply(buf, k, |x| -x, src, dst, len),
        UnaryOp::Abs => batch_unary_apply(buf, k, f64::abs, src, dst, len),
        UnaryOp::Sqrt => batch_unary_apply(buf, k, f64::sqrt, src, dst, len),
        UnaryOp::Exp => batch_unary_apply(buf, k, f64::exp, src, dst, len),
        UnaryOp::Log => batch_unary_apply(buf, k, f64::ln, src, dst, len),
        UnaryOp::Sin => batch_unary_apply(buf, k, f64::sin, src, dst, len),
        UnaryOp::Cos => batch_unary_apply(buf, k, f64::cos, src, dst, len),
        UnaryOp::Tanh => batch_unary_apply(buf, k, f64::tanh, src, dst, len),
        UnaryOp::Sinh => batch_unary_apply(buf, k, f64::sinh, src, dst, len),
        UnaryOp::Cosh => batch_unary_apply(buf, k, f64::cosh, src, dst, len),
        UnaryOp::Arcsinh => batch_unary_apply(buf, k, f64::asinh, src, dst, len),
        UnaryOp::Arctan => batch_unary_apply(buf, k, f64::atan, src, dst, len),
        UnaryOp::Erf => batch_unary_apply(buf, k, erf_approx, src, dst, len),
        UnaryOp::Sign => batch_unary_apply(buf, k, sign, src, dst, len),
        UnaryOp::Floor => batch_unary_apply(buf, k, f64::floor, src, dst, len),
        UnaryOp::Ceiling => batch_unary_apply(buf, k, f64::ceil, src, dst, len),
    }
}

/// Execute a primal instruction slice against a lane-batched buffer.
///
/// `Dispatch` runs the union of blocks any lane's selector picks, skipping a
/// block only when no lane needs it, `Conditional` then still selects the
/// output per lane from within that union.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn eval_batch_instructions(
    buf: &mut [f64],
    k: usize,
    instructions: &[Instruction],
    consts: &ConstPool,
    ts: &[f64],
    y_cols: &[f64],
    n_states: usize,
    inputs: &[f64],
) -> Result<(), BatchEvalError> {
    // Reused per-lane scratch: broadcast-scalar operands and matmul row sums.
    let mut lane_tmp = vec![0.0_f64; k];
    let mut pc = 0_usize;
    while pc < instructions.len() {
        if let Instruction::Dispatch {
            selector,
            blocks_idx,
            blocks_len,
        } = instructions[pc]
        {
            let base = blocks_idx as usize;
            let n = blocks_len as usize;
            let span_end = dispatch_span_end(consts, pc, blocks_idx, blocks_len);
            let sel_base = selector as usize * k;
            // Union over lanes: a block runs if any lane selects it, as the
            // un-blocked tape did, and branch roots are never recycled.
            let mut needed = vec![false; n];
            for l in 0..k {
                if let Some(active) = active_branch(buf[sel_base + l], n) {
                    needed[active] = true;
                }
            }
            for (b, &need) in needed.iter().enumerate() {
                if need {
                    let (rel, len) = consts.branch_blocks[base + b];
                    let start = pc + rel as usize;
                    let end = start + len as usize;
                    eval_batch_instructions(
                        buf,
                        k,
                        &instructions[start..end],
                        consts,
                        ts,
                        y_cols,
                        n_states,
                        inputs,
                    )?;
                }
            }
            pc = span_end;
            continue;
        }

        match instructions[pc] {
            Instruction::LoadScalar { value, dst } => {
                let base = dst as usize * k;
                buf[base..base + k].fill(value);
            },
            Instruction::LoadTime { dst } => {
                let base = dst as usize * k;
                buf[base..base + k].copy_from_slice(&ts[..k]);
            },
            Instruction::LoadArray { data_idx, len, dst } => {
                let src = consts.get_array(data_idx, len);
                let base = dst as usize * k;
                for (e, &v) in src.iter().enumerate() {
                    let o = base + e * k;
                    buf[o..o + k].fill(v);
                }
            },
            Instruction::FillZero { dst, len } => {
                let base = dst as usize * k;
                buf[base..base + len as usize * k].fill(0.0);
            },
            Instruction::LoadStateVector { start, end, dst } => {
                let base = dst as usize * k;
                let slen = (end - start) as usize;
                // Transposing gather: element e, lane l reads column l of the
                // (n_states, k) F-contiguous state matrix.
                for e in 0..slen {
                    let o = base + e * k;
                    let row = start as usize + e;
                    for l in 0..k {
                        buf[o + l] = y_cols[l * n_states + row];
                    }
                }
            },
            Instruction::LoadStateVectorDot { .. } => {
                return Err(BatchEvalError::StateDotUnsupported);
            },
            Instruction::LoadInputParameter { offset, width, dst } => {
                let base = dst as usize * k;
                let off = offset as usize;
                for e in 0..width as usize {
                    let v = inputs[off + e];
                    let o = base + e * k;
                    buf[o..o + k].fill(v);
                }
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
                batch_binary(
                    buf,
                    k,
                    op,
                    a as usize,
                    b as usize,
                    dst as usize,
                    len as usize,
                    kind,
                    &mut lane_tmp,
                );
            },

            Instruction::Unary { op, src, dst, len } => {
                batch_unary(buf, k, op, src as usize, dst as usize, len as usize);
            },

            Instruction::MaxReduce { src, src_len, dst } => {
                let (src, src_len) = (src as usize, src_len as usize);
                let dbase = dst as usize * k;
                for l in 0..k {
                    let mut max_val = f64::NEG_INFINITY;
                    for e in 0..src_len {
                        let v = buf[(src + e) * k + l];
                        if v > max_val {
                            max_val = v;
                        }
                    }
                    buf[dbase + l] = max_val;
                }
            },

            Instruction::MinReduce { src, src_len, dst } => {
                let (src, src_len) = (src as usize, src_len as usize);
                let dbase = dst as usize * k;
                for l in 0..k {
                    let mut min_val = f64::INFINITY;
                    for e in 0..src_len {
                        let v = buf[(src + e) * k + l];
                        if v < min_val {
                            min_val = v;
                        }
                    }
                    buf[dbase + l] = min_val;
                }
            },

            Instruction::ReduceArgSelect {
                basis_src,
                picker_src,
                len,
                is_max,
                dst,
            } => {
                let (basis, picker, len) = (basis_src as usize, picker_src as usize, len as usize);
                let dbase = dst as usize * k;
                // Per lane: first-occurrence argmax/argmin with a strict
                // comparison from element 0, matching the scalar eval.
                for l in 0..k {
                    let mut best_idx = 0;
                    let mut best_val = buf[picker * k + l];
                    for e in 1..len {
                        let v = buf[(picker + e) * k + l];
                        if (is_max && v > best_val) || (!is_max && v < best_val) {
                            best_val = v;
                            best_idx = e;
                        }
                    }
                    buf[dbase + l] = buf[(basis + best_idx) * k + l];
                }
            },

            Instruction::Index {
                src,
                start,
                dst,
                len,
            } => {
                // Slot elements [start, start+len) map to contiguous lane blocks,
                // so the whole window is one contiguous copy (memmove-safe).
                let s = (src as usize + start as usize) * k;
                let n = len as usize * k;
                buf.copy_within(s..s + n, dst as usize * k);
            },

            Instruction::Concat {
                sources_idx,
                sources_len,
                dst,
            } => {
                let mut write_pos = dst as usize * k;
                for i in 0..sources_len as usize {
                    let (src_off, src_len) = consts.concat_sources[sources_idx as usize + i];
                    let n = src_len as usize * k;
                    buf.copy_within(src_off as usize * k..src_off as usize * k + n, write_pos);
                    write_pos += n;
                }
            },

            Instruction::MatMul {
                csr_idx,
                vec_src,
                dst,
            } => {
                let csr = &consts.csr_data[csr_idx as usize];
                let (vec_src, dst) = (vec_src as usize, dst as usize);
                for row in 0..csr.shape.rows {
                    // `lane_tmp` is disjoint from `buf` so the inner lane loop
                    // vectorises; entries accumulate in scalar SpMV order.
                    lane_tmp.fill(0.0);
                    let (start, end) = (csr.indptr[row], csr.indptr[row + 1]);
                    for (&col, &val) in csr.indices[start..end].iter().zip(&csr.data[start..end]) {
                        let cbase = (vec_src + col) * k;
                        let x = &buf[cbase..cbase + k];
                        for (acc, &xi) in lane_tmp.iter_mut().zip(x) {
                            *acc += val * xi;
                        }
                    }
                    let rbase = (dst + row) * k;
                    buf[rbase..rbase + k].copy_from_slice(&lane_tmp);
                }
            },

            Instruction::DenseMatMul {
                mat_src,
                rows,
                cols,
                vec_src,
                dst,
            } => {
                let (mat_src, vec_src, dst) = (mat_src as usize, vec_src as usize, dst as usize);
                let cols = cols as usize;
                for row in 0..rows as usize {
                    lane_tmp.fill(0.0);
                    for col in 0..cols {
                        let mbase = (mat_src + row * cols + col) * k;
                        let vbase = (vec_src + col) * k;
                        let m = &buf[mbase..mbase + k];
                        let v = &buf[vbase..vbase + k];
                        for ((acc, &mi), &vi) in lane_tmp.iter_mut().zip(m).zip(v) {
                            *acc += mi * vi;
                        }
                    }
                    let rbase = (dst + row) * k;
                    buf[rbase..rbase + k].copy_from_slice(&lane_tmp);
                }
            },

            Instruction::Interp1DLinear {
                interp_idx,
                src,
                dst,
                len,
            } => {
                let interp = &consts.interpolants[interp_idx as usize];
                let (src, dst) = (src as usize, dst as usize);
                for e in 0..len as usize {
                    let (sb, db) = ((src + e) * k, (dst + e) * k);
                    for l in 0..k {
                        buf[db + l] = interp_linear_1d(&interp.x_data, &interp.y_data, buf[sb + l]);
                    }
                }
            },

            Instruction::Interp1DLinearDeriv {
                interp_idx,
                src,
                dst,
                len,
            } => {
                let interp = &consts.interpolants[interp_idx as usize];
                let (src, dst) = (src as usize, dst as usize);
                for e in 0..len as usize {
                    let (sb, db) = ((src + e) * k, (dst + e) * k);
                    for l in 0..k {
                        buf[db + l] = interp_linear_1d_slope_lookup(
                            &interp.x_data,
                            &interp.y_data,
                            buf[sb + l],
                        );
                    }
                }
            },

            Instruction::Interp1DCubic {
                interp_idx,
                src,
                dst,
                len,
            } => {
                let interp = &consts.cubic_interpolants[interp_idx as usize];
                let (src, dst) = (src as usize, dst as usize);
                for e in 0..len as usize {
                    let (sb, db) = ((src + e) * k, (dst + e) * k);
                    for l in 0..k {
                        buf[db + l] =
                            interp_cubic_1d(&interp.breakpoints, &interp.coeffs, buf[sb + l]);
                    }
                }
            },

            Instruction::Interp1DCubicDeriv {
                interp_idx,
                src,
                dst,
                len,
            } => {
                let interp = &consts.cubic_interpolants[interp_idx as usize];
                let (src, dst) = (src as usize, dst as usize);
                for e in 0..len as usize {
                    let (sb, db) = ((src + e) * k, (dst + e) * k);
                    for l in 0..k {
                        buf[db + l] =
                            interp_cubic_1d_deriv(&interp.breakpoints, &interp.coeffs, buf[sb + l]);
                    }
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
                for e in 0..len as usize {
                    for l in 0..k {
                        for (a, coord) in coords.iter_mut().enumerate().take(ndim) {
                            let (off, slen) = consts.interp_nd_sources[sources_idx as usize + a];
                            let j = if slen == 1 { 0 } else { e };
                            *coord = buf[(off as usize + j) * k + l];
                        }
                        let cell = locate_nd_cell(
                            &interp.breakpoints,
                            &interp.coeffs,
                            order,
                            &coords[..ndim],
                            &mut dxs,
                        );
                        buf[(dst + e) * k + l] = tensor_horner(cell, &dxs[..ndim], order);
                    }
                }
            },

            Instruction::InterpNdPartial {
                interp_idx,
                sources_idx,
                axis,
                dst,
                len,
            } => {
                let interp = &consts.nd_interpolants[interp_idx as usize];
                let ndim = interp.breakpoints.len();
                let order = interp.order as usize;
                let dst = dst as usize;
                let mut coords = [0.0_f64; 3];
                let mut dxs = [0.0_f64; 3];
                for e in 0..len as usize {
                    for l in 0..k {
                        for (a, coord) in coords.iter_mut().enumerate().take(ndim) {
                            let (off, slen) = consts.interp_nd_sources[sources_idx as usize + a];
                            let j = if slen == 1 { 0 } else { e };
                            *coord = buf[(off as usize + j) * k + l];
                        }
                        let cell = locate_nd_cell(
                            &interp.breakpoints,
                            &interp.coeffs,
                            order,
                            &coords[..ndim],
                            &mut dxs,
                        );
                        buf[(dst + e) * k + l] =
                            tensor_horner_partial(cell, &dxs[..ndim], order, axis as usize);
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
                let sel_base = selector as usize * k;
                let dst = dst as usize;
                let out_len = out_len as usize;
                // Each lane selects its branch independently; branch offsets are
                // slot offsets, scaled by k at the copy site.
                for l in 0..k {
                    match active_branch(buf[sel_base + l], branches_len as usize) {
                        Some(i) => {
                            let (branch_off, _) = consts.branch_offsets[branches_idx as usize + i];
                            let bbase = branch_off as usize;
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

            Instruction::Dispatch { .. } => unreachable!("handled above"),
        }
        pc += 1;
    }
    Ok(())
}

impl CompiledExpr {
    /// Evaluate the tape for `k` lanes (time points) at once.
    ///
    /// `scratch` must hold at least `scratch_len() * k` elements. Slot `s` occupies
    /// `[s*k, (s + slot_len)*k)`, and element `e` of it holds its `k` lane values
    /// contiguously at `[(s+e)*k, (s+e+1)*k)`. `ts` supplies the `k` time values,
    /// `y_cols` is the `(n_states, k)` F-contiguous state matrix, and `inputs` is
    /// shared across lanes. Returns the root slot as `(out_len, k)` lane-minor:
    /// element `e`, lane `l` at relative index `e*k + l`.
    ///
    /// Results are bitwise identical to `k` independent [`eval`](Self::eval) calls.
    /// Primal-only: a tangent or state-derivative load returns [`BatchEvalError`].
    pub fn eval_batch<'s>(
        &self,
        scratch: &'s mut [f64],
        k: usize,
        ts: &[f64],
        y_cols: &[f64],
        inputs: &[f64],
    ) -> Result<&'s [f64], BatchEvalError> {
        let ir = self.ir();
        debug_assert!(
            ir.split_eval_info().is_none(),
            "eval_batch requires a primal (non-split-eval) tape"
        );
        debug_assert!(k >= 1, "eval_batch needs at least one lane");
        debug_assert!(ts.len() >= k, "ts must have at least k time values");
        debug_assert!(
            scratch.len() >= ir.buffer_size() * k,
            "scratch too small for k lanes"
        );
        let n_states = y_cols.len() / k;
        eval_batch_instructions(
            scratch,
            k,
            ir.instructions(),
            ir.consts(),
            ts,
            y_cols,
            n_states,
            inputs,
        )?;
        let root = ir.root_slot();
        Ok(&scratch[root.offset_usize() * k..(root.offset_usize() + root.len_usize()) * k])
    }
}

#[cfg(test)]
mod tests {
    // Test-fixture arithmetic favours readable data generation over FMA.
    #![allow(clippy::suboptimal_flops)]
    use super::*;
    use crate::arena::{Arena, NodeId};
    use crate::node::{CsrData, InterpolantData, Node, Shape};

    /// Evaluate `k` lanes of random `(t, y)` through both the scalar `eval`
    /// (once per lane) and `eval_batch`, asserting bitwise equality.
    fn assert_batch_matches_scalar(
        arena: &Arena,
        root: NodeId,
        n_states: usize,
        k: usize,
        ts: &[f64],
        y_cols: &[f64],
        inputs: &[f64],
    ) {
        let expr = CompiledExpr::new(arena, root);
        let out_len = expr.output_len();

        // Scalar reference: one eval per lane.
        let mut scalar = vec![0.0_f64; out_len * k];
        let mut s = vec![0.0_f64; expr.scratch_len()];
        for l in 0..k {
            let y = &y_cols[l * n_states..(l + 1) * n_states];
            let res = expr.eval(&mut s, ts[l], y, &[], inputs);
            scalar[l * out_len..(l + 1) * out_len].copy_from_slice(res);
        }

        // Batched: one eval_batch over all lanes.
        let mut batch_scratch = vec![0.0_f64; expr.scratch_len() * k];
        let root_slice = expr
            .eval_batch(&mut batch_scratch, k, ts, y_cols, inputs)
            .expect("primal tape must batch-evaluate");
        for l in 0..k {
            for e in 0..out_len {
                let got = root_slice[e * k + l];
                let want = scalar[l * out_len + e];
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "lane {l}, elem {e}: batch {got} != scalar {want}"
                );
            }
        }
    }

    /// Build a column-major `(n_states, k)` state matrix from a closure.
    fn state_cols(n_states: usize, k: usize, f: impl Fn(usize, usize) -> f64) -> Vec<f64> {
        let mut cols = vec![0.0_f64; n_states * k];
        for l in 0..k {
            for i in 0..n_states {
                cols[l * n_states + i] = f(i, l);
            }
        }
        cols
    }

    #[test]
    fn all_binary_ops_match_scalar() {
        // Every BinaryOp variant, guarding drift in the duplicated op table.
        let ops = [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Pow,
            BinaryOp::Minimum,
            BinaryOp::Maximum,
            BinaryOp::Modulo,
            BinaryOp::Hypot,
            BinaryOp::EqualHeaviside,
            BinaryOp::NotEqualHeaviside,
            BinaryOp::Equality,
        ];
        let k = 7;
        let ts: Vec<f64> = (0..k).map(|l| l as f64 * 0.3).collect();
        let y_cols = state_cols(2, k, |i, l| 0.5 + i as f64 + l as f64 * 0.11);
        for op in ops {
            let mut arena = Arena::new();
            let a = arena.alloc(Node::StateVector { start: 0, end: 1 });
            let b = arena.alloc(Node::StateVector { start: 1, end: 2 });
            let root = alloc_binary(&mut arena, op, a, b);
            assert_batch_matches_scalar(&arena, root, 2, k, &ts, &y_cols, &[]);
        }
    }

    #[test]
    fn all_unary_ops_match_scalar() {
        let ops = [
            UnaryOp::Neg,
            UnaryOp::Abs,
            UnaryOp::Sqrt,
            UnaryOp::Exp,
            UnaryOp::Log,
            UnaryOp::Sin,
            UnaryOp::Cos,
            UnaryOp::Tanh,
            UnaryOp::Sinh,
            UnaryOp::Cosh,
            UnaryOp::Arcsinh,
            UnaryOp::Arctan,
            UnaryOp::Erf,
            UnaryOp::Sign,
            UnaryOp::Floor,
            UnaryOp::Ceiling,
        ];
        let k = 5;
        let ts: Vec<f64> = (0..k).map(|l| l as f64 * 0.2).collect();
        // Positive inputs so Sqrt/Log stay in-domain; varied per lane.
        let y_cols = state_cols(1, k, |_, l| 0.3 + l as f64 * 0.37);
        for op in ops {
            let mut arena = Arena::new();
            let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
            let root = alloc_unary(&mut arena, op, x);
            assert_batch_matches_scalar(&arena, root, 1, k, &ts, &y_cols, &[]);
        }
    }

    #[test]
    fn broadcast_scalar_vector_per_lane() {
        // (scalar y0) * (vector y1..y4): ScalarVector with a per-lane scalar.
        let mut arena = Arena::new();
        let s = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let v = arena.alloc(Node::StateVector { start: 1, end: 4 });
        let root = arena.alloc(Node::Mul(s, v));
        let k = 4;
        let ts = vec![0.0; k];
        let y_cols = state_cols(4, k, |i, l| 1.0 + i as f64 * 0.5 + l as f64);
        assert_batch_matches_scalar(&arena, root, 4, k, &ts, &y_cols, &[]);
    }

    #[test]
    fn conditional_selects_different_branches_per_lane() {
        // selector = y0 (1.0 or 2.0 per lane) picks branch 1 or 2.
        let mut arena = Arena::new();
        let selector = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let y2 = arena.alloc(Node::StateVector { start: 2, end: 3 });
        let two = arena.alloc(Node::Scalar(2.0));
        let three = arena.alloc(Node::Scalar(3.0));
        let b1 = arena.alloc(Node::Add(y1, two));
        let b2 = arena.alloc(Node::Mul(y2, three));
        let root = arena.alloc(Node::Conditional {
            selector,
            branches: vec![b1, b2],
        });
        let k = 4;
        let ts = vec![0.0; k];
        // Alternate selector 1.0 / 2.0 so lanes take different branches, plus a
        // lane whose selector matches nothing (0.0 -> zero-filled).
        let y_cols = state_cols(3, k, |i, l| match i {
            0 => {
                if l == 3 {
                    0.0
                } else {
                    (l % 2 + 1) as f64
                }
            },
            _ => i as f64 + l as f64 * 0.25,
        });
        assert_batch_matches_scalar(&arena, root, 3, k, &ts, &y_cols, &[]);
    }

    #[test]
    fn reduce_argselect_tie_breaking_per_lane() {
        // pybamm.max subgradient: basis[argmax(picker)] with first-occurrence
        // ties, exercised with per-lane tie positions.
        let mut arena = Arena::new();
        let picker = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let root = arena.alloc(Node::MaxReduce(picker));
        let k = 3;
        let ts = vec![0.0; k];
        // Lane 0: [5,5,3] (tie at 0), lane 1: [1,5,5] (tie at 1), lane 2: strict.
        let picks = [[5.0, 5.0, 3.0], [1.0, 5.0, 5.0], [1.0, 2.0, 3.0]];
        let y_cols = state_cols(3, k, |i, l| picks[l][i]);
        assert_batch_matches_scalar(&arena, root, 3, k, &ts, &y_cols, &[]);
    }

    #[test]
    fn sparse_matmul_matches_scalar() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 4 });
        let mat = arena.alloc(Node::SparseMatrix(Box::new(
            CsrData::try_new(
                vec![0, 2, 4],
                vec![0, 1, 2, 3],
                vec![1.0, -1.0, 2.0, 3.0],
                Shape::matrix(2, 4),
            )
            .expect("valid matrix"),
        )));
        let root = arena.alloc(Node::MatMul(mat, y));
        let k = 6;
        let ts = vec![0.0; k];
        let y_cols = state_cols(4, k, |i, l| 0.5 + i as f64 + l as f64 * 0.13);
        assert_batch_matches_scalar(&arena, root, 4, k, &ts, &y_cols, &[]);
    }

    #[test]
    fn interpolant_matches_scalar() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let root = arena.alloc(Node::Interpolant1DLinear {
            data: Box::new(
                InterpolantData::try_new(vec![0.0, 1.0, 2.0], vec![0.0, 10.0, 25.0])
                    .expect("valid interpolant"),
            ),
            child: x,
        });
        let k = 5;
        let ts = vec![0.0; k];
        // Spread across, below, and above the knot range to hit extrapolation.
        let y_cols = state_cols(1, k, |_, l| -0.5 + l as f64 * 0.7);
        assert_batch_matches_scalar(&arena, root, 1, k, &ts, &y_cols, &[]);
    }

    #[test]
    fn ragged_tail_k1_matches_scalar() {
        // k == 1 is the ragged-tail degenerate case (compact stride 1).
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let root = arena.alloc(Node::Mul(y, two));
        let ts = [0.4];
        let y_cols = [1.5, -2.5];
        assert_batch_matches_scalar(&arena, root, 2, 1, &ts, &y_cols, &[]);
    }

    #[test]
    fn time_and_concat_match_scalar() {
        // Concat([t, y0]) exercises LoadTime + Concat over lanes.
        let mut arena = Arena::new();
        let t = arena.alloc(Node::Time);
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let root = arena.alloc(Node::Concat(vec![t, y0]));
        let k = 4;
        let ts: Vec<f64> = (0..k).map(|l| 0.1 + l as f64 * 0.9).collect();
        let y_cols = state_cols(1, k, |_, l| l as f64 * 2.0 - 1.0);
        assert_batch_matches_scalar(&arena, root, 1, k, &ts, &y_cols, &[]);
    }

    #[test]
    fn tangent_load_returns_error() {
        // A split-eval (tangent) tape must be rejected, not silently mis-evaluated.
        use crate::tangent::tangent_wrt_states;
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let two = arena.alloc(Node::Scalar(2.0));
        let root = arena.alloc(Node::Mul(y, two));
        let troot = tangent_wrt_states(&mut arena, root);
        let expr = CompiledExpr::new(&arena, troot);
        let mut scratch = vec![0.0_f64; expr.scratch_len() * 2];
        let err = expr.eval_batch(&mut scratch, 2, &[0.0, 0.0], &[1.0, 2.0], &[]);
        assert_eq!(err, Err(BatchEvalError::NonPrimalInstruction));
    }

    #[test]
    fn dispatch_skips_blocks_no_lane_selects() {
        // Selector is an InputParameter, so all lanes agree: only branch 2's
        // block may run, and the result must match the scalar path bitwise.
        let mut arena = Arena::new();
        let selector = arena.alloc(Node::InputParameter {
            name: "s".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let b1 = arena.alloc(Node::Sin(y));
        let mut b2 = y;
        for _ in 0..6 {
            b2 = arena.alloc(Node::Exp(b2));
        }
        let root = arena.alloc(Node::Conditional {
            selector,
            branches: vec![b1, b2],
        });
        let k = 4;
        let ts = vec![0.0; k];
        let y_cols = state_cols(1, k, |_, l| 0.1 + l as f64 * 0.05);
        for sel in [1.0_f64, 2.0, 0.0] {
            assert_batch_matches_scalar(&arena, root, 1, k, &ts, &y_cols, &[sel]);
        }
    }

    #[test]
    fn dispatch_falls_back_to_the_union_on_lane_divergence() {
        // y-derived selector: lanes pick different branches, so both blocks must
        // run. Bitwise parity with per-lane scalar eval is the assertion.
        let mut arena = Arena::new();
        let selector = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let y2 = arena.alloc(Node::StateVector { start: 2, end: 3 });
        let b1 = arena.alloc(Node::Sin(y1));
        let b2 = arena.alloc(Node::Exp(y2));
        let root = arena.alloc(Node::Conditional {
            selector,
            branches: vec![b1, b2],
        });
        let k = 4;
        let ts = vec![0.0; k];
        // Lanes 0..2 alternate selector 1/2; lane 3 matches nothing.
        let y_cols = state_cols(3, k, |i, l| match i {
            0 => {
                if l == 3 {
                    0.0
                } else {
                    (l % 2 + 1) as f64
                }
            },
            _ => i as f64 + l as f64 * 0.25,
        });
        assert_batch_matches_scalar(&arena, root, 3, k, &ts, &y_cols, &[]);
    }

    #[test]
    fn dispatch_skips_unneeded_block_even_when_it_would_error() {
        // Branch 2 loads a state derivative, which eval_batch rejects whenever it
        // is evaluated, so this passes only if branch 2's block is truly skipped.
        let mut arena = Arena::new();
        let selector = arena.alloc(Node::InputParameter {
            name: "s".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let b1 = arena.alloc(Node::Sin(y));
        let y_dot = arena.alloc(Node::StateVectorDot { start: 0, end: 1 });
        let b2 = arena.alloc(Node::Exp(y_dot));
        let root = arena.alloc(Node::Conditional {
            selector,
            branches: vec![b1, b2],
        });
        let expr = CompiledExpr::new(&arena, root);
        let k = 4;
        let ts = vec![0.0; k];
        let y_cols = state_cols(1, k, |_, l| 0.1 + l as f64 * 0.05);
        let mut scratch = vec![0.0_f64; expr.scratch_len() * k];
        let result = expr.eval_batch(&mut scratch, k, &ts, &y_cols, &[1.0]);
        assert!(
            result.is_ok(),
            "branch 2 must never run when no lane selects it: {result:?}"
        );
    }

    #[test]
    fn two_dispatches_in_one_tape_match_scalar() {
        // Two independent Conditionals lower to two Dispatch instructions, so the
        // union-dispatch loop must resume after the first to find the second.
        let mut arena = Arena::new();
        let sel1 = arena.alloc(Node::InputParameter {
            name: "s1".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let sel2 = arena.alloc(Node::InputParameter {
            name: "s2".to_string(),
            index: 1,
            offset: 1,
            width: 1,
        });
        let y1 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y2 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let a1 = arena.alloc(Node::Sin(y1));
        let a2 = arena.alloc(Node::Cos(y1));
        let cond1 = arena.alloc(Node::Conditional {
            selector: sel1,
            branches: vec![a1, a2],
        });
        let b1 = arena.alloc(Node::Exp(y2));
        let b2 = arena.alloc(Node::Sqrt(y2));
        let cond2 = arena.alloc(Node::Conditional {
            selector: sel2,
            branches: vec![b1, b2],
        });
        let root = arena.alloc(Node::Add(cond1, cond2));
        let expr = CompiledExpr::new(&arena, root);
        assert_eq!(
            expr.ir().branch_block_lens().len(),
            4,
            "two two-branch dispatches, or this test covers nothing"
        );
        let k = 4;
        let ts = vec![0.0; k];
        let y_cols = state_cols(2, k, |i, l| match i {
            0 => 0.2 + l as f64 * 0.1,
            _ => 0.5 + l as f64 * 0.2,
        });
        for sels in [[1.0, 1.0], [2.0, 2.0], [1.0, 2.0], [0.0, 1.0]] {
            assert_batch_matches_scalar(&arena, root, 2, k, &ts, &y_cols, &sels);
        }
    }

    #[test]
    fn dispatch_on_boundary_and_nan_selectors_matches_scalar() {
        // The semantics contract's edge cases: half-integer window boundaries,
        // NaN, negative and infinite selectors all match no branch.
        let mut arena = Arena::new();
        let selector = arena.alloc(Node::InputParameter {
            name: "s".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let b1 = arena.alloc(Node::Sin(y));
        let b2 = arena.alloc(Node::Cos(y));
        let root = arena.alloc(Node::Conditional {
            selector,
            branches: vec![b1, b2],
        });
        let k = 4;
        let ts = vec![0.0; k];
        let y_cols = state_cols(1, k, |_, l| 0.1 + l as f64 * 0.05);
        for sel in [0.5_f64, 1.5, 2.5, -1.0, f64::NAN, f64::INFINITY] {
            assert_batch_matches_scalar(&arena, root, 1, k, &ts, &y_cols, &[sel]);
        }

        // And per-lane: one weird selector per lane, so the union is empty for
        // some lanes while others still pick a branch.
        let lane_selector = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let lb1 = arena.alloc(Node::Sin(y1));
        let lb2 = arena.alloc(Node::Cos(y1));
        let lane_root = arena.alloc(Node::Conditional {
            selector: lane_selector,
            branches: vec![lb1, lb2],
        });
        let weird = [f64::NAN, 0.5, 1.5, 2.5, -1.0, f64::INFINITY, 1.0, 2.0];
        let kw = weird.len();
        let y_cols = state_cols(2, kw, |i, l| if i == 0 { weird[l] } else { 0.3 });
        assert_batch_matches_scalar(&arena, lane_root, 2, kw, &vec![0.0; kw], &y_cols, &[]);
    }

    fn alloc_binary(arena: &mut Arena, op: BinaryOp, a: NodeId, b: NodeId) -> NodeId {
        match op {
            BinaryOp::Add => arena.alloc(Node::Add(a, b)),
            BinaryOp::Sub => arena.alloc(Node::Sub(a, b)),
            BinaryOp::Mul => arena.alloc(Node::Mul(a, b)),
            BinaryOp::Div => arena.alloc(Node::Div(a, b)),
            BinaryOp::Pow => arena.alloc(Node::Pow(a, b)),
            BinaryOp::Minimum => arena.alloc(Node::Minimum(a, b)),
            BinaryOp::Maximum => arena.alloc(Node::Maximum(a, b)),
            BinaryOp::Modulo => arena.alloc(Node::Modulo(a, b)),
            BinaryOp::Hypot => arena.alloc(Node::Hypot(a, b)),
            BinaryOp::EqualHeaviside => arena.alloc(Node::EqualHeaviside(a, b)),
            BinaryOp::NotEqualHeaviside => arena.alloc(Node::NotEqualHeaviside(a, b)),
            BinaryOp::Equality => arena.alloc(Node::Equality(a, b)),
        }
    }

    fn alloc_unary(arena: &mut Arena, op: UnaryOp, x: NodeId) -> NodeId {
        match op {
            UnaryOp::Neg => arena.alloc(Node::Neg(x)),
            UnaryOp::Abs => arena.alloc(Node::Abs(x)),
            UnaryOp::Sqrt => arena.alloc(Node::Sqrt(x)),
            UnaryOp::Exp => arena.alloc(Node::Exp(x)),
            UnaryOp::Log => arena.alloc(Node::Log(x)),
            UnaryOp::Sin => arena.alloc(Node::Sin(x)),
            UnaryOp::Cos => arena.alloc(Node::Cos(x)),
            UnaryOp::Tanh => arena.alloc(Node::Tanh(x)),
            UnaryOp::Sinh => arena.alloc(Node::Sinh(x)),
            UnaryOp::Cosh => arena.alloc(Node::Cosh(x)),
            UnaryOp::Arcsinh => arena.alloc(Node::Arcsinh(x)),
            UnaryOp::Arctan => arena.alloc(Node::Arctan(x)),
            UnaryOp::Erf => arena.alloc(Node::Erf(x)),
            UnaryOp::Sign => arena.alloc(Node::Sign(x)),
            UnaryOp::Floor => arena.alloc(Node::Floor(x)),
            UnaryOp::Ceiling => arena.alloc(Node::Ceiling(x)),
        }
    }
}
