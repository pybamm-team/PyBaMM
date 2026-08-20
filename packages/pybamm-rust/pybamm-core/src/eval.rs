// Intentional u32 usage for compact instruction storage - expression graphs
// won't exceed 4B nodes in practice
#![allow(clippy::cast_possible_truncation)]

//! Expression evaluator using `TypedIr`.
//!
//! `CompiledExpr` wraps a `TypedIr` and provides efficient evaluation against a
//! caller-supplied scratch buffer. It also supports evaluation with tangent
//! inputs for forward-mode automatic differentiation.

use crate::arena::{Arena, NodeId};
use crate::branch_regions::{active_branch, dispatch_span_end};
use crate::ir::{BinaryOp, BroadcastKind, ConstPool, Instruction, TypedIr, UnaryOp};

/// Linear interpolation with binary search. Extends the boundary segment
/// linearly outside the data domain (no flat clamp).
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn interp_linear_1d(x_data: &[f64], y_data: &[f64], x: f64) -> f64 {
    let n = x_data.len();
    if n == 1 {
        return y_data[0];
    }
    // Select the segment [lo, lo+1]: clamp to the first/last segment for
    // out-of-domain x so the boundary line is extended.
    let lo = if x <= x_data[0] {
        0
    } else if x >= x_data[n - 1] {
        n - 2
    } else {
        let mut lo = 0;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = usize::midpoint(lo, hi);
            if x_data[mid] <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };
    let t = (x - x_data[lo]) / (x_data[lo + 1] - x_data[lo]);
    t.mul_add(y_data[lo + 1] - y_data[lo], y_data[lo])
}

/// Pre-computed slope lookup for linear-interpolation derivative (extends
/// boundary segment outside data domain, matching the value function).
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn interp_linear_1d_slope_lookup(x_data: &[f64], slopes: &[f64], x: f64) -> f64 {
    let n = x_data.len();
    if n < 2 || slopes.is_empty() {
        return 0.0;
    }
    if x <= x_data[0] {
        return slopes[0];
    }
    if x >= x_data[n - 1] {
        return slopes[n - 2];
    }
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = usize::midpoint(lo, hi);
        if x_data[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    slopes[lo]
}

/// Local slope of a piecewise-linear interpolant at `x`, computed from the
/// primal knot values. Matches `interp_linear_1d_slope_lookup`'s segment
/// choice and `compute_interpolant_slopes`' value, so the reverse-AD adjoint
/// equals the forward derivative path exactly.
pub(crate) fn interp_linear_1d_deriv(x_data: &[f64], y_data: &[f64], x: f64) -> f64 {
    let n = x_data.len();
    if n < 2 {
        return 0.0;
    }
    let seg = if x <= x_data[0] {
        0
    } else if x >= x_data[n - 1] {
        n - 2
    } else {
        let mut lo = 0;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = usize::midpoint(lo, hi);
            if x_data[mid] <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };
    let dx = x_data[seg + 1] - x_data[seg];
    if dx.abs() > f64::EPSILON {
        (y_data[seg + 1] - y_data[seg]) / dx
    } else {
        0.0
    }
}

/// Interval index for `x`, clamped to `[0, nseg-1]` (extends boundary polynomial).
/// `breakpoints.len() >= 2` is guaranteed by Python lowering.
#[allow(clippy::inline_always)]
#[inline(always)]
fn locate_cubic_interval(breakpoints: &[f64], x: f64) -> usize {
    debug_assert!(breakpoints.len() >= 2);
    let nseg = breakpoints.len() - 1;
    if x <= breakpoints[0] {
        return 0;
    }
    if x >= breakpoints[nseg] {
        return nseg - 1;
    }
    let mut lo = 0;
    let mut hi = nseg;
    while hi - lo > 1 {
        let mid = usize::midpoint(lo, hi);
        if breakpoints[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Piecewise-cubic interpolation (cubic spline / pchip), power-basis Horner.
#[allow(clippy::inline_always, clippy::suboptimal_flops)]
#[inline(always)]
pub(crate) fn interp_cubic_1d(breakpoints: &[f64], coeffs: &[[f64; 4]], x: f64) -> f64 {
    let i = locate_cubic_interval(breakpoints, x);
    let dx = x - breakpoints[i];
    let [c0, c1, c2, c3] = coeffs[i];
    c0 + (c1 + (c2 + c3 * dx) * dx) * dx
}

/// Derivative of the piecewise-cubic interpolant: `p'(dx) = c1 + 2*c2*dx + 3*c3*dx^2`.
#[allow(clippy::inline_always, clippy::suboptimal_flops)]
#[inline(always)]
pub(crate) fn interp_cubic_1d_deriv(breakpoints: &[f64], coeffs: &[[f64; 4]], x: f64) -> f64 {
    let i = locate_cubic_interval(breakpoints, x);
    let dx = x - breakpoints[i];
    let [_c0, c1, c2, c3] = coeffs[i];
    c1 + (2.0 * c2 + 3.0 * c3 * dx) * dx
}

/// Locate the N-D cell for `coords`, writing per-axis offsets into `dxs` and
/// returning the cell's `order^ndim` power-basis coefficients (clamped per
/// axis, matching scipy `RegularGridInterpolator` with `fill_value=None`).
pub(crate) fn locate_nd_cell<'a>(
    breakpoints: &[Vec<f64>],
    coeffs: &'a [f64],
    order: usize,
    coords: &[f64],
    dxs: &mut [f64; 3],
) -> &'a [f64] {
    let ndim = breakpoints.len();
    debug_assert!((2..=3).contains(&ndim) && coords.len() == ndim);
    let mut cell = 0;
    for a in 0..ndim {
        let knots = &breakpoints[a];
        let i = locate_cubic_interval(knots, coords[a]);
        dxs[a] = coords[a] - knots[i];
        cell = cell * (knots.len() - 1) + i;
    }
    let csize = order.pow(ndim as u32);
    &coeffs[cell * csize..(cell + 1) * csize]
}

/// Evaluate an N-D tensor-product polynomial in nested Horner form.
/// `coeffs` has `order^(dxs.len())` entries, axis-0 power slowest, ascending.
#[allow(clippy::suboptimal_flops)]
pub(crate) fn tensor_horner(coeffs: &[f64], dxs: &[f64], order: usize) -> f64 {
    debug_assert!(order >= 2);
    if dxs.len() == 1 {
        let mut acc = coeffs[order - 1];
        for a in (0..order - 1).rev() {
            acc = acc * dxs[0] + coeffs[a];
        }
        return acc;
    }
    let stride = coeffs.len() / order;
    let mut acc = tensor_horner(&coeffs[(order - 1) * stride..], &dxs[1..], order);
    for a in (0..order - 1).rev() {
        acc = acc * dxs[0] + tensor_horner(&coeffs[a * stride..(a + 1) * stride], &dxs[1..], order);
    }
    acc
}

/// Partial derivative of the N-D tensor-product polynomial along `axis`.
#[allow(clippy::suboptimal_flops)]
pub(crate) fn tensor_horner_partial(coeffs: &[f64], dxs: &[f64], order: usize, axis: usize) -> f64 {
    debug_assert!(order >= 2);
    debug_assert!(axis < dxs.len());
    if axis == 0 {
        if dxs.len() == 1 {
            let mut acc = (order - 1) as f64 * coeffs[order - 1];
            for a in (1..order - 1).rev() {
                acc = acc * dxs[0] + a as f64 * coeffs[a];
            }
            return acc;
        }
        let stride = coeffs.len() / order;
        let mut acc =
            (order - 1) as f64 * tensor_horner(&coeffs[(order - 1) * stride..], &dxs[1..], order);
        for a in (1..order - 1).rev() {
            acc = acc * dxs[0]
                + a as f64 * tensor_horner(&coeffs[a * stride..(a + 1) * stride], &dxs[1..], order);
        }
        return acc;
    }
    let stride = coeffs.len() / order;
    let mut acc =
        tensor_horner_partial(&coeffs[(order - 1) * stride..], &dxs[1..], order, axis - 1);
    for a in (0..order - 1).rev() {
        acc = acc * dxs[0]
            + tensor_horner_partial(
                &coeffs[a * stride..(a + 1) * stride],
                &dxs[1..],
                order,
                axis - 1,
            );
    }
    acc
}

/// Approximation of the error function
/// Tolerance `BinaryOp::Equality` compares within, shared with the batched
/// tangent sweep so both paths agree.
pub(crate) const EQUALITY_EPS: f64 = 1e-14;

///
/// Shared with constant folding in `simplify` so folded values match
/// runtime evaluation exactly.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn erf_approx(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / 0.327_591_1f64.mul_add(x, 1.0);
    let poly = t * t.mul_add(
        t.mul_add(
            t.mul_add(t.mul_add(1.061_405_429, -1.453_152_027), 1.421_413_741),
            -0.284_496_736,
        ),
        0.254_829_592,
    );
    sign * poly.mul_add(-(-x * x).exp(), 1.0)
}

/// Sign function with `sign(0) = 0` (unlike `f64::signum`).
///
/// Shared with constant folding in `simplify` so folded values match
/// runtime evaluation exactly.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn sign(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x.signum() }
}

/// Carve a read window `[src, src+len)` and a write window `[dst, dst+len)`
/// out of `buf` as disjoint slices, which the IR allocator guarantees.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn split_src_dst(
    buf: &mut [f64],
    src: usize,
    dst: usize,
    len: usize,
) -> (&[f64], &mut [f64]) {
    debug_assert!(
        src + len <= dst || dst + len <= src,
        "src/dst windows overlap (src={src}, dst={dst}, len={len})"
    );
    if dst < src {
        let (left, right) = buf.split_at_mut(src);
        (&right[..len], &mut left[dst..dst + len])
    } else {
        let (left, right) = buf.split_at_mut(dst);
        (&left[src..src + len], &mut right[..len])
    }
}

/// Carve two read windows `a` and `b` and a write window `dst`, all length
/// `len`, out of `buf`. `dst` must be disjoint from both `a` and `b` (the IR
/// allocator guarantees this); `a` and `b` may overlap each other (read-only).
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn split_dst_two_src(
    buf: &mut [f64],
    a: usize,
    b: usize,
    dst: usize,
    len: usize,
) -> (&[f64], &[f64], &mut [f64]) {
    debug_assert!(
        a + len <= dst || dst + len <= a,
        "operand a overlaps dst (a={a}, dst={dst}, len={len})"
    );
    debug_assert!(
        b + len <= dst || dst + len <= b,
        "operand b overlaps dst (b={b}, dst={dst}, len={len})"
    );
    let (left, rest) = buf.split_at_mut(dst);
    let (dst_s, right) = rest.split_at_mut(len);
    let a_s = if a + len <= dst {
        &left[a..a + len]
    } else {
        let off = a - (dst + len);
        &right[off..off + len]
    };
    let b_s = if b + len <= dst {
        &left[b..b + len]
    } else {
        let off = b - (dst + len);
        &right[off..off + len]
    };
    (a_s, b_s, dst_s)
}

/// Apply a binary operation with broadcasting.
#[allow(clippy::inline_always)]
#[inline(always)]
fn broadcast_apply<F: Fn(f64, f64) -> f64>(
    buf: &mut [f64],
    f: F,
    a: usize,
    b: usize,
    dst: usize,
    len: usize,
    kind: BroadcastKind,
) {
    match kind {
        BroadcastKind::ScalarScalar => {
            buf[dst] = f(buf[a], buf[b]);
        },
        BroadcastKind::ScalarVector => {
            let scalar = buf[a];
            let (b_s, d_s) = split_src_dst(buf, b, dst, len);
            for (o, &y) in d_s.iter_mut().zip(b_s) {
                *o = f(scalar, y);
            }
        },
        BroadcastKind::VectorScalar => {
            let scalar = buf[b];
            let (a_s, d_s) = split_src_dst(buf, a, dst, len);
            for (o, &x) in d_s.iter_mut().zip(a_s) {
                *o = f(x, scalar);
            }
        },
        BroadcastKind::VectorVector => {
            let (a_s, b_s, d_s) = split_dst_two_src(buf, a, b, dst, len);
            for ((o, &x), &y) in d_s.iter_mut().zip(a_s).zip(b_s) {
                *o = f(x, y);
            }
        },
    }
}

/// Evaluate a binary operation.
#[allow(clippy::inline_always)]
#[inline(always)]
fn eval_binary_op(
    buf: &mut [f64],
    op: BinaryOp,
    a: usize,
    b: usize,
    dst: usize,
    len: usize,
    kind: BroadcastKind,
) {
    match op {
        BinaryOp::Add => broadcast_apply(buf, |x, y| x + y, a, b, dst, len, kind),
        BinaryOp::Sub => broadcast_apply(buf, |x, y| x - y, a, b, dst, len, kind),
        BinaryOp::Mul => broadcast_apply(buf, |x, y| x * y, a, b, dst, len, kind),
        BinaryOp::Div => broadcast_apply(buf, |x, y| x / y, a, b, dst, len, kind),
        BinaryOp::Pow => broadcast_apply(buf, f64::powf, a, b, dst, len, kind),
        BinaryOp::Minimum => broadcast_apply(buf, f64::min, a, b, dst, len, kind),
        BinaryOp::Maximum => broadcast_apply(buf, f64::max, a, b, dst, len, kind),
        BinaryOp::Modulo => broadcast_apply(buf, |x, y| x % y, a, b, dst, len, kind),
        BinaryOp::Hypot => broadcast_apply(buf, f64::hypot, a, b, dst, len, kind),
        BinaryOp::EqualHeaviside => broadcast_apply(
            buf,
            |x, y| if x <= y { 1.0 } else { 0.0 },
            a,
            b,
            dst,
            len,
            kind,
        ),
        BinaryOp::NotEqualHeaviside => broadcast_apply(
            buf,
            |x, y| if x < y { 1.0 } else { 0.0 },
            a,
            b,
            dst,
            len,
            kind,
        ),
        BinaryOp::Equality => {
            broadcast_apply(
                buf,
                |x, y| {
                    if (x - y).abs() < EQUALITY_EPS {
                        1.0
                    } else {
                        0.0
                    }
                },
                a,
                b,
                dst,
                len,
                kind,
            );
        },
    }
}

/// Apply a unary function.
#[allow(clippy::inline_always)]
#[inline(always)]
fn unary_apply<F: Fn(f64) -> f64>(buf: &mut [f64], f: F, src: usize, dst: usize, len: usize) {
    if src == dst {
        for x in &mut buf[dst..dst + len] {
            *x = f(*x);
        }
        return;
    }
    let (src_s, dst_s) = split_src_dst(buf, src, dst, len);
    for (o, &x) in dst_s.iter_mut().zip(src_s) {
        *o = f(x);
    }
}

/// Evaluate a unary operation.
#[allow(clippy::inline_always)]
#[inline(always)]
fn eval_unary_op(buf: &mut [f64], op: UnaryOp, src: usize, dst: usize, len: usize) {
    match op {
        UnaryOp::Neg => unary_apply(buf, |x| -x, src, dst, len),
        UnaryOp::Abs => unary_apply(buf, f64::abs, src, dst, len),
        UnaryOp::Sqrt => unary_apply(buf, f64::sqrt, src, dst, len),
        UnaryOp::Exp => unary_apply(buf, f64::exp, src, dst, len),
        UnaryOp::Log => unary_apply(buf, f64::ln, src, dst, len),
        UnaryOp::Sin => unary_apply(buf, f64::sin, src, dst, len),
        UnaryOp::Cos => unary_apply(buf, f64::cos, src, dst, len),
        UnaryOp::Tanh => unary_apply(buf, f64::tanh, src, dst, len),
        UnaryOp::Sinh => unary_apply(buf, f64::sinh, src, dst, len),
        UnaryOp::Cosh => unary_apply(buf, f64::cosh, src, dst, len),
        UnaryOp::Arcsinh => unary_apply(buf, f64::asinh, src, dst, len),
        UnaryOp::Arctan => unary_apply(buf, f64::atan, src, dst, len),
        UnaryOp::Erf => unary_apply(buf, erf_approx, src, dst, len),
        UnaryOp::Sign => unary_apply(buf, sign, src, dst, len),
        UnaryOp::Floor => unary_apply(buf, f64::floor, src, dst, len),
        UnaryOp::Ceiling => unary_apply(buf, f64::ceil, src, dst, len),
    }
}

/// Seed vectors for tangent evaluation (forward-mode AD).
///
/// A seed is the direction the JVP is taken in, so one entry per state or per
/// parameter; `None` leaves that family of tangent reads at zero. Colored Jacobian
/// assembly seeds `dy` with a 1 in every column of the current color.
#[derive(Debug, Default)]
pub struct TangentInputs<'a> {
    /// Tangent of the state vector, indexed like `y`.
    pub dy: Option<&'a [f64]>,
    /// Tangent of the parameters, one entry per parameter regardless of width:
    /// indexed by `InputParameter::index`, *not* by offset into `inputs`.
    pub dp: Option<&'a [f64]>,
}

#[derive(Debug, Clone, Copy)]
struct PrimalEvalInputs<'a> {
    t: f64,
    y: &'a [f64],
    y_dot: &'a [f64],
    inputs: &'a [f64],
}

#[derive(Debug, Clone, Copy)]
struct EvalContext<'a> {
    primal: Option<PrimalEvalInputs<'a>>,
    tangent: Option<&'a TangentInputs<'a>>,
}

impl<'a> EvalContext<'a> {
    const fn with_primal(
        t: f64,
        y: &'a [f64],
        y_dot: &'a [f64],
        inputs: &'a [f64],
        tangent: Option<&'a TangentInputs<'a>>,
    ) -> Self {
        Self {
            primal: Some(PrimalEvalInputs {
                t,
                y,
                y_dot,
                inputs,
            }),
            tangent,
        }
    }

    const fn tangent_only(tangent: &'a TangentInputs<'a>) -> Self {
        Self {
            primal: None,
            tangent: Some(tangent),
        }
    }

    const fn primal(self) -> PrimalEvalInputs<'a> {
        self.primal
            .expect("primal instructions require full evaluation inputs")
    }

    fn tangent_state(self) -> Option<&'a [f64]> {
        self.tangent.and_then(|tangent| tangent.dy)
    }

    fn tangent_parameter(self, index: usize) -> f64 {
        self.tangent
            .and_then(|tangent| tangent.dp)
            .map_or(0.0, |dp| dp[index])
    }
}

/// Execute a slice of instructions against a shared buffer, returning the number
/// of instructions actually executed.
///
/// `Dispatch` advances past the blocks of every inactive branch, so the count is
/// the honest cost of this evaluation rather than the tape length. The
/// accumulator costs one register; callers that ignore it pay nothing.
#[inline]
fn eval_instructions(
    buf: &mut [f64],
    instructions: &[Instruction],
    consts: &ConstPool,
    ctx: EvalContext<'_>,
) -> usize {
    let mut executed = 0_usize;
    // Named `pc`, not `i`: many arms bind their own loop-local `i`.
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
            executed += 1;
            if let Some(active) = active_branch(buf[selector as usize], n) {
                let (rel, len) = consts.branch_blocks[base + active];
                let start = pc + rel as usize;
                executed +=
                    eval_instructions(buf, &instructions[start..start + len as usize], consts, ctx);
            }
            pc = span_end;
            continue;
        }

        match instructions[pc] {
            Instruction::LoadScalar { value, dst } => {
                buf[dst as usize] = value;
            },
            Instruction::LoadTime { dst } => {
                buf[dst as usize] = ctx.primal().t;
            },
            Instruction::LoadArray { data_idx, len, dst } => {
                let src = consts.get_array(data_idx, len);
                buf[dst as usize..dst as usize + len as usize].copy_from_slice(src);
            },
            Instruction::FillZero { dst, len } => {
                buf[dst as usize..dst as usize + len as usize].fill(0.0);
            },
            Instruction::LoadStateVector { start, end, dst } => {
                let primal = ctx.primal();
                buf[dst as usize..dst as usize + (end - start) as usize]
                    .copy_from_slice(&primal.y[start as usize..end as usize]);
            },
            Instruction::LoadStateVectorDot { start, end, dst } => {
                let primal = ctx.primal();
                buf[dst as usize..dst as usize + (end - start) as usize]
                    .copy_from_slice(&primal.y_dot[start as usize..end as usize]);
            },
            Instruction::LoadInputParameter { offset, width, dst } => {
                let primal = ctx.primal();
                let offset = offset as usize;
                let width = width as usize;
                buf[dst as usize..dst as usize + width]
                    .copy_from_slice(&primal.inputs[offset..offset + width]);
            },
            Instruction::LoadTangentState { start, end, dst } => {
                if let Some(dy) = ctx.tangent_state() {
                    buf[dst as usize..dst as usize + (end - start) as usize]
                        .copy_from_slice(&dy[start as usize..end as usize]);
                } else {
                    buf[dst as usize..dst as usize + (end - start) as usize].fill(0.0);
                }
            },
            Instruction::LoadTangentParameter { index, dst } => {
                buf[dst as usize] = ctx.tangent_parameter(index as usize);
            },

            Instruction::Binary {
                op,
                a,
                b,
                dst,
                len,
                kind,
            } => {
                eval_binary_op(
                    buf,
                    op,
                    a as usize,
                    b as usize,
                    dst as usize,
                    len as usize,
                    kind,
                );
            },

            Instruction::Unary { op, src, dst, len } => {
                eval_unary_op(buf, op, src as usize, dst as usize, len as usize);
            },

            Instruction::MaxReduce { src, src_len, dst } => {
                let src = src as usize;
                let src_len = src_len as usize;
                let mut max_val = f64::NEG_INFINITY;
                for i in 0..src_len {
                    let v = buf[src + i];
                    if v > max_val {
                        max_val = v;
                    }
                }
                buf[dst as usize] = max_val;
            },

            Instruction::MinReduce { src, src_len, dst } => {
                let src = src as usize;
                let src_len = src_len as usize;
                let mut min_val = f64::INFINITY;
                for i in 0..src_len {
                    let v = buf[src + i];
                    if v < min_val {
                        min_val = v;
                    }
                }
                buf[dst as usize] = min_val;
            },

            Instruction::ReduceArgSelect {
                basis_src,
                picker_src,
                len,
                is_max,
                dst,
            } => {
                let basis = basis_src as usize;
                let picker = picker_src as usize;
                let len = len as usize;
                // Argmax/argmin takes the earliest element under a strict
                // comparison, matching the primal MaxReduce/MinReduce eval.
                let mut best_idx = 0;
                let mut best_val = buf[picker];
                for i in 1..len {
                    let v = buf[picker + i];
                    if (is_max && v > best_val) || (!is_max && v < best_val) {
                        best_val = v;
                        best_idx = i;
                    }
                }
                buf[dst as usize] = buf[basis + best_idx];
            },

            Instruction::Index {
                src,
                start,
                dst,
                len,
            } => {
                buf.copy_within(
                    src as usize + start as usize..src as usize + start as usize + len as usize,
                    dst as usize,
                );
            },

            Instruction::Concat {
                sources_idx,
                sources_len,
                dst,
            } => {
                let mut write_pos = dst as usize;
                for i in 0..sources_len as usize {
                    let (src_off, src_len) = consts.concat_sources[sources_idx as usize + i];
                    buf.copy_within(
                        src_off as usize..src_off as usize + src_len as usize,
                        write_pos,
                    );
                    write_pos += src_len as usize;
                }
            },

            Instruction::MatMul {
                csr_idx,
                vec_src,
                dst,
            } => {
                let csr = &consts.csr_data[csr_idx as usize];
                let vec_src = vec_src as usize;
                let dst = dst as usize;
                for row in 0..csr.shape.rows {
                    let start = csr.indptr[row];
                    let end = csr.indptr[row + 1];
                    let cols = &csr.indices[start..end];
                    let vals = &csr.data[start..end];
                    let mut sum = 0.0;
                    // The data-dependent gather `buf[vec_src + col]` stays
                    // bounds-checked; the row's value/index reads do not.
                    for (&col, &val) in cols.iter().zip(vals) {
                        sum += val * buf[vec_src + col];
                    }
                    buf[dst + row] = sum;
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
                for row in 0..rows as usize {
                    let mut sum = 0.0;
                    for col in 0..cols as usize {
                        sum += buf[mat_src + row * cols as usize + col] * buf[vec_src + col];
                    }
                    buf[dst + row] = sum;
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
                    buf[dst + i] = interp_linear_1d(&interp.x_data, &interp.y_data, buf[src + i]);
                }
            },

            Instruction::Interp1DLinearDeriv {
                interp_idx,
                src,
                dst,
                len,
            } => {
                let interp = &consts.interpolants[interp_idx as usize];
                let src = src as usize;
                let dst = dst as usize;
                for i in 0..len as usize {
                    buf[dst + i] =
                        interp_linear_1d_slope_lookup(&interp.x_data, &interp.y_data, buf[src + i]);
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
                    buf[dst + i] =
                        interp_cubic_1d(&interp.breakpoints, &interp.coeffs, buf[src + i]);
                }
            },
            Instruction::Interp1DCubicDeriv {
                interp_idx,
                src,
                dst,
                len,
            } => {
                let interp = &consts.cubic_interpolants[interp_idx as usize];
                let src = src as usize;
                let dst = dst as usize;
                for i in 0..len as usize {
                    buf[dst + i] =
                        interp_cubic_1d_deriv(&interp.breakpoints, &interp.coeffs, buf[src + i]);
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
                    for (a, coord) in coords.iter_mut().enumerate().take(ndim) {
                        let (off, slen) = consts.interp_nd_sources[sources_idx as usize + a];
                        // Length-1 children broadcast over the output length.
                        let j = if slen == 1 { 0 } else { i };
                        *coord = buf[off as usize + j];
                    }
                    let cell = locate_nd_cell(
                        &interp.breakpoints,
                        &interp.coeffs,
                        order,
                        &coords[..ndim],
                        &mut dxs,
                    );
                    buf[dst + i] = tensor_horner(cell, &dxs[..ndim], order);
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
                for i in 0..len as usize {
                    for (a, coord) in coords.iter_mut().enumerate().take(ndim) {
                        let (off, slen) = consts.interp_nd_sources[sources_idx as usize + a];
                        let j = if slen == 1 { 0 } else { i };
                        *coord = buf[off as usize + j];
                    }
                    let cell = locate_nd_cell(
                        &interp.breakpoints,
                        &interp.coeffs,
                        order,
                        &coords[..ndim],
                        &mut dxs,
                    );
                    buf[dst + i] = tensor_horner_partial(cell, &dxs[..ndim], order, axis as usize);
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
                match active_branch(buf[selector as usize], branches_len as usize) {
                    Some(i) => {
                        let (branch_off, _) = consts.branch_offsets[branches_idx as usize + i];
                        buf.copy_within(branch_off as usize..branch_off as usize + out_len, dst);
                    },
                    None => buf[dst..dst + out_len].fill(0.0),
                }
            },
            Instruction::Dispatch { .. } => unreachable!("handled above"),
        }
        executed += 1;
        pc += 1;
    }
    executed
}

/// Zero-allocation expression evaluator.
///
/// Wraps a `TypedIr` and provides efficient evaluation. Callers must supply
/// an external scratch buffer of length `scratch_len()` so that the same
/// `CompiledExpr` can be shared across concurrent or re-entrant solves without
/// interior-mutable state.
#[derive(Debug, Clone)]
pub struct CompiledExpr {
    ir: TypedIr,
}

impl CompiledExpr {
    /// Compile an expression DAG into a new `CompiledExpr`.
    pub fn new(arena: &Arena, root: NodeId) -> Self {
        Self::from_ir(TypedIr::from_arena(arena, root))
    }

    /// Compile with a no-reuse (SSA) slot layout for reverse-mode AD. After
    /// [`eval`](Self::eval) the scratch holds every intermediate value, so the
    /// reverse backward pass reads operands by their stable slots.
    pub fn new_pinned(arena: &Arena, root: NodeId) -> Self {
        Self::from_ir(TypedIr::from_arena_pinned(arena, root))
    }

    /// Wrap a pre-built `TypedIr`.
    pub const fn from_ir(ir: TypedIr) -> Self {
        Self { ir }
    }

    /// Reference to the underlying `TypedIr`.
    #[inline]
    pub const fn ir(&self) -> &TypedIr {
        &self.ir
    }

    /// Length of the scratch buffer this expression needs for evaluation.
    #[inline]
    pub const fn scratch_len(&self) -> usize {
        self.ir.buffer_size()
    }

    /// Output length of the root expression.
    #[inline]
    pub const fn output_len(&self) -> usize {
        self.ir.output_len()
    }

    /// Evaluate the expression into `scratch` (length `>= scratch_len()`),
    /// returning the slice holding the root result.
    pub fn eval<'s>(
        &self,
        scratch: &'s mut [f64],
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
    ) -> &'s [f64] {
        self.eval_internal(scratch, t, y, y_dot, inputs, None).1
    }

    /// [`eval`](Self::eval), also returning how many instructions ran.
    ///
    /// The direct test of only-active-branch execution: a reported tape length
    /// cannot distinguish work that was skipped from work that was not counted.
    pub fn eval_counted<'s>(
        &self,
        scratch: &'s mut [f64],
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
    ) -> (usize, &'s [f64]) {
        self.eval_internal(scratch, t, y, y_dot, inputs, None)
    }

    /// Evaluate with tangent inputs for forward-mode AD (JVP).
    pub fn eval_with_tangent<'s>(
        &self,
        scratch: &'s mut [f64],
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
        tangent: &TangentInputs<'_>,
    ) -> &'s [f64] {
        self.eval_internal(scratch, t, y, y_dot, inputs, Some(tangent))
            .1
    }

    /// [`eval_with_tangent`](Self::eval_with_tangent), also returning how many
    /// instructions ran.
    ///
    /// The direct test of only-active-branch execution on a tape that carries
    /// tangent work, where the split layout puts one branch in two blocks.
    pub fn eval_counted_with_tangent<'s>(
        &self,
        scratch: &'s mut [f64],
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
        tangent: &TangentInputs<'_>,
    ) -> (usize, &'s [f64]) {
        self.eval_internal(scratch, t, y, y_dot, inputs, Some(tangent))
    }

    fn eval_internal<'s>(
        &self,
        scratch: &'s mut [f64],
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
        tangent: Option<&TangentInputs<'_>>,
    ) -> (usize, &'s [f64]) {
        let executed = eval_instructions(
            scratch,
            self.ir.instructions(),
            self.ir.consts(),
            EvalContext::with_primal(t, y, y_dot, inputs, tangent),
        );

        let root = self.ir.root_slot();
        (
            executed,
            &scratch[root.offset_usize()..root.offset_usize() + root.len_usize()],
        )
    }

    /// Evaluate the primal section of a split-eval expression into `scratch`,
    /// returning a [`PrimalCache`] that owns the buffer. The tangent sweep runs
    /// through the returned cache, which encodes the "primal first, same
    /// scratch" contract that a bare tangent call cannot express.
    pub fn eval_primal<'a>(
        &'a self,
        scratch: &'a mut [f64],
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
    ) -> PrimalCache<'a> {
        self.run_primal_section(scratch, t, y, y_dot, inputs);
        PrimalCache {
            expr: self,
            scratch,
        }
    }

    /// Raw primal-section evaluation into `scratch` (no cache). Primitive for
    /// the sensitivity and batched-tangent paths, whose primal pass and tangent
    /// sweeps write separate buffers and so cannot share a borrow-based cache;
    /// prefer [`eval_primal`](Self::eval_primal) elsewhere.
    pub fn run_primal_section(
        &self,
        scratch: &mut [f64],
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
    ) {
        let primal_end = self
            .ir
            .split_eval_info()
            .map_or_else(|| self.ir.instructions().len(), |s| s.primal_end);

        eval_instructions(
            scratch,
            &self.ir.instructions()[..primal_end],
            self.ir.consts(),
            EvalContext::with_primal(t, y, y_dot, inputs, None),
        );
    }

    /// Raw tangent-section evaluation, reusing the primal region a prior
    /// [`run_primal_section`](Self::run_primal_section) left in `scratch`.
    /// Internal primitive for the sensitivity path; prefer
    /// [`PrimalCache::eval_tangent`] elsewhere.
    pub(crate) fn run_tangent_section<'s>(
        &self,
        scratch: &'s mut [f64],
        tangent: &TangentInputs<'_>,
    ) -> &'s [f64] {
        debug_assert!(
            self.ir.split_eval_info().is_some(),
            "run_tangent_section requires a split-eval IR"
        );
        let primal_end = self
            .ir
            .split_eval_info()
            .map_or_else(|| self.ir.instructions().len(), |s| s.primal_end);

        eval_instructions(
            scratch,
            &self.ir.instructions()[primal_end..],
            self.ir.consts(),
            EvalContext::tangent_only(tangent),
        );

        let root = self.ir.root_slot();
        &scratch[root.offset_usize()..root.offset_usize() + root.len_usize()]
    }

    /// True when this expression was compiled with `from_arena_split_eval`,
    /// i.e. split primal/tangent evaluation via [`eval_primal`](Self::eval_primal)
    /// is available.
    #[inline]
    pub const fn has_split_eval(&self) -> bool {
        self.ir.split_eval_info().is_some()
    }

    /// Evaluate and copy the result into `out`, using `scratch` for working space.
    #[inline]
    pub fn eval_into(
        &self,
        scratch: &mut [f64],
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
        out: &mut [f64],
    ) {
        let result = self.eval(scratch, t, y, y_dot, inputs);
        out[..result.len()].copy_from_slice(result);
    }
}

/// Borrow-based typestate for split evaluation.
///
/// Produced only by [`CompiledExpr::eval_primal`], it owns the scratch buffer
/// whose primal region was just filled. Because a cache is the only way to
/// reach the tangent sweep, calling tangent before primal or on a different
/// buffer is a compile error rather than a silent stale-buffer read.
pub struct PrimalCache<'a> {
    expr: &'a CompiledExpr,
    scratch: &'a mut [f64],
}

impl std::fmt::Debug for PrimalCache<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrimalCache").finish_non_exhaustive()
    }
}

impl PrimalCache<'_> {
    /// Evaluate the tangent section for `tangent`, reusing the primal region
    /// filled by [`CompiledExpr::eval_primal`], and return the root slice.
    pub fn eval_tangent(&mut self, tangent: &TangentInputs<'_>) -> &[f64] {
        self.expr.run_tangent_section(self.scratch, tangent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{
        ArrayData, CsrData, CubicInterpolantData, InterpolantData, NdInterpolantData, Node, Shape,
    };

    #[test]
    fn external_scratch_eval_matches_len() {
        // dy/dt expression: 2.0 * y[0]
        let mut arena = Arena::new();
        let sv = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let two = arena.alloc(Node::Scalar(2.0));
        let expr_node = arena.alloc(Node::Mul(two, sv));
        let expr = CompiledExpr::new(&arena, expr_node);

        let mut scratch = vec![0.0; expr.scratch_len()];
        let result = expr.eval(&mut scratch, 0.0, &[3.0], &[], &[]);
        assert_eq!(result, &[6.0]);
    }

    #[test]
    fn size_check() {
        use std::mem::size_of;
        // These live in hot per-node arrays, so guard against accidental bloat;
        // bump a bound deliberately if a type must grow.
        assert!(
            size_of::<Instruction>() <= 32,
            "Instruction grew to {} bytes",
            size_of::<Instruction>()
        );
        assert!(
            size_of::<Node>() <= 48,
            "Node grew to {} bytes",
            size_of::<Node>()
        );
        assert_eq!(size_of::<NodeId>(), 4);
        assert_eq!(size_of::<BroadcastKind>(), 1);
        assert_eq!(size_of::<BinaryOp>(), 1);
        assert_eq!(size_of::<UnaryOp>(), 1);
    }

    #[test]
    fn test_eval_scalar() {
        let mut arena = Arena::new();
        let id = arena.alloc(Node::Scalar(42.0));
        let compiled = CompiledExpr::new(&arena, id);
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &[], &[], &[]);
        assert_eq!(result, &[42.0]);
    }

    #[test]
    fn test_eval_time() {
        let mut arena = Arena::new();
        let id = arena.alloc(Node::Time);
        let compiled = CompiledExpr::new(&arena, id);
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 3.5, &[], &[], &[]);
        assert_eq!(result, &[3.5]);
    }

    #[test]
    fn test_eval_state_vector() {
        let mut arena = Arena::new();
        let id = arena.alloc(Node::StateVector { start: 1, end: 3 });
        let compiled = CompiledExpr::new(&arena, id);
        let y = [10.0, 20.0, 30.0, 40.0];
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &y, &[], &[]);
        assert_eq!(result, &[20.0, 30.0]);
    }

    #[test]
    fn test_eval_array() {
        let mut arena = Arena::new();
        let id = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0, 2.0, 3.0],
            shape: Shape::vector(3),
        })));
        let compiled = CompiledExpr::new(&arena, id);
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &[], &[], &[]);
        assert_eq!(result, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_eval_add_scalars() {
        let mut arena = Arena::new();
        let a = arena.alloc(Node::Scalar(2.0));
        let b = arena.alloc(Node::Scalar(3.0));
        let sum = arena.alloc(Node::Add(a, b));
        let compiled = CompiledExpr::new(&arena, sum);
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &[], &[], &[]);
        assert_eq!(result, &[5.0]);
    }

    #[test]
    fn test_eval_add_vectors() {
        let mut arena = Arena::new();
        let a = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0, 2.0, 3.0],
            shape: Shape::vector(3),
        })));
        let b = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![10.0, 20.0, 30.0],
            shape: Shape::vector(3),
        })));
        let sum = arena.alloc(Node::Add(a, b));
        let compiled = CompiledExpr::new(&arena, sum);
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &[], &[], &[]);
        assert_eq!(result, &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_eval_scalar_times_vector() {
        let mut arena = Arena::new();
        let s_node = arena.alloc(Node::Scalar(2.0));
        let v = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0, 2.0, 3.0],
            shape: Shape::vector(3),
        })));
        let prod = arena.alloc(Node::Mul(s_node, v));
        let compiled = CompiledExpr::new(&arena, prod);
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &[], &[], &[]);
        assert_eq!(result, &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_eval_nested_arithmetic() {
        let mut arena = Arena::new();
        let two = arena.alloc(Node::Scalar(2.0));
        let three = arena.alloc(Node::Scalar(3.0));
        let four = arena.alloc(Node::Scalar(4.0));
        let one = arena.alloc(Node::Scalar(1.0));
        let sum = arena.alloc(Node::Add(two, three));
        let prod = arena.alloc(Node::Mul(sum, four));
        let result_node = arena.alloc(Node::Sub(prod, one));
        let compiled = CompiledExpr::new(&arena, result_node);
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &[], &[], &[]);
        assert_eq!(result, &[19.0]);
    }

    #[test]
    fn test_eval_neg() {
        let mut arena = Arena::new();
        let a = arena.alloc(Node::Scalar(5.0));
        let neg = arena.alloc(Node::Neg(a));
        let compiled = CompiledExpr::new(&arena, neg);
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &[], &[], &[]);
        assert_eq!(result, &[-5.0]);
    }

    #[test]
    fn test_eval_sqrt_vector() {
        let mut arena = Arena::new();
        let a = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0, 4.0, 9.0],
            shape: Shape::vector(3),
        })));
        let node = arena.alloc(Node::Sqrt(a));
        let compiled = CompiledExpr::new(&arena, node);
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &[], &[], &[]);
        assert_eq!(result, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_eval_sin_cos() {
        let mut arena = Arena::new();
        let zero = arena.alloc(Node::Scalar(0.0));
        let sin = arena.alloc(Node::Sin(zero));
        let cos = arena.alloc(Node::Cos(zero));
        let sum = arena.alloc(Node::Add(sin, cos));
        let compiled = CompiledExpr::new(&arena, sum);
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &[], &[], &[]);
        assert_eq!(result, &[1.0]);
    }

    #[test]
    fn test_eval_max_reduce() {
        let mut arena = Arena::new();
        let a = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0, 5.0, 3.0],
            shape: Shape::vector(3),
        })));
        let node = arena.alloc(Node::MaxReduce(a));
        let compiled = CompiledExpr::new(&arena, node);
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &[], &[], &[]);
        assert_eq!(result, &[5.0]);
    }

    #[test]
    fn test_eval_reduce_arg_select_max() {
        // basis[argmax(picker)]; picker=[1,5,3] -> k=1 -> basis[1]=20
        let mut arena = Arena::new();
        let picker = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0, 5.0, 3.0],
            shape: Shape::vector(3),
        })));
        let basis = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![10.0, 20.0, 30.0],
            shape: Shape::vector(3),
        })));
        let node = arena.alloc(Node::ReduceArgSelect {
            basis,
            picker,
            is_max: true,
        });
        let compiled = CompiledExpr::new(&arena, node);
        let mut s = vec![0.0; compiled.scratch_len()];
        assert_eq!(compiled.eval(&mut s, 0.0, &[], &[], &[]), &[20.0]);
    }

    #[test]
    fn test_eval_reduce_arg_select_min() {
        // picker=[1,5,3] -> argmin k=0 -> basis[0]=10
        let mut arena = Arena::new();
        let picker = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0, 5.0, 3.0],
            shape: Shape::vector(3),
        })));
        let basis = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![10.0, 20.0, 30.0],
            shape: Shape::vector(3),
        })));
        let node = arena.alloc(Node::ReduceArgSelect {
            basis,
            picker,
            is_max: false,
        });
        let compiled = CompiledExpr::new(&arena, node);
        let mut s = vec![0.0; compiled.scratch_len()];
        assert_eq!(compiled.eval(&mut s, 0.0, &[], &[], &[]), &[10.0]);
    }

    #[test]
    fn test_eval_reduce_arg_select_tie_first_wins() {
        // picker=[5,5,3] max ties at 0 and 1 -> first index 0 -> basis[0]=10
        let mut arena = Arena::new();
        let picker = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![5.0, 5.0, 3.0],
            shape: Shape::vector(3),
        })));
        let basis = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![10.0, 20.0, 30.0],
            shape: Shape::vector(3),
        })));
        let node = arena.alloc(Node::ReduceArgSelect {
            basis,
            picker,
            is_max: true,
        });
        let compiled = CompiledExpr::new(&arena, node);
        let mut s = vec![0.0; compiled.scratch_len()];
        assert_eq!(compiled.eval(&mut s, 0.0, &[], &[], &[]), &[10.0]);
    }

    #[test]
    fn test_eval_reduce_arg_select_min_tie_first_wins() {
        // picker=[5,3,3] min ties at 1 and 2 -> first index 1 -> basis[1]=20
        let mut arena = Arena::new();
        let picker = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![5.0, 3.0, 3.0],
            shape: Shape::vector(3),
        })));
        let basis = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![10.0, 20.0, 30.0],
            shape: Shape::vector(3),
        })));
        let node = arena.alloc(Node::ReduceArgSelect {
            basis,
            picker,
            is_max: false,
        });
        let compiled = CompiledExpr::new(&arena, node);
        let mut s = vec![0.0; compiled.scratch_len()];
        assert_eq!(compiled.eval(&mut s, 0.0, &[], &[], &[]), &[20.0]);
    }

    #[test]
    fn test_eval_index() {
        let mut arena = Arena::new();
        let arr = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![10.0, 20.0, 30.0, 40.0, 50.0],
            shape: Shape::vector(5),
        })));
        let idx = arena.alloc(Node::Index {
            child: arr,
            start: 1,
            end: 4,
        });
        let compiled = CompiledExpr::new(&arena, idx);
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &[], &[], &[]);
        assert_eq!(result, &[20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_eval_concat() {
        let mut arena = Arena::new();
        let a = arena.alloc(Node::Scalar(1.0));
        let b = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![2.0, 3.0],
            shape: Shape::vector(2),
        })));
        let c = arena.alloc(Node::Scalar(4.0));
        let concat = arena.alloc(Node::Concat(vec![a, b, c]));
        let compiled = CompiledExpr::new(&arena, concat);
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &[], &[], &[]);
        assert_eq!(result, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_eval_matmul() {
        let mut arena = Arena::new();
        let sparse = arena.alloc(Node::SparseMatrix(Box::new(CsrData {
            indptr: vec![0, 1, 2],
            indices: vec![0, 1],
            data: vec![2.0, 3.0],
            shape: Shape::matrix(2, 3),
        })));
        let v = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0, 2.0, 3.0],
            shape: Shape::vector(3),
        })));
        let matmul = arena.alloc(Node::MatMul(sparse, v));
        let compiled = CompiledExpr::new(&arena, matmul);
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &[], &[], &[]);
        assert_eq!(result, &[2.0, 6.0]);
    }

    #[test]
    fn test_eval_dense_matmul() {
        let mut arena = Arena::new();
        let a = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // row-major 2x3
            shape: Shape::matrix(2, 3),
        })));
        let v = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0, 2.0, 3.0],
            shape: Shape::vector(3),
        })));
        let matmul = arena.alloc(Node::MatMul(a, v));
        let compiled = CompiledExpr::new(&arena, matmul);
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &[], &[], &[]);
        assert_eq!(result, &[14.0, 32.0]);
    }

    #[test]
    fn test_eval_interpolant() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::Scalar(1.5));
        let interp = arena.alloc(Node::Interpolant1DLinear {
            data: Box::new(InterpolantData {
                x_data: vec![0.0, 1.0, 2.0],
                y_data: vec![0.0, 10.0, 20.0],
            }),
            child: x,
        });
        let compiled = CompiledExpr::new(&arena, interp);
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &[], &[], &[]);
        assert_eq!(result, &[15.0]);
    }

    #[test]
    fn test_eval_conditional() {
        let mut arena = Arena::new();
        let selector = arena.alloc(Node::Scalar(2.0));
        let branch1 = arena.alloc(Node::Scalar(100.0));
        let branch2 = arena.alloc(Node::Scalar(200.0));
        let branch3 = arena.alloc(Node::Scalar(300.0));
        let cond = arena.alloc(Node::Conditional {
            selector,
            branches: vec![branch1, branch2, branch3],
        });
        let compiled = CompiledExpr::new(&arena, cond);
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &[], &[], &[]);
        assert_eq!(result, &[200.0]);
    }

    #[test]
    fn test_from_ir() {
        use crate::ir::TypedIr;

        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let two = arena.alloc(Node::Scalar(2.0));
        let expr = arena.alloc(Node::Mul(two, y));

        let ir = TypedIr::from_arena(&arena, expr);
        let compiled = CompiledExpr::from_ir(ir);

        let y = [1.0, 2.0, 3.0];
        let mut s = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut s, 0.0, &y, &[], &[]);
        assert_eq!(result, &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_ir_accessor() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::Scalar(1.0));
        let compiled = CompiledExpr::new(&arena, x);

        // Verify we can access the IR
        assert_eq!(compiled.ir().output_len(), 1);
        assert_eq!(compiled.ir().instructions().len(), 1);
    }

    #[test]
    fn test_eval_interpolant_breakpoint_exactness() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let interp = arena.alloc(Node::Interpolant1DLinear {
            data: Box::new(InterpolantData {
                x_data: vec![0.0, 1.0, 3.0, 7.0],
                y_data: vec![10.0, 20.0, 60.0, -5.0],
            }),
            child: x,
        });
        let compiled = CompiledExpr::new(&arena, interp);
        let mut s = vec![0.0; compiled.scratch_len()];
        for (xv, expected) in [(0.0, 10.0), (1.0, 20.0), (3.0, 60.0), (7.0, -5.0)] {
            let result = compiled.eval(&mut s, 0.0, &[xv], &[], &[]);
            assert!(
                (result[0] - expected).abs() < 1e-14,
                "at x={xv}: expected {expected}, got {}",
                result[0]
            );
        }
    }

    #[test]
    fn test_eval_interpolant_linear_extrapolation() {
        // Outside the data domain, linear interp extends the boundary segment
        // (matches scipy interp1d(fill_value="extrapolate") and casadi interpn_linear).
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let interp = arena.alloc(Node::Interpolant1DLinear {
            data: Box::new(InterpolantData {
                x_data: vec![1.0, 2.0, 3.0],
                y_data: vec![100.0, 200.0, 300.0], // slope 100 everywhere
            }),
            child: x,
        });
        let compiled = CompiledExpr::new(&arena, interp);
        let mut s = vec![0.0; compiled.scratch_len()];

        // Below range: extend first segment (slope 100): y(-5) = 100 + 100*(-5-1) = -500
        let below = compiled.eval(&mut s, 0.0, &[-5.0], &[], &[]);
        assert_eq!(below, &[-500.0]);

        // Above range: extend last segment: y(5) = 300 + 100*(5-3) = 500
        let above = compiled.eval(&mut s, 0.0, &[5.0], &[], &[]);
        assert_eq!(above, &[500.0]);
    }

    #[test]
    fn test_eval_conditional_out_of_range_fills_zero() {
        let mut arena = Arena::new();
        let branch1 = arena.alloc(Node::Scalar(100.0));
        let branch2 = arena.alloc(Node::Scalar(200.0));
        let branches = vec![branch1, branch2];

        // selector = 0 is below range (branches are 1-indexed)
        let sel0 = arena.alloc(Node::Scalar(0.0));
        let cond = arena.alloc(Node::Conditional {
            selector: sel0,
            branches: branches.clone(),
        });
        let compiled = CompiledExpr::new(&arena, cond);
        let mut s = vec![0.0; compiled.scratch_len()];
        assert_eq!(compiled.eval(&mut s, 0.0, &[], &[], &[]), &[0.0]);

        // selector = 10 is above range
        let sel10 = arena.alloc(Node::Scalar(10.0));
        let cond2 = arena.alloc(Node::Conditional {
            selector: sel10,
            branches: branches.clone(),
        });
        let compiled2 = CompiledExpr::new(&arena, cond2);
        let mut s2 = vec![0.0; compiled2.scratch_len()];
        assert_eq!(compiled2.eval(&mut s2, 0.0, &[], &[], &[]), &[0.0]);

        // selector = -1 is below range
        let sel_neg = arena.alloc(Node::Scalar(-1.0));
        let cond3 = arena.alloc(Node::Conditional {
            selector: sel_neg,
            branches,
        });
        let compiled3 = CompiledExpr::new(&arena, cond3);
        let mut s3 = vec![0.0; compiled3.scratch_len()];
        assert_eq!(compiled3.eval(&mut s3, 0.0, &[], &[], &[]), &[0.0]);
    }

    #[test]
    fn test_eval_conditional_nan_fills_zero() {
        let mut arena = Arena::new();
        let sel = arena.alloc(Node::Scalar(f64::NAN));
        let branch = arena.alloc(Node::Scalar(999.0));
        let cond = arena.alloc(Node::Conditional {
            selector: sel,
            branches: vec![branch],
        });
        let compiled = CompiledExpr::new(&arena, cond);
        let mut s = vec![0.0; compiled.scratch_len()];
        assert_eq!(compiled.eval(&mut s, 0.0, &[], &[], &[]), &[0.0]);
    }

    #[test]
    fn test_eval_interpolant_cubic() {
        // One interval [0, 2] with p(dx) = 1 + 2*dx + 3*dx^2 + 4*dx^3.
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let interp = arena.alloc(Node::Interpolant1DCubic {
            data: Box::new(CubicInterpolantData {
                breakpoints: vec![0.0, 2.0],
                coeffs: vec![[1.0, 2.0, 3.0, 4.0]],
            }),
            child: x,
        });
        let compiled = CompiledExpr::new(&arena, interp);
        let mut s = vec![0.0; compiled.scratch_len()];

        // In-interval at x=1 (dx=1): 1+2+3+4 = 10
        assert_eq!(compiled.eval(&mut s, 0.0, &[1.0], &[], &[]), &[10.0]);
        // At left breakpoint x=0 (dx=0): 1
        assert_eq!(compiled.eval(&mut s, 0.0, &[0.0], &[], &[]), &[1.0]);
        // Extrapolate right at x=3 (clamp to interval 0, dx=3): 1+6+27+108 = 142
        assert_eq!(compiled.eval(&mut s, 0.0, &[3.0], &[], &[]), &[142.0]);
    }

    #[test]
    fn test_eval_interpolant_cubic_breakpoints() {
        // Discontinuous constants pin interval selection at interior points, at
        // breakpoints (right-continuous like scipy PPoly) and past the right edge.
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let interp = arena.alloc(Node::Interpolant1DCubic {
            data: Box::new(CubicInterpolantData {
                breakpoints: vec![0.0, 1.0, 3.0],
                // interval 0: p(dx)=5 (constant); interval 1: p(dx)=7 (constant)
                coeffs: vec![[5.0, 0.0, 0.0, 0.0], [7.0, 0.0, 0.0, 0.0]],
            }),
            child: x,
        });
        let compiled = CompiledExpr::new(&arena, interp);
        let mut s = vec![0.0; compiled.scratch_len()];
        // Interior points
        assert_eq!(compiled.eval(&mut s, 0.0, &[0.5], &[], &[]), &[5.0]);
        assert_eq!(compiled.eval(&mut s, 0.0, &[2.0], &[], &[]), &[7.0]);
        // Breakpoints: left edge, interior knot (right interval), right edge
        assert_eq!(compiled.eval(&mut s, 0.0, &[0.0], &[], &[]), &[5.0]);
        assert_eq!(compiled.eval(&mut s, 0.0, &[1.0], &[], &[]), &[7.0]);
        assert_eq!(compiled.eval(&mut s, 0.0, &[3.0], &[], &[]), &[7.0]);
    }

    #[test]
    fn test_eval_interpolant_nd_bilinear() {
        // One cell [0,2]x[0,2] with p(dx0,dx1) = 1 + 3*dx1 + 2*dx0 + 4*dx0*dx1.
        // Coeff layout: index = a0*order + a1 -> [c00, c01, c10, c11].
        let mut arena = Arena::new();
        let x0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let x1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let interp = arena.alloc(Node::InterpolantNd {
            data: Box::new(NdInterpolantData {
                breakpoints: vec![vec![0.0, 2.0], vec![0.0, 2.0]],
                coeffs: vec![1.0, 3.0, 2.0, 4.0],
                order: 2,
            }),
            children: vec![x0, x1],
        });
        let compiled = CompiledExpr::new(&arena, interp);
        let mut s = vec![0.0; compiled.scratch_len()];

        // In-cell at (1,1): 1 + 3 + 2 + 4 = 10
        assert_eq!(compiled.eval(&mut s, 0.0, &[1.0, 1.0], &[], &[]), &[10.0]);
        // Corner (0,0): constant term only
        assert_eq!(compiled.eval(&mut s, 0.0, &[0.0, 0.0], &[], &[]), &[1.0]);
        // Extrapolate both axes at (3,4): 1 + 3*4 + 2*3 + 4*3*4 = 67
        assert_eq!(compiled.eval(&mut s, 0.0, &[3.0, 4.0], &[], &[]), &[67.0]);
    }

    #[test]
    fn test_eval_interpolant_nd_cell_selection() {
        // Two cells along axis 0 x one along axis 1; discontinuous constants pin
        // cell choice, right-continuous like scipy and clamped at both edges.
        let mut arena = Arena::new();
        let x0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let x1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let interp = arena.alloc(Node::InterpolantNd {
            data: Box::new(NdInterpolantData {
                breakpoints: vec![vec![0.0, 1.0, 3.0], vec![0.0, 1.0]],
                // cell (0,0): p=5; cell (1,0): p=7
                coeffs: vec![5.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0],
                order: 2,
            }),
            children: vec![x0, x1],
        });
        let compiled = CompiledExpr::new(&arena, interp);
        let mut s = vec![0.0; compiled.scratch_len()];
        assert_eq!(compiled.eval(&mut s, 0.0, &[0.5, 0.5], &[], &[]), &[5.0]);
        assert_eq!(compiled.eval(&mut s, 0.0, &[2.0, 0.5], &[], &[]), &[7.0]);
        // Interior knot x0=1: right cell; edges clamp to boundary cells.
        assert_eq!(compiled.eval(&mut s, 0.0, &[1.0, 0.5], &[], &[]), &[7.0]);
        assert_eq!(compiled.eval(&mut s, 0.0, &[0.0, 0.5], &[], &[]), &[5.0]);
        assert_eq!(compiled.eval(&mut s, 0.0, &[3.0, 0.5], &[], &[]), &[7.0]);
    }

    #[test]
    fn test_eval_interpolant_nd_tricubic() {
        // One cell [0,2]^3, order 4: p = 3 + dx0^3 + 2*dx1^2 + 5*dx2.
        // Coeff index = (a0*4 + a1)*4 + a2.
        let mut arena = Arena::new();
        let x0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let x1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let x2 = arena.alloc(Node::StateVector { start: 2, end: 3 });
        let mut coeffs = vec![0.0; 64];
        coeffs[0] = 3.0; // (0,0,0)
        coeffs[(3 * 4) * 4] = 1.0; // dx0^3
        coeffs[2 * 4] = 2.0; // dx1^2
        coeffs[1] = 5.0; // dx2^1
        let interp = arena.alloc(Node::InterpolantNd {
            data: Box::new(NdInterpolantData {
                breakpoints: vec![vec![0.0, 2.0], vec![0.0, 2.0], vec![0.0, 2.0]],
                coeffs,
                order: 4,
            }),
            children: vec![x0, x1, x2],
        });
        let compiled = CompiledExpr::new(&arena, interp);
        let mut s = vec![0.0; compiled.scratch_len()];
        // (1,1,1): 3 + 1 + 2 + 5 = 11
        assert_eq!(
            compiled.eval(&mut s, 0.0, &[1.0, 1.0, 1.0], &[], &[]),
            &[11.0]
        );
        // Extrapolate axis 0 at (3,0,0): 3 + 27 = 30
        assert_eq!(
            compiled.eval(&mut s, 0.0, &[3.0, 0.0, 0.0], &[], &[]),
            &[30.0]
        );
    }

    #[test]
    fn test_eval_interpolant_nd_vector_children_broadcast() {
        // p(dx0,dx1) = dx0 + 10*dx1 on one cell [0,10]x[0,10]; vector child
        // (len 3) on axis 0, length-1 (broadcast) child on axis 1.
        let mut arena = Arena::new();
        let x0 = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let x1 = arena.alloc(Node::StateVector { start: 3, end: 4 });
        let interp = arena.alloc(Node::InterpolantNd {
            data: Box::new(NdInterpolantData {
                breakpoints: vec![vec![0.0, 10.0], vec![0.0, 10.0]],
                coeffs: vec![0.0, 10.0, 1.0, 0.0],
                order: 2,
            }),
            children: vec![x0, x1],
        });
        let compiled = CompiledExpr::new(&arena, interp);
        let mut s = vec![0.0; compiled.scratch_len()];
        let r = compiled.eval(&mut s, 0.0, &[1.0, 2.0, 3.0, 0.5], &[], &[]);
        assert_eq!(r, &[6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_eval_conditional_half_open_window() {
        // Selector matching uses (branch_index - 0.5, branch_index + 0.5)
        // branch_index is 1-indexed: branch 0 matches selector ∈ (0.5, 1.5)
        let mut arena = Arena::new();
        let branch1 = arena.alloc(Node::Scalar(100.0));
        let branch2 = arena.alloc(Node::Scalar(200.0));
        let branches = vec![branch1, branch2];

        // 0.6 is in (0.5, 1.5) → selects branch 0
        let sel = arena.alloc(Node::Scalar(0.6));
        let cond = arena.alloc(Node::Conditional {
            selector: sel,
            branches: branches.clone(),
        });
        let compiled = CompiledExpr::new(&arena, cond);
        let mut s = vec![0.0; compiled.scratch_len()];
        assert_eq!(compiled.eval(&mut s, 0.0, &[], &[], &[]), &[100.0]);

        // 1.4 is in (0.5, 1.5) → selects branch 0
        let sel2 = arena.alloc(Node::Scalar(1.4));
        let cond2 = arena.alloc(Node::Conditional {
            selector: sel2,
            branches: branches.clone(),
        });
        let compiled2 = CompiledExpr::new(&arena, cond2);
        let mut s2 = vec![0.0; compiled2.scratch_len()];
        assert_eq!(compiled2.eval(&mut s2, 0.0, &[], &[], &[]), &[100.0]);

        // Exactly 0.5 is NOT in the open interval (0.5, 1.5) → fills zero
        let sel3 = arena.alloc(Node::Scalar(0.5));
        let cond3 = arena.alloc(Node::Conditional {
            selector: sel3,
            branches,
        });
        let compiled3 = CompiledExpr::new(&arena, cond3);
        let mut s3 = vec![0.0; compiled3.scratch_len()];
        assert_eq!(compiled3.eval(&mut s3, 0.0, &[], &[], &[]), &[0.0]);
    }
}
