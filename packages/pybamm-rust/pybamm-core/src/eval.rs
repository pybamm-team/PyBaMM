//! Scalar interpreter for a `TypedIr` tape, including tangent inputs for forward-mode AD.

// The op helpers must inline as one chain into the interpreter loop.
#![allow(clippy::inline_always)]

use crate::arena::DagRef;
use crate::branch_regions::active_branch;
use crate::ir::ids::{NdInterpolantIdx, SlotTableIdx};
use crate::ir::{
    BinaryOp, BroadcastKind, ConstPool, Instruction, Slot, SplitEvalInfo, TypedIr, UnaryOp,
};
use crate::node::{CsrData, InterpolantData, NdInterpolantData};

/// `match $op`, expanding `$body` in each arm with `$f` bound to that arm's function.
///
/// Each arm binds its own closure type, so `$body` monomorphises per op.
macro_rules! dispatch_op {
    ($op:expr, $f:ident => $body:expr, { $($variant:path => $func:expr,)+ }) => {
        match $op {
            $($variant => {
                let $f = $func;
                $body
            },)+
        }
    };
}
pub(crate) use dispatch_op;

/// Expand `$body` with `$f` bound to `$op`'s scalar `Fn(f64, f64) -> f64`.
///
/// The one definition of each [`BinaryOp`]'s value, shared by every interpreter so
/// their primal results cannot drift apart. Operations IEEE 754 specifies exactly use
/// `f64`'s; transcendentals and `min`/`max`, which `f64` leaves non-deterministic, use
/// `libm`'s, so values are bitwise the same on every target.
macro_rules! with_binary_op {
    ($op:expr, $f:ident => $body:expr) => {
        $crate::eval::dispatch_op!($op, $f => $body, {
            $crate::ir::BinaryOp::Add => |x: f64, y: f64| x + y,
            $crate::ir::BinaryOp::Sub => |x: f64, y: f64| x - y,
            $crate::ir::BinaryOp::Mul => |x: f64, y: f64| x * y,
            $crate::ir::BinaryOp::Div => |x: f64, y: f64| x / y,
            $crate::ir::BinaryOp::Pow => libm::pow,
            $crate::ir::BinaryOp::Minimum => libm::fminimum_num,
            $crate::ir::BinaryOp::Maximum => libm::fmaximum_num,
            $crate::ir::BinaryOp::Modulo => |x: f64, y: f64| x % y,
            $crate::ir::BinaryOp::Hypot => libm::hypot,
            $crate::ir::BinaryOp::EqualHeaviside => {
                |x: f64, y: f64| if x <= y { 1.0 } else { 0.0 }
            },
            $crate::ir::BinaryOp::NotEqualHeaviside => {
                |x: f64, y: f64| if x < y { 1.0 } else { 0.0 }
            },
            $crate::ir::BinaryOp::Equality => |x: f64, y: f64| {
                if (x - y).abs() < $crate::eval::EQUALITY_EPS { 1.0 } else { 0.0 }
            },
        })
    };
}
pub(crate) use with_binary_op;

/// Expand `$body` with `$f` bound to `$op`'s scalar `Fn(f64) -> f64`.
///
/// The [`UnaryOp`] counterpart of [`with_binary_op`].
macro_rules! with_unary_op {
    ($op:expr, $f:ident => $body:expr) => {
        $crate::eval::dispatch_op!($op, $f => $body, {
            $crate::ir::UnaryOp::Neg => |x: f64| -x,
            $crate::ir::UnaryOp::Abs => f64::abs,
            $crate::ir::UnaryOp::Sqrt => f64::sqrt,
            $crate::ir::UnaryOp::Exp => libm::exp,
            $crate::ir::UnaryOp::Log => libm::log,
            $crate::ir::UnaryOp::Sin => libm::sin,
            $crate::ir::UnaryOp::Cos => libm::cos,
            $crate::ir::UnaryOp::Tanh => libm::tanh,
            $crate::ir::UnaryOp::Sinh => libm::sinh,
            $crate::ir::UnaryOp::Cosh => libm::cosh,
            $crate::ir::UnaryOp::Arcsinh => libm::asinh,
            $crate::ir::UnaryOp::Arctan => libm::atan,
            $crate::ir::UnaryOp::Erf => $crate::eval::erf,
            $crate::ir::UnaryOp::Sign => $crate::eval::sign,
            $crate::ir::UnaryOp::Floor => f64::floor,
            $crate::ir::UnaryOp::Ceiling => f64::ceil,
        })
    };
}
pub(crate) use with_unary_op;

impl BinaryOp {
    /// This op's value at one pair of elements, as every interpreter computes it.
    #[inline]
    pub(crate) fn apply(self, x: f64, y: f64) -> f64 {
        with_binary_op!(self, f => f(x, y))
    }
}

impl UnaryOp {
    /// This op's value at one element, as every interpreter computes it.
    #[inline]
    pub(crate) fn apply(self, x: f64) -> f64 {
        with_unary_op!(self, f => f(x))
    }
}

/// The segment `[lo, lo + 1]` of `knots` (at least two) holding `x`: right-continuous at
/// an interior knot, so never a zero-width segment between repeated ones, and clamped to the
/// first or last segment outside the domain.
///
/// The one knot search behind every 1-D and N-D interpolant.
#[inline(always)]
fn locate_segment(knots: &[f64], x: f64) -> usize {
    debug_assert!(knots.len() >= 2);
    let n = knots.len();
    if x <= knots[0] {
        return 0;
    }
    if x >= knots[n - 1] {
        return n - 2;
    }
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = usize::midpoint(lo, hi);
        if knots[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Linear interpolation, extending the boundary segment outside the domain (no flat clamp).
///
/// Unfused `y0 + t * (y1 - y0)`, bit for bit what `CasADi`'s `interpn_linear` computes.
#[inline(always)]
pub(crate) fn interp_linear_1d(table: &InterpolantData, x: f64) -> f64 {
    let (x_data, y_data) = (&table.x_data, &table.y_data);
    if x_data.len() == 1 {
        return y_data[0];
    }
    let lo = locate_segment(x_data, x);
    let t = (x - x_data[lo]) / (x_data[lo + 1] - x_data[lo]);
    y_data[lo] + t * (y_data[lo + 1] - y_data[lo])
}

/// Slope of the linear interpolant at `x`, from the table's pre-computed slopes.
#[inline(always)]
pub(crate) fn interp_linear_1d_deriv(table: &InterpolantData, x: f64) -> f64 {
    let (x_data, slopes) = (&table.x_data, &table.slopes);
    if slopes.is_empty() {
        return 0.0;
    }
    slopes[locate_segment(x_data, x)]
}

/// Piecewise-cubic interpolation (cubic spline / pchip), power-basis Horner.
#[inline(always)]
pub(crate) fn interp_cubic_1d(breakpoints: &[f64], coeffs: &[[f64; 4]], x: f64) -> f64 {
    let i = locate_segment(breakpoints, x);
    let dx = x - breakpoints[i];
    let [c0, c1, c2, c3] = coeffs[i];
    c0 + (c1 + (c2 + c3 * dx) * dx) * dx
}

/// Derivative of the piecewise-cubic interpolant: `p'(dx) = c1 + 2*c2*dx + 3*c3*dx^2`.
#[inline(always)]
pub(crate) fn interp_cubic_1d_deriv(breakpoints: &[f64], coeffs: &[[f64; 4]], x: f64) -> f64 {
    let i = locate_segment(breakpoints, x);
    let dx = x - breakpoints[i];
    let [_c0, c1, c2, c3] = coeffs[i];
    c1 + (2.0 * c2 + 3.0 * c3 * dx) * dx
}

/// An N-D interpolant's cell geometry, computed once per instruction rather than per element.
#[derive(Clone, Copy)]
pub(crate) struct NdCell {
    /// Per-axis polynomial order.
    order: usize,
    ndim: usize,
    /// Coefficient stride of each axis within a cell: `order^(ndim - 1 - axis)`.
    strides: [usize; 3],
}

impl NdCell {
    pub(crate) fn of(interp: &NdInterpolantData) -> Self {
        let (order, ndim) = (interp.order as usize, interp.breakpoints.len());
        debug_assert!((2..=3).contains(&ndim) && order >= 2);
        let mut strides = [1; 3];
        for axis in (0..ndim - 1).rev() {
            strides[axis] = strides[axis + 1] * order;
        }
        Self {
            order,
            ndim,
            strides,
        }
    }

    pub(crate) const fn ndim(self) -> usize {
        self.ndim
    }

    /// The coefficients of the cell holding `coords`, with per-axis offsets written to `dxs`;
    /// clamped per axis like scipy `RegularGridInterpolator(fill_value=None)`.
    pub(crate) fn locate<'a>(
        self,
        interp: &'a NdInterpolantData,
        coords: &[f64],
        dxs: &mut [f64; 3],
    ) -> &'a [f64] {
        let mut cell = 0;
        for (a, knots) in interp.breakpoints.iter().enumerate() {
            let i = locate_segment(knots, coords[a]);
            dxs[a] = coords[a] - knots[i];
            cell = cell * (knots.len() - 1) + i;
        }
        let size = self.strides[0] * self.order;
        &interp.coeffs[cell * size..][..size]
    }

    /// The polynomial at offsets `dxs` within the cell whose coefficients are `coeffs`.
    pub(crate) fn value(self, coeffs: &[f64], dxs: &[f64; 3]) -> f64 {
        tensor_horner(coeffs, &dxs[..self.ndim], &self.strides, self.order)
    }

    /// Its partial derivative along `axis`.
    pub(crate) fn partial(self, coeffs: &[f64], dxs: &[f64; 3], axis: usize) -> f64 {
        debug_assert!(axis < self.ndim);
        tensor_horner_partial(coeffs, &dxs[..self.ndim], &self.strides, self.order, axis)
    }
}

/// Buffer offset of output element `i`'s coordinate in N-D interpolant input `source`;
/// a length-1 source broadcasts over the output.
#[inline(always)]
pub(crate) const fn nd_source_offset(source: Slot, i: usize) -> usize {
    source.offset_usize() + if source.is_scalar() { 0 } else { i }
}

/// Coordinates of output element `i` of an N-D interpolant, `read` from each source.
#[inline(always)]
pub(crate) fn nd_coords(sources: &[Slot], i: usize, read: impl Fn(usize) -> f64) -> [f64; 3] {
    let mut coords = [0.0; 3];
    for (coord, &source) in coords.iter_mut().zip(sources) {
        *coord = read(nd_source_offset(source, i));
    }
    coords
}

/// `InterpNd` (`axis` `None`) or `InterpNdPartial` into `out`, over `k` lanes of a
/// lane-minor buffer; the scalar interpreter is `k = 1`.
pub(crate) fn interp_nd_lanes(
    buf: &mut [f64],
    k: usize,
    consts: &ConstPool,
    interp_idx: NdInterpolantIdx,
    sources_idx: SlotTableIdx,
    axis: Option<usize>,
    out: Slot,
) {
    let interp = &consts.nd_interpolants[interp_idx.index()];
    let cell = NdCell::of(interp);
    let sources = &consts.interp_nd_sources[sources_idx.index()..][..cell.ndim()];
    let mut dxs = [0.0_f64; 3];
    for e in 0..out.len_usize() {
        for l in 0..k {
            let coords = nd_coords(sources, e, |offset| buf[offset * k + l]);
            let coeffs = cell.locate(interp, &coords, &mut dxs);
            buf[(out.offset_usize() + e) * k + l] = axis.map_or_else(
                || cell.value(coeffs, &dxs),
                |axis| cell.partial(coeffs, &dxs, axis),
            );
        }
    }
}

/// Evaluate an N-D tensor-product polynomial in nested Horner form.
///
/// `coeffs` has `order^(dxs.len())` entries, axis-0 power slowest, ascending; `strides[0]` is
/// axis 0's coefficient stride, `order^(dxs.len() - 1)`.
fn tensor_horner(coeffs: &[f64], dxs: &[f64], strides: &[usize], order: usize) -> f64 {
    if dxs.len() == 1 {
        let mut acc = coeffs[order - 1];
        for a in (0..order - 1).rev() {
            acc = acc * dxs[0] + coeffs[a];
        }
        return acc;
    }
    let (stride, inner_dxs, inner_strides) = (strides[0], &dxs[1..], &strides[1..]);
    let slab = |a: usize| &coeffs[a * stride..(a + 1) * stride];
    let mut acc = tensor_horner(slab(order - 1), inner_dxs, inner_strides, order);
    for a in (0..order - 1).rev() {
        acc = acc * dxs[0] + tensor_horner(slab(a), inner_dxs, inner_strides, order);
    }
    acc
}

/// Partial derivative of the N-D tensor-product polynomial along `axis`.
fn tensor_horner_partial(
    coeffs: &[f64],
    dxs: &[f64],
    strides: &[usize],
    order: usize,
    axis: usize,
) -> f64 {
    if axis == 0 && dxs.len() == 1 {
        let mut acc = (order - 1) as f64 * coeffs[order - 1];
        for a in (1..order - 1).rev() {
            acc = acc * dxs[0] + a as f64 * coeffs[a];
        }
        return acc;
    }
    let (stride, inner_dxs, inner_strides) = (strides[0], &dxs[1..], &strides[1..]);
    let slab = |a: usize| &coeffs[a * stride..(a + 1) * stride];
    if axis == 0 {
        let top = tensor_horner(slab(order - 1), inner_dxs, inner_strides, order);
        let mut acc = (order - 1) as f64 * top;
        for a in (1..order - 1).rev() {
            acc = acc * dxs[0] + a as f64 * tensor_horner(slab(a), inner_dxs, inner_strides, order);
        }
        return acc;
    }
    let partial =
        |a: usize| tensor_horner_partial(slab(a), inner_dxs, inner_strides, order, axis - 1);
    let mut acc = partial(order - 1);
    for a in (0..order - 1).rev() {
        acc = acc * dxs[0] + partial(a);
    }
    acc
}

/// Tolerance `BinaryOp::Equality` compares within.
pub(crate) const EQUALITY_EPS: f64 = 1e-14;

/// The error function, to within an ulp.
///
/// Must be accurate, as `Erf` is differentiated exactly as `2/sqrt(pi) * exp(-a^2)`.
#[inline(always)]
pub(crate) fn erf(x: f64) -> f64 {
    libm::erf(x)
}

/// Sign function with `sign(0) = 0` (unlike `f64::signum`).
#[inline(always)]
pub(crate) fn sign(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x.signum() }
}

/// `MaxReduce`: the largest of `values` by strict `>` from `-inf`, so a NaN never wins.
#[inline(always)]
pub(crate) fn max_reduce(values: impl IntoIterator<Item = f64>) -> f64 {
    values
        .into_iter()
        .fold(f64::NEG_INFINITY, |best, v| if v > best { v } else { best })
}

/// `MinReduce`: the smallest of `values` by strict `<` from `+inf`, so a NaN never wins.
#[inline(always)]
pub(crate) fn min_reduce(values: impl IntoIterator<Item = f64>) -> f64 {
    values
        .into_iter()
        .fold(f64::INFINITY, |best, v| if v < best { v } else { best })
}

/// Position of the element [`max_reduce`] (`is_max`) or [`min_reduce`] returns, `0` if none:
/// the one `ReduceArgSelect` picks and reverse mode credits. Ties keep the earliest.
#[inline(always)]
pub(crate) fn arg_reduce(values: impl IntoIterator<Item = f64>, is_max: bool) -> usize {
    // The first non-NaN element is the incumbent, so a NaN never wins, even against
    // an all `-inf` (`+inf`) remainder the reduction returns.
    let mut best: Option<(usize, f64)> = None;
    for (i, v) in values.into_iter().enumerate() {
        let wins = match best {
            None => !v.is_nan(),
            Some((_, b)) if is_max => v > b,
            Some((_, b)) => v < b,
        };
        if wins {
            best = Some((i, v));
        }
    }
    best.map_or(0, |(i, _)| i)
}

/// Split `buf` into a read window `[src, src+src_len)` and a disjoint write window
/// `[dst, dst+dst_len)`.
#[inline(always)]
pub(crate) fn split_read_write(
    buf: &mut [f64],
    src: usize,
    src_len: usize,
    dst: usize,
    dst_len: usize,
) -> (&[f64], &mut [f64]) {
    debug_assert!(
        src + src_len <= dst || dst + dst_len <= src,
        "read and write windows overlap (src={src}+{src_len}, dst={dst}+{dst_len})"
    );
    if dst < src {
        let (left, right) = buf.split_at_mut(src);
        (&right[..src_len], &mut left[dst..dst + dst_len])
    } else {
        let (left, right) = buf.split_at_mut(dst);
        (&left[src..src + src_len], &mut right[..dst_len])
    }
}

/// Split `buf` into read windows `a`, `b` and a write window `dst`, each `(start, len)`.
///
/// `dst` must be disjoint from `a` and `b`, which may overlap each other.
#[inline(always)]
pub(crate) fn split_two_read_write(
    buf: &mut [f64],
    a: (usize, usize),
    b: (usize, usize),
    (dst, dst_len): (usize, usize),
) -> (&[f64], &[f64], &mut [f64]) {
    let (left, rest) = buf.split_at_mut(dst);
    let (dst_s, right) = rest.split_at_mut(dst_len);
    let (left, right): (&[f64], &[f64]) = (left, right);
    let read = |(start, len): (usize, usize)| {
        debug_assert!(
            start + len <= dst || dst + dst_len <= start,
            "read window {start}+{len} overlaps write window {dst}+{dst_len}"
        );
        if start + len <= dst {
            &left[start..start + len]
        } else {
            &right[start - (dst + dst_len)..][..len]
        }
    };
    (read(a), read(b), dst_s)
}

/// Each row of `csr` as its stored `(columns, values)`, in row order.
///
/// The one row walk behind every sparse product; `CsrData::try_new` guarantees the splits fit.
#[inline(always)]
pub(crate) fn csr_rows(csr: &CsrData) -> impl Iterator<Item = (&[usize], &[f64])> {
    let (mut cols, mut vals) = (&csr.indices[..], &csr.data[..]);
    csr.indptr.windows(2).map(move |w| {
        let (row_cols, rest_cols) = cols.split_at(w[1] - w[0]);
        let (row_vals, rest_vals) = vals.split_at(w[1] - w[0]);
        (cols, vals) = (rest_cols, rest_vals);
        (row_cols, row_vals)
    })
}

/// Apply a binary operation with broadcasting.
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
            let (b_s, d_s) = split_read_write(buf, b, len, dst, len);
            for (o, &y) in d_s.iter_mut().zip(b_s) {
                *o = f(scalar, y);
            }
        },
        BroadcastKind::VectorScalar => {
            let scalar = buf[b];
            let (a_s, d_s) = split_read_write(buf, a, len, dst, len);
            for (o, &x) in d_s.iter_mut().zip(a_s) {
                *o = f(x, scalar);
            }
        },
        BroadcastKind::VectorVector => zip_apply(buf, f, a, b, dst, len),
    }
}

/// Write `f(a[i], b[i])` to `dst[i]` for each `i < len`, with `dst` disjoint from `a` and `b`.
#[inline(always)]
pub(crate) fn zip_apply<F: Fn(f64, f64) -> f64>(
    buf: &mut [f64],
    f: F,
    a: usize,
    b: usize,
    dst: usize,
    len: usize,
) {
    let (a_s, b_s, d_s) = split_two_read_write(buf, (a, len), (b, len), (dst, len));
    for ((o, &x), &y) in d_s.iter_mut().zip(a_s).zip(b_s) {
        *o = f(x, y);
    }
}

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
    with_binary_op!(op, f => broadcast_apply(buf, f, a, b, dst, len, kind));
}

/// Write `f` of each element of `[src, src+len)` to the disjoint `[dst, dst+len)`.
#[inline(always)]
pub(crate) fn unary_apply<F: Fn(f64) -> f64>(
    buf: &mut [f64],
    f: F,
    src: usize,
    dst: usize,
    len: usize,
) {
    let (src_s, dst_s) = split_read_write(buf, src, len, dst, len);
    for (o, &x) in dst_s.iter_mut().zip(src_s) {
        *o = f(x);
    }
}

#[inline(always)]
fn eval_unary_op(buf: &mut [f64], op: UnaryOp, src: usize, dst: usize, len: usize) {
    with_unary_op!(op, f => unary_apply(buf, f, src, dst, len));
}

/// Seed vectors for tangent evaluation (forward-mode AD); `None` reads as zero.
#[derive(Debug, Default)]
pub struct TangentInputs<'a> {
    /// Tangent of the state vector, indexed like `y`.
    pub dy: Option<&'a [f64]>,
    /// Tangent of the parameters, indexed by `InputParameter::index`, not input offset.
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

/// Execute a slice of instructions against a shared buffer, returning how many it
/// skipped in inactive `Dispatch` blocks.
///
/// Counting skips rather than runs keeps bookkeeping out of the per-instruction path.
#[inline]
fn eval_instructions(
    buf: &mut [f64],
    instructions: &[Instruction],
    consts: &ConstPool,
    ctx: EvalContext<'_>,
) -> usize {
    let mut skipped = 0_usize;
    let mut pc = 0_usize;
    while pc < instructions.len() {
        match instructions[pc] {
            Instruction::Dispatch {
                selector,
                blocks_idx,
                blocks_len,
                span_len,
            } => {
                let span_end = pc + span_len.index();
                let active = active_branch(buf[selector.index()], blocks_len.index());
                let ran = active.map_or(0, |active| {
                    let block = consts.branch_blocks[blocks_idx.index() + active].bounds(pc);
                    block.len() - eval_instructions(buf, &instructions[block], consts, ctx)
                });
                skipped += span_end - (pc + 1) - ran;
                pc = span_end;
                continue;
            },
            Instruction::LoadScalar { value, dst } => {
                buf[dst.index()] = value;
            },
            Instruction::LoadTime { dst } => {
                buf[dst.index()] = ctx.primal().t;
            },
            Instruction::LoadArray { data_idx, len, dst } => {
                let src = consts.get_array(data_idx, len);
                buf[dst.index()..dst.index() + len.index()].copy_from_slice(src);
            },
            Instruction::FillZero { dst, len } => {
                buf[dst.index()..dst.index() + len.index()].fill(0.0);
            },
            Instruction::LoadStateVector { start, width, dst } => {
                let primal = ctx.primal();
                let (start, width) = (start.index(), width.index());
                buf[dst.index()..dst.index() + width]
                    .copy_from_slice(&primal.y[start..start + width]);
            },
            Instruction::LoadStateVectorDot { start, width, dst } => {
                let primal = ctx.primal();
                let (start, width) = (start.index(), width.index());
                buf[dst.index()..dst.index() + width]
                    .copy_from_slice(&primal.y_dot[start..start + width]);
            },
            Instruction::LoadInputParameter { offset, width, dst } => {
                let primal = ctx.primal();
                let offset = offset.index();
                let width = width.index();
                buf[dst.index()..dst.index() + width]
                    .copy_from_slice(&primal.inputs[offset..offset + width]);
            },
            Instruction::LoadTangentState { start, width, dst } => {
                let (start, width) = (start.index(), width.index());
                let slot = &mut buf[dst.index()..dst.index() + width];
                if let Some(dy) = ctx.tangent_state() {
                    slot.copy_from_slice(&dy[start..start + width]);
                } else {
                    slot.fill(0.0);
                }
            },
            Instruction::LoadTangentParameter { index, dst } => {
                buf[dst.index()] = ctx.tangent_parameter(index.index());
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
                    a.index(),
                    b.index(),
                    dst.index(),
                    len.index(),
                    kind,
                );
            },

            Instruction::Unary { op, src, dst, len } => {
                eval_unary_op(buf, op, src.index(), dst.index(), len.index());
            },

            Instruction::MaxReduce { src, src_len, dst } => {
                let src = src.index();
                buf[dst.index()] = max_reduce(buf[src..src + src_len.index()].iter().copied());
            },

            Instruction::MinReduce { src, src_len, dst } => {
                let src = src.index();
                buf[dst.index()] = min_reduce(buf[src..src + src_len.index()].iter().copied());
            },

            Instruction::ReduceArgSelect {
                basis_src,
                picker_src,
                len,
                is_max,
                dst,
            } => {
                let picker = picker_src.index();
                let best = arg_reduce(buf[picker..picker + len.index()].iter().copied(), is_max);
                buf[dst.index()] = buf[basis_src.index() + best];
            },

            Instruction::Index {
                src,
                start,
                dst,
                len,
            } => {
                let src = src.offset_by(start);
                buf.copy_within(src..src + len.index(), dst.index());
            },

            Instruction::Concat {
                sources_idx,
                sources_len,
                dst,
            } => {
                let mut write_pos = dst.index();
                for i in 0..sources_len.index() {
                    let slot = consts.concat_sources[sources_idx.index() + i];
                    buf.copy_within(slot.bounds(), write_pos);
                    write_pos += slot.len_usize();
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
                    vec_src.index(),
                    csr.shape.cols,
                    dst.index(),
                    csr.shape.rows,
                );
                for (o, (cols, vals)) in out.iter_mut().zip(csr_rows(csr)) {
                    let mut sum = 0.0;
                    for (&col, &val) in cols.iter().zip(vals) {
                        sum += val * x[col];
                    }
                    *o = sum;
                }
            },

            Instruction::DenseMatMul {
                mat_src,
                rows,
                cols,
                vec_src,
                dst,
            } => {
                let (mat_src, vec_src, dst) = (mat_src.index(), vec_src.index(), dst.index());
                let cols = cols.index();
                for row in 0..rows.index() {
                    let mat_row = &buf[mat_src + row * cols..][..cols];
                    let mut sum = 0.0;
                    for (&m, &v) in mat_row.iter().zip(&buf[vec_src..vec_src + cols]) {
                        sum += m * v;
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
                let interp = &consts.interpolants[interp_idx.index()];
                let f = |x| interp_linear_1d(interp, x);
                unary_apply(buf, f, src.index(), dst.index(), len.index());
            },

            Instruction::Interp1DLinearDeriv {
                interp_idx,
                src,
                dst,
                len,
            } => {
                let interp = &consts.interpolants[interp_idx.index()];
                let f = |x| interp_linear_1d_deriv(interp, x);
                unary_apply(buf, f, src.index(), dst.index(), len.index());
            },

            Instruction::Interp1DCubic {
                interp_idx,
                src,
                dst,
                len,
            } => {
                let interp = &consts.cubic_interpolants[interp_idx.index()];
                let f = |x| interp_cubic_1d(&interp.breakpoints, &interp.coeffs, x);
                unary_apply(buf, f, src.index(), dst.index(), len.index());
            },
            Instruction::Interp1DCubicDeriv {
                interp_idx,
                src,
                dst,
                len,
            } => {
                let interp = &consts.cubic_interpolants[interp_idx.index()];
                let f = |x| interp_cubic_1d_deriv(&interp.breakpoints, &interp.coeffs, x);
                unary_apply(buf, f, src.index(), dst.index(), len.index());
            },

            Instruction::InterpNd {
                interp_idx,
                sources_idx,
                dst,
                len,
            } => {
                let out = Slot::new(dst, len);
                interp_nd_lanes(buf, 1, consts, interp_idx, sources_idx, None, out);
            },
            Instruction::InterpNdPartial {
                interp_idx,
                sources_idx,
                axis,
                dst,
                len,
            } => {
                let (axis, out) = (Some(axis.index()), Slot::new(dst, len));
                interp_nd_lanes(buf, 1, consts, interp_idx, sources_idx, axis, out);
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
                match active_branch(buf[selector.index()], branches_len.index()) {
                    Some(i) => {
                        let slot = consts.branch_offsets[branches_idx.index() + i];
                        buf.copy_within(slot.offset_usize()..slot.offset_usize() + out_len, dst);
                    },
                    None => buf[dst..dst + out_len].fill(0.0),
                }
            },
        }
        pc += 1;
    }
    skipped
}

/// Zero-allocation expression evaluator.
///
/// Callers supply the scratch buffer, so one instance can be shared across concurrent solves.
#[derive(Debug, Clone)]
pub struct CompiledExpr {
    ir: TypedIr,
}

impl CompiledExpr {
    /// Compile an expression DAG into a new `CompiledExpr`.
    pub fn new(dag: DagRef<'_>) -> Self {
        Self::from_ir(TypedIr::from_dag(dag))
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

    /// [`eval_with_tangent`](Self::eval_with_tangent), also returning how many instructions ran.
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
        let instructions = self.ir.instructions();
        let skipped = eval_instructions(
            scratch,
            instructions,
            self.ir.consts(),
            EvalContext::with_primal(t, y, y_dot, inputs, tangent),
        );
        (
            instructions.len() - skipped,
            &scratch[self.ir.root_slot().bounds()],
        )
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

/// A [`CompiledExpr`] split into a primal and a tangent section.
///
/// No batch path; rewrapped via [`CompiledExpr::from_ir`], a tangent load errs in `eval_batch`:
///
/// ```compile_fail
/// # use pybamm_core::{Arena, DagRef, Node, SplitCompiledExpr};
/// # let mut arena = Arena::new();
/// # let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
/// let split = SplitCompiledExpr::compile(DagRef::new(&arena, y));
/// let _ = split.eval_batch(&mut [0.0; 8], 1, &[0.0], &[1.0], 1, &[]);
/// ```
#[derive(Debug, Clone)]
pub struct SplitCompiledExpr {
    expr: CompiledExpr,
    split: SplitEvalInfo,
}

impl SplitCompiledExpr {
    /// Lower and compile `dag` with primal and tangent instructions partitioned.
    pub fn compile(dag: DagRef<'_>) -> Self {
        let (ir, split) = TypedIr::from_dag_split_eval(dag);
        Self {
            expr: CompiledExpr::from_ir(ir),
            split,
        }
    }

    /// The whole tape, primal section first.
    #[inline]
    pub const fn ir(&self) -> &TypedIr {
        self.expr.ir()
    }

    /// Where the primal section ends, in instructions and in buffer slots.
    #[inline]
    pub const fn split_info(&self) -> SplitEvalInfo {
        self.split
    }

    /// Length of the scratch buffer both sections share.
    #[inline]
    pub const fn scratch_len(&self) -> usize {
        self.expr.scratch_len()
    }

    /// Both sections in one walk, for a single tangent.
    pub fn eval_with_tangent<'s>(
        &self,
        scratch: &'s mut [f64],
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
        tangent: &TangentInputs<'_>,
    ) -> &'s [f64] {
        self.expr
            .eval_with_tangent(scratch, t, y, y_dot, inputs, tangent)
    }

    /// Evaluate the primal section into `scratch`, returning the [`PrimalCache`] that
    /// runs the tangent sweep on the same buffer.
    pub fn eval_primal<'a>(
        &'a self,
        scratch: &'a mut [f64],
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
    ) -> PrimalCache<'a> {
        eval_instructions(
            scratch,
            &self.ir().instructions()[..self.split.primal_end],
            self.ir().consts(),
            EvalContext::with_primal(t, y, y_dot, inputs, None),
        );
        PrimalCache {
            expr: self,
            scratch,
        }
    }
}

/// Borrow-based typestate for split evaluation.
///
/// The only route to the tangent sweep, so tangent-before-primal is a compile error.
pub struct PrimalCache<'a> {
    expr: &'a SplitCompiledExpr,
    scratch: &'a mut [f64],
}

impl std::fmt::Debug for PrimalCache<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrimalCache").finish_non_exhaustive()
    }
}

impl PrimalCache<'_> {
    /// Evaluate the tangent section on the cached primal region, returning the root slice.
    pub fn eval_tangent(&mut self, tangent: &TangentInputs<'_>) -> &[f64] {
        let ir = self.expr.ir();
        eval_instructions(
            self.scratch,
            &ir.instructions()[self.expr.split.primal_end..],
            ir.consts(),
            EvalContext::tangent_only(tangent),
        );
        &self.scratch[ir.root_slot().bounds()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{Arena, NodeId};
    use crate::ir::Slot;
    use crate::ir::ids::{BufSlot, Width};
    use crate::node::{
        ArrayData, CsrData, CubicInterpolantData, InterpolantData, NdInterpolantData, Node, Shape,
    };
    use crate::params::ParamIdx;

    /// Compile `root` and evaluate it once at `(t, y)`, with no `y_dot` or inputs.
    fn eval_at(arena: &Arena, root: NodeId, t: f64, y: &[f64]) -> Vec<f64> {
        let compiled = CompiledExpr::new(DagRef::new(arena, root));
        let mut scratch = vec![0.0; compiled.scratch_len()];
        compiled.eval(&mut scratch, t, y, &[], &[]).to_vec()
    }

    fn scalar(arena: &mut Arena, value: f64) -> NodeId {
        arena.alloc(Node::Scalar(value))
    }

    fn array(arena: &mut Arena, data: &[f64]) -> NodeId {
        arena.alloc(Node::Array(Box::new(ArrayData {
            data: data.to_vec(),
            shape: Shape::vector(data.len()),
        })))
    }

    #[test]
    fn erf_matches_the_reference_to_an_ulp() {
        // scipy.special.erf, printed at full precision.
        let reference = [
            (0.0, 0.0),
            (1e-8, 1.128_379_167_095_512_5e-8),
            (0.1, 0.112_462_916_018_284_9),
            (0.5, 0.520_499_877_813_046_5),
            (0.843_75, 0.767_225_661_232_341_6),
            (1.0, 0.842_700_792_949_714_8),
            (1.25, 0.922_900_128_256_458_3),
            (2.0, 0.995_322_265_018_952_7),
            (3.0, 0.999_977_909_503_001_4),
            (4.755_430_192_319_827, 0.999_999_999_982_464_1),
            (6.0, 1.0),
        ];
        for (x, want) in reference {
            for signum in [1.0, -1.0] {
                let got = erf(signum * x);
                let want = signum * want;
                assert!(
                    (got - want).abs() <= 2.0 * f64::EPSILON * want.abs(),
                    "erf({}) = {got:.17e}, reference {want:.17e}",
                    signum * x
                );
            }
        }
        // scipy.special.erfc; `1 - erf(x)` is exact, hence the absolute floor.
        let tail = [
            (3.0, 2.209_049_699_858_544_5e-5),
            (4.0, 1.541_725_790_028_002e-8),
            (4.755_430_192_319_827, 1.753_587_985_582_014e-11),
        ];
        for (x, want) in tail {
            let got = 1.0 - erf(x);
            assert!(
                (got - want).abs() <= 1e-5 * want + 2.0 * f64::EPSILON,
                "1 - erf({x}) = {got:.17e}, erfc reference {want:.17e}"
            );
        }

        assert_eq!(
            erf(0.0).to_bits(),
            0.0_f64.to_bits(),
            "erf(0) must be exactly +0"
        );
        assert_eq!(
            erf(-0.0).to_bits(),
            (-0.0_f64).to_bits(),
            "erf(-0) must be exactly -0"
        );
    }

    #[test]
    fn the_erf_derivative_rule_matches_the_primal_it_differentiates() {
        let rule = |x: f64| std::f64::consts::FRAC_2_SQRT_PI * (-x * x).exp();
        for i in 0..=60 {
            let x = 0.05 + f64::from(i) * 0.05;
            let step = 1e-4 * x;
            let quotient = (erf(x + step) - erf(x - step)) / (2.0 * step);
            // 1e-6 relative covers the central difference's O(step^2) truncation.
            assert!(
                (quotient - rule(x)).abs() <= 1e-6 * rule(x) + 1e-12,
                "at x = {x}: quotient {quotient:.12e} vs rule {:.12e}",
                rule(x)
            );
        }
    }

    #[test]
    fn size_check() {
        use std::mem::size_of;
        // Guard hot per-node types against accidental bloat.
        assert_eq!(size_of::<BufSlot>(), 4);
        assert_eq!(size_of::<Width>(), 4);
        assert_eq!(size_of::<ParamIdx>(), 4);
        assert_eq!(size_of::<Slot>(), 8);
        assert_eq!(size_of::<NodeId>(), 4);
        assert_eq!(size_of::<BroadcastKind>(), 1);
        assert_eq!(size_of::<BinaryOp>(), 1);
        assert_eq!(size_of::<UnaryOp>(), 1);
    }

    #[test]
    fn single_nodes_evaluate_to_their_closed_form() {
        type Build = fn(&mut Arena) -> NodeId;
        let cases: [(&str, Build, &[f64]); 17] = [
            ("scalar", |a| scalar(a, 42.0), &[42.0]),
            ("time", |a| a.alloc(Node::Time), &[3.5]),
            (
                "state",
                |a| a.alloc(Node::StateVector { start: 1, end: 3 }),
                &[20.0, 30.0],
            ),
            ("array", |a| array(a, &[1.0, 2.0, 3.0]), &[1.0, 2.0, 3.0]),
            (
                "add scalars",
                |a| {
                    let (x, y) = (scalar(a, 2.0), scalar(a, 3.0));
                    a.alloc(Node::Add(x, y))
                },
                &[5.0],
            ),
            (
                "add vectors",
                |a| {
                    let x = array(a, &[1.0, 2.0, 3.0]);
                    let y = array(a, &[10.0, 20.0, 30.0]);
                    a.alloc(Node::Add(x, y))
                },
                &[11.0, 22.0, 33.0],
            ),
            (
                "scalar times vector",
                |a| {
                    let (s, v) = (scalar(a, 2.0), array(a, &[1.0, 2.0, 3.0]));
                    a.alloc(Node::Mul(s, v))
                },
                &[2.0, 4.0, 6.0],
            ),
            (
                "(2 + 3) * 4 - 1",
                |a| {
                    let (two, three) = (scalar(a, 2.0), scalar(a, 3.0));
                    let (four, one) = (scalar(a, 4.0), scalar(a, 1.0));
                    let sum = a.alloc(Node::Add(two, three));
                    let product = a.alloc(Node::Mul(sum, four));
                    a.alloc(Node::Sub(product, one))
                },
                &[19.0],
            ),
            (
                "neg",
                |a| {
                    let x = scalar(a, 5.0);
                    a.alloc(Node::Neg(x))
                },
                &[-5.0],
            ),
            (
                "sqrt",
                |a| {
                    let x = array(a, &[1.0, 4.0, 9.0]);
                    a.alloc(Node::Sqrt(x))
                },
                &[1.0, 2.0, 3.0],
            ),
            (
                "sin(0) + cos(0)",
                |a| {
                    let zero = scalar(a, 0.0);
                    let (sin, cos) = (a.alloc(Node::Sin(zero)), a.alloc(Node::Cos(zero)));
                    a.alloc(Node::Add(sin, cos))
                },
                &[1.0],
            ),
            (
                "max reduce",
                |a| {
                    let x = array(a, &[1.0, 5.0, 3.0]);
                    a.alloc(Node::MaxReduce(x))
                },
                &[5.0],
            ),
            (
                "index",
                |a| {
                    let child = array(a, &[10.0, 20.0, 30.0, 40.0, 50.0]);
                    a.alloc(Node::Index {
                        child,
                        start: 1,
                        end: 4,
                    })
                },
                &[20.0, 30.0, 40.0],
            ),
            (
                "concat",
                |a| {
                    let (x, y, z) = (scalar(a, 1.0), array(a, &[2.0, 3.0]), scalar(a, 4.0));
                    a.alloc(Node::Concat(vec![x, y, z]))
                },
                &[1.0, 2.0, 3.0, 4.0],
            ),
            (
                "sparse matmul",
                |a| {
                    let m = a.alloc(Node::SparseMatrix(Box::new(CsrData {
                        indptr: vec![0, 1, 2],
                        indices: vec![0, 1],
                        data: vec![2.0, 3.0],
                        shape: Shape::matrix(2, 3),
                    })));
                    let v = array(a, &[1.0, 2.0, 3.0]);
                    a.alloc(Node::MatMul(m, v))
                },
                &[2.0, 6.0],
            ),
            (
                "dense matmul",
                |a| {
                    let m = a.alloc(Node::Array(Box::new(ArrayData {
                        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // row-major 2x3
                        shape: Shape::matrix(2, 3),
                    })));
                    let v = array(a, &[1.0, 2.0, 3.0]);
                    a.alloc(Node::MatMul(m, v))
                },
                &[14.0, 32.0],
            ),
            (
                "linear interpolant",
                |a| {
                    let child = scalar(a, 1.5);
                    let table =
                        InterpolantData::try_new(vec![0.0, 1.0, 2.0], vec![0.0, 10.0, 20.0])
                            .expect("valid interpolant");
                    a.alloc(Node::Interpolant1DLinear {
                        data: Box::new(table),
                        child,
                    })
                },
                &[15.0],
            ),
        ];
        for (label, build, expected) in cases {
            let mut arena = Arena::new();
            let root = build(&mut arena);
            assert_eq!(
                eval_at(&arena, root, 3.5, &[10.0, 20.0, 30.0, 40.0]),
                expected,
                "{label}"
            );
        }
    }

    #[test]
    fn reduce_arg_select_picks_the_basis_at_the_first_extremum() {
        for (picker_values, is_max, expected) in [
            ([1.0, 5.0, 3.0], true, 20.0),
            ([1.0, 5.0, 3.0], false, 10.0),
            ([5.0, 5.0, 3.0], true, 10.0),
            ([5.0, 3.0, 3.0], false, 20.0),
        ] {
            let mut arena = Arena::new();
            let picker = array(&mut arena, &picker_values);
            let basis = array(&mut arena, &[10.0, 20.0, 30.0]);
            let node = arena.alloc(Node::ReduceArgSelect {
                basis,
                picker,
                is_max,
            });
            assert_eq!(
                eval_at(&arena, node, 0.0, &[]),
                [expected],
                "picker {picker_values:?}, is_max={is_max}"
            );
        }
    }

    /// The subgradient must credit the element the reduction returned, NaNs included.
    #[test]
    fn arg_select_skips_a_nan_as_its_reduction_does() {
        // A remainder at the reduction's starting value still beats a leading NaN.
        assert_eq!(arg_reduce([f64::NAN, f64::NEG_INFINITY], true), 1);
        assert_eq!(arg_reduce([f64::NAN, f64::INFINITY], false), 1);
        assert_eq!(arg_reduce([f64::NAN, f64::NAN], true), 0);

        let picker = [f64::NAN, 1.0, 3.0, f64::NAN, -2.0];
        assert_eq!(max_reduce(picker).to_bits(), 3.0_f64.to_bits());
        assert_eq!(min_reduce(picker).to_bits(), (-2.0_f64).to_bits());
        // Positions 2 and 4, where the reductions found 3 and -2.
        for (is_max, expected) in [(true, 30.0_f64), (false, 50.0)] {
            let mut arena = Arena::new();
            let picker = array(&mut arena, &picker);
            let basis = array(&mut arena, &[10.0, 20.0, 30.0, 40.0, 50.0]);
            let node = arena.alloc(Node::ReduceArgSelect {
                basis,
                picker,
                is_max,
            });
            let picked = eval_at(&arena, node, 0.0, &[])[0];
            assert_eq!(picked.to_bits(), expected.to_bits(), "is_max={is_max}");
        }
    }

    #[test]
    fn a_conditional_takes_the_branch_whose_open_window_holds_the_selector() {
        // Branch i (1-based) owns (i - 0.5, i + 0.5); any other selector, NaN included, is zero.
        for (selector, branches, expected) in [
            (2.0, &[100.0, 200.0, 300.0][..], 200.0),
            (0.6, &[100.0, 200.0], 100.0),
            (1.4, &[100.0, 200.0], 100.0),
            (0.5, &[100.0, 200.0], 0.0),
            (0.0, &[100.0, 200.0], 0.0),
            (10.0, &[100.0, 200.0], 0.0),
            (-1.0, &[100.0, 200.0], 0.0),
            (f64::NAN, &[999.0], 0.0),
        ] {
            let mut arena = Arena::new();
            let selector_id = scalar(&mut arena, selector);
            let branches = branches.iter().map(|&b| scalar(&mut arena, b)).collect();
            let cond = arena.alloc(Node::Conditional {
                selector: selector_id,
                branches,
            });
            assert_eq!(
                eval_at(&arena, cond, 0.0, &[]),
                [expected],
                "selector {selector}"
            );
        }
    }

    #[test]
    fn test_eval_interpolant_breakpoint_exactness() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let interp = arena.alloc(Node::Interpolant1DLinear {
            data: Box::new(
                InterpolantData::try_new(vec![0.0, 1.0, 3.0, 7.0], vec![10.0, 20.0, 60.0, -5.0])
                    .expect("valid interpolant"),
            ),
            child: x,
        });
        for (xv, expected) in [(0.0, 10.0), (1.0, 20.0), (3.0, 60.0), (7.0, -5.0)] {
            let result = eval_at(&arena, interp, 0.0, &[xv]);
            assert!(
                (result[0] - expected).abs() < 1e-14,
                "at x={xv}: expected {expected}, got {}",
                result[0]
            );
        }
    }

    #[test]
    fn linear_interpolant_matches_casadi_bit_for_bit() {
        // `casadi.MX.interpn_linear` values, as PyBaMM builds it; a fused multiply-add
        // misses every one of these by an ulp.
        let table = InterpolantData::try_new(vec![0.0, 0.1, 0.3, 0.7], vec![0.2, -1.3, 2.9, 0.05])
            .expect("valid interpolant");
        let casadi: [(f64, f64); 6] = [
            (0.079_896_907_216_494_84, -0.998_453_608_247_422_7),
            (0.141_752_577_319_587_62, -0.423_195_876_288_659_84),
            (0.152_061_855_670_103_1, -0.206_701_030_927_834_76),
            (0.162_371_134_020_618_53, 0.009_793_814_432_989_212),
            (-0.219_072_164_948_453_6, 3.486_082_474_226_804),
            (0.708_762_886_597_938_2, -0.012_435_567_010_309_79),
        ];
        for (x, want) in casadi {
            let got = interp_linear_1d(&table, x);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "at x={x}: {got:e} vs CasADi {want:e}"
            );
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn a_repeated_knot_is_a_right_continuous_step() {
        // As `casadi.MX.interpn_linear` takes it: the knot belongs to the segment after it.
        let table = InterpolantData::try_new(vec![0.0, 1.0, 1.0, 2.0], vec![0.0, 1.0, 3.0, 5.0])
            .expect("a repeated interior knot is valid");
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let interp = arena.alloc(Node::Interpolant1DLinear {
            data: Box::new(table.clone()),
            child: x,
        });
        let cases = [
            (-1.0, -1.0, 1.0),
            (0.5, 0.5, 1.0),
            (0.75, 0.75, 1.0),
            (1.0, 3.0, 2.0),
            (1.5, 4.0, 2.0),
            (3.0, 7.0, 2.0),
        ];
        for (xv, value, slope) in cases {
            assert_eq!(
                eval_at(&arena, interp, 0.0, &[xv]),
                [value],
                "value at {xv}"
            );
            assert_eq!(interp_linear_1d_deriv(&table, xv), slope, "slope at {xv}");
        }
    }

    #[test]
    fn test_eval_interpolant_linear_extrapolation() {
        // Matches scipy `interp1d(fill_value="extrapolate")`.
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let interp = arena.alloc(Node::Interpolant1DLinear {
            data: Box::new(
                InterpolantData::try_new(vec![1.0, 2.0, 3.0], vec![100.0, 200.0, 300.0])
                    .expect("valid interpolant"),
            ),
            child: x,
        });
        assert_eq!(eval_at(&arena, interp, 0.0, &[-5.0]), [-500.0]);
        assert_eq!(eval_at(&arena, interp, 0.0, &[5.0]), [500.0]);
    }

    #[test]
    fn test_eval_interpolant_cubic() {
        // p(dx) = 1 + 2*dx + 3*dx^2 + 4*dx^3 on [0, 2].
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let interp = arena.alloc(Node::Interpolant1DCubic {
            data: Box::new(CubicInterpolantData {
                breakpoints: vec![0.0, 2.0],
                coeffs: vec![[1.0, 2.0, 3.0, 4.0]],
            }),
            child: x,
        });
        for (xv, expected) in [(1.0, 10.0), (0.0, 1.0), (3.0, 142.0)] {
            assert_eq!(eval_at(&arena, interp, 0.0, &[xv]), [expected], "x={xv}");
        }
    }

    #[test]
    fn test_eval_interpolant_cubic_breakpoints() {
        // Right-continuous at breakpoints, like scipy `PPoly`.
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let interp = arena.alloc(Node::Interpolant1DCubic {
            data: Box::new(CubicInterpolantData {
                breakpoints: vec![0.0, 1.0, 3.0],
                coeffs: vec![[5.0, 0.0, 0.0, 0.0], [7.0, 0.0, 0.0, 0.0]],
            }),
            child: x,
        });
        for (xv, expected) in [(0.5, 5.0), (2.0, 7.0), (0.0, 5.0), (1.0, 7.0), (3.0, 7.0)] {
            assert_eq!(eval_at(&arena, interp, 0.0, &[xv]), [expected], "x={xv}");
        }
    }

    #[test]
    fn test_eval_interpolant_nd_bilinear() {
        // p = 1 + 3*dx1 + 2*dx0 + 4*dx0*dx1, coeffs indexed `a0*order + a1`.
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
        for (y, expected) in [([1.0, 1.0], 10.0), ([0.0, 0.0], 1.0), ([3.0, 4.0], 67.0)] {
            assert_eq!(eval_at(&arena, interp, 0.0, &y), [expected], "at {y:?}");
        }
    }

    #[test]
    fn test_eval_interpolant_nd_cell_selection() {
        // Right-continuous like scipy, clamped at both edges.
        let mut arena = Arena::new();
        let x0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let x1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let interp = arena.alloc(Node::InterpolantNd {
            data: Box::new(NdInterpolantData {
                breakpoints: vec![vec![0.0, 1.0, 3.0], vec![0.0, 1.0]],
                coeffs: vec![5.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0],
                order: 2,
            }),
            children: vec![x0, x1],
        });
        for (x, expected) in [(0.5, 5.0), (2.0, 7.0), (1.0, 7.0), (0.0, 5.0), (3.0, 7.0)] {
            assert_eq!(
                eval_at(&arena, interp, 0.0, &[x, 0.5]),
                [expected],
                "x0={x}"
            );
        }
    }

    #[test]
    fn test_eval_interpolant_nd_tricubic() {
        // p = 3 + dx0^3 + 2*dx1^2 + 5*dx2, coeffs indexed `(a0*4 + a1)*4 + a2`.
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
        assert_eq!(eval_at(&arena, interp, 0.0, &[1.0, 1.0, 1.0]), [11.0]);
        assert_eq!(eval_at(&arena, interp, 0.0, &[3.0, 0.0, 0.0]), [30.0]);
    }

    #[test]
    fn test_eval_interpolant_nd_vector_children_broadcast() {
        // p = dx0 + 10*dx1; the length-1 axis-1 child broadcasts.
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
        assert_eq!(
            eval_at(&arena, interp, 0.0, &[1.0, 2.0, 3.0, 0.5]),
            [6.0, 7.0, 8.0]
        );
    }
}
