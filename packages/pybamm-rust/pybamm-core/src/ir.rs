//! Lowering from the expression DAG to a flat, executable instruction tape.
//!
//! [`TypedIr`] is what actually runs: one [`Instruction`] per node in topological
//! order, operands and results addressed as [`Slot`]s into a single scratch
//! buffer, and arrays, CSR matrices and interpolant tables held in a
//! [`ConstPool`] and addressed by index. Element counts are resolved here, not at
//! evaluation.
//!
//! The slot layout is the caller's one choice. [`from_arena`](TypedIr::from_arena)
//! reuses a slot once its value is dead; `from_arena_split_eval` separates primal
//! from tangent work so one primal pass serves every color; `from_arena_pinned`
//! keeps every intermediate, leaving the buffer standing as reverse-AD's value
//! tape.
//!
//! All three privatise conditional branch cones and emit each branch as a
//! contiguous block, so evaluation skips the branches it did not select.

// Intentional u32 usage for compact instruction storage - expression graphs
// won't exceed 4B nodes in practice
#![allow(clippy::cast_possible_truncation)]

use crate::arena::{Arena, NodeId};
use crate::branch_regions::{
    RegionGroup, privatise_conditionals, schedule_regions, schedule_regions_partitioned,
};
use crate::node::{CsrData, Node};

/// Metrics for buffer slot reuse quality.
#[derive(Clone, Debug)]
pub struct SlotStats {
    /// Elements the lowered tape actually needs.
    pub buffer_size: usize,
    /// Elements a slot-per-node layout would need, the no-reuse baseline.
    pub naive_size: usize,
    /// Instructions emitted.
    pub num_instructions: usize,
    /// `buffer_size / naive_size`; 1.0 means no slot was ever reused.
    pub reuse_ratio: f64,
}

/// Buffer slot within the evaluation arena: `(offset, len)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub offset: u32,
    pub len: u32,
}

impl Slot {
    /// A slot of `len` elements starting at `offset` in the evaluation buffer.
    #[inline]
    pub const fn new(offset: u32, len: u32) -> Self {
        Self { offset, len }
    }

    /// Start offset, widened for indexing.
    #[inline]
    pub const fn offset_usize(self) -> usize {
        self.offset as usize
    }

    /// Element count, widened for indexing.
    #[inline]
    pub const fn len_usize(self) -> usize {
        self.len as usize
    }

    /// Whether this slot holds a single element, and so broadcasts against any
    /// other operand.
    #[inline]
    pub const fn is_scalar(self) -> bool {
        self.len == 1
    }
}

/// Compile-time broadcast pattern for binary operations.
///
/// Resolving which side is scalar once at lowering lets evaluation pick a
/// specialised loop per variant instead of testing operand widths per element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BroadcastKind {
    /// Both operands are single elements.
    ScalarScalar = 0,
    /// Left operand is a single element, broadcast across the right.
    ScalarVector = 1,
    /// Right operand is a single element, broadcast across the left.
    VectorScalar = 2,
    /// Operands are equal-length vectors, paired element-wise.
    VectorVector = 3,
}

impl BroadcastKind {
    /// Classify a pair of operand widths.
    ///
    /// # Panics
    ///
    /// Panics if the widths neither match nor have a scalar side. [`first_invalid`]
    /// rejects that combination before lowering, so a panic here means the DAG
    /// reached the builder unvalidated.
    #[inline]
    pub fn from_lens(a_len: usize, b_len: usize) -> Self {
        assert!(
            broadcast_widths_compatible(a_len, b_len),
            "binary operand widths are incompatible: left={a_len}, right={b_len}"
        );
        match (a_len, b_len) {
            (1, 1) => Self::ScalarScalar,
            (1, _) => Self::ScalarVector,
            (_, 1) => Self::VectorScalar,
            _ => Self::VectorVector,
        }
    }
}

#[inline]
const fn broadcast_widths_compatible(a_len: usize, b_len: usize) -> bool {
    a_len == b_len || (a_len == 1 && b_len > 1) || (b_len == 1 && a_len > 1)
}

/// Element-wise binary operation.
///
/// Every variant applies one `f64` operation across the broadcast operands, with
/// no domain guards: division by zero yields an infinity and out-of-domain powers
/// a NaN, exactly as the hardware would. The comparison variants return 1.0 or
/// 0.0 so they can be multiplied into expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    /// `f64::powf`. `simplify` rewrites the exponents in `{2, 3, 4, -1, -2}` to
    /// multiply/divide chains and folds away `1` and (scalar-base) `0`, so none of
    /// those reach here; every other exponent, integer or not, does.
    Pow,
    /// `f64::min`, which returns the non-NaN operand if exactly one is NaN.
    Minimum,
    /// `f64::max`, which returns the non-NaN operand if exactly one is NaN.
    Maximum,
    /// Truncated remainder (`%`), so the sign follows the dividend rather than
    /// the divisor as Python's `%` does.
    Modulo,
    Hypot,
    /// `a <= b`, the branch-inclusive Heaviside: 1.0 when the operands are equal.
    EqualHeaviside,
    /// `a < b`, the strict Heaviside: 0.0 when the operands are equal.
    NotEqualHeaviside,
    /// `|a - b| < 1e-14`, a tolerance test rather than an exact comparison, so
    /// values that differ only by rounding still compare equal.
    Equality,
}

/// Element-wise unary operation.
///
/// Each variant is the `f64` function of the same name applied per element, again
/// without domain guards: `sqrt(-1)` is NaN and `log(0)` is `-inf` rather than an
/// error. Guarding is the model's job, through a smoothed operator in the
/// expression itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum UnaryOp {
    Neg,
    Abs,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sinh,
    Cosh,
    Arcsinh,
    Arctan,
    /// Abramowitz & Stegun 7.1.26 approximation (error below 1.5e-7), not a libm
    /// call, and shared with constant folding so a folded value matches a runtime
    /// one bit for bit.
    Erf,
    /// `sign(0) = 0`, unlike `f64::signum`.
    Sign,
    Floor,
    Ceiling,
}

/// Single evaluation step.
///
/// The field names are a fixed vocabulary across variants: `dst`, `src`, `a` and
/// `b` are element offsets into the evaluation buffer (not [`Slot`]s, since the
/// length is already fixed at lowering), `len` counts `f64` elements, and any `*_idx`
/// addresses the [`ConstPool`] rather than the buffer. Variants are `Copy` and
/// fixed-size, so a tape is one flat `Vec` an interpreter can walk without
/// chasing pointers.
///
/// Write windows never overlap their read windows; the slot allocator guarantees
/// it, and evaluation relies on it to split the buffer into disjoint slices.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub enum Instruction {
    /// Load a scalar constant into `dst`.
    LoadScalar { value: f64, dst: u32 },
    /// Load the current time value into `dst`.
    LoadTime { dst: u32 },
    /// Load array data from `ConstPool` into `dst`.
    LoadArray { data_idx: u32, len: u32, dst: u32 },
    /// Fill `dst` with zeros (length `len`).
    FillZero { dst: u32, len: u32 },
    /// Load state vector slice `[start, end)` into `dst`.
    LoadStateVector { start: u32, end: u32, dst: u32 },
    /// Load state derivative slice `[start, end)` into `dst`.
    LoadStateVectorDot { start: u32, end: u32, dst: u32 },
    /// Load `width` consecutive packed input values starting at `offset` into `dst`.
    LoadInputParameter { offset: u32, width: u32, dst: u32 },
    /// Load tangent state vector slice `[start, end)` into `dst`.
    LoadTangentState { start: u32, end: u32, dst: u32 },
    /// Load the parameter seed `dp[index]` into `dst`.
    LoadTangentParameter { index: u32, dst: u32 },
    /// Element-wise binary operation: `dst = a <op> b` with broadcast.
    Binary {
        op: BinaryOp,
        a: u32,
        b: u32,
        dst: u32,
        len: u32,
        kind: BroadcastKind,
    },
    /// Element-wise unary operation: `dst = op(src)`.
    Unary {
        op: UnaryOp,
        src: u32,
        dst: u32,
        len: u32,
    },
    /// Reduce to a scalar: `dst = max(src[..src_len])`.
    MaxReduce { src: u32, src_len: u32, dst: u32 },
    /// Reduce to a scalar: `dst = min(src[..src_len])`.
    MinReduce { src: u32, src_len: u32, dst: u32 },
    /// Reduce subgradient: `dst = basis[k]`, `k` = argmax/argmin of
    /// `picker[..len]` (first occurrence, strict comparison).
    ReduceArgSelect {
        basis_src: u32,
        picker_src: u32,
        len: u32,
        is_max: bool,
        dst: u32,
    },
    /// Slice: `dst = src[start..start+len]`.
    Index {
        src: u32,
        start: u32,
        dst: u32,
        len: u32,
    },
    /// Concatenate `sources_len` slot ranges into `dst`.
    Concat {
        sources_idx: u32,
        sources_len: u32,
        dst: u32,
    },
    /// Sparse matrix-vector multiply: `dst = CSR(csr_idx) @ vec(vec_src)`.
    MatMul {
        csr_idx: u32,
        vec_src: u32,
        dst: u32,
    },
    /// Dense matrix-vector multiply: `dst = A @ v`, A row-major at slot `mat_src` (rows × cols).
    DenseMatMul {
        mat_src: u32,
        rows: u32,
        cols: u32,
        vec_src: u32,
        dst: u32,
    },
    /// 1-D linear interpolation: `dst[i] = interp(src[i])`.
    Interp1DLinear {
        interp_idx: u32,
        src: u32,
        dst: u32,
        len: u32,
    },
    /// 1-D linear interpolation derivative.
    Interp1DLinearDeriv {
        interp_idx: u32,
        src: u32,
        dst: u32,
        len: u32,
    },
    /// 1-D cubic (pchip) interpolation.
    Interp1DCubic {
        interp_idx: u32,
        src: u32,
        dst: u32,
        len: u32,
    },
    /// 1-D cubic (pchip) interpolation derivative.
    Interp1DCubicDeriv {
        interp_idx: u32,
        src: u32,
        dst: u32,
        len: u32,
    },
    /// N-D tensor-product interpolation.
    InterpNd {
        interp_idx: u32,
        sources_idx: u32,
        dst: u32,
        len: u32,
    },
    /// N-D tensor-product interpolation partial derivative along `axis`.
    InterpNdPartial {
        interp_idx: u32,
        sources_idx: u32,
        axis: u32,
        dst: u32,
        len: u32,
    },
    /// Piecewise conditional: select branches by `selector` value.
    Conditional {
        selector: u32,
        branches_idx: u32,
        branches_len: u32,
        dst: u32,
        out_len: u32,
    },
    /// Skip all but the active branch's instruction block.
    ///
    /// Sits immediately before the blocks it guards. `blocks_idx` indexes
    /// [`ConstPool::branch_blocks`], holding `(rel_start, len)` per branch where
    /// `rel_start` is an offset **from this instruction's own index**, the tape
    /// is executed as sub-slices (split-eval), so absolute indices would be
    /// wrong. A `len` of 0 means that branch owns no instructions here.
    Dispatch {
        selector: u32,
        blocks_idx: u32,
        blocks_len: u32,
    },
}

/// Interned table for a 1-D linear interpolant.
///
/// Evaluation binary-searches `x_data`, so the knots must be strictly increasing;
/// outside the data range the boundary segment is extended linearly rather than
/// clamped flat.
#[derive(Debug, Clone)]
pub struct InterpolantEntry {
    /// Knot vector, strictly increasing.
    pub x_data: Vec<f64>,
    /// Function values at knots.
    pub y_data: Vec<f64>,
}

/// Interned table for a 1-D cubic interpolant, pre-divided into per-interval
/// power-basis coefficients so evaluation is a search plus a Horner step.
#[derive(Debug, Clone)]
pub struct CubicInterpolantEntry {
    /// Interval breakpoints, length `nseg + 1`.
    pub breakpoints: Vec<f64>,
    /// Per-interval power-basis coefficients `[c0, c1, c2, c3]`, length `nseg`.
    /// `p(dx) = c0 + c1·dx + c2·dx² + c3·dx³`, `dx = x - breakpoints[i]`.
    pub coeffs: Vec<[f64; 4]>,
}

/// Interned table for a 2-D or 3-D tensor-product interpolant.
///
/// One flat coefficient block per cell, so a lookup is one cell index per axis
/// followed by a tensor Horner evaluation; `order` says whether the cells are
/// multilinear or cubic in every axis, since the two are not mixed.
#[derive(Debug, Clone)]
pub struct NdInterpolantEntry {
    /// Per-axis knot vectors (2 or 3 axes), each of length `nseg_a + 1`.
    pub breakpoints: Vec<Vec<f64>>,
    /// Flat per-cell power-basis tensors: cell-major, `order^ndim` coeffs per cell.
    pub coeffs: Vec<f64>,
    /// Per-axis polynomial order: 2 = multilinear, 4 = tensor cubic.
    pub order: u32,
}

/// Side tables an instruction addresses by index instead of carrying inline.
///
/// Instructions are fixed-size and `Copy`, so anything variable-length lives here
/// and is referenced by a `u32`: dense arrays, CSR matrices, interpolant tables,
/// the operand lists of `Concat` and the branch ranges of conditionals. Adding
/// an entry deduplicates nothing, so a tape may hold the same array twice; CSE on
/// the DAG is what keeps that rare.
#[derive(Debug, Default, Clone)]
pub struct ConstPool {
    /// Flattened array data; `array_offsets[i]` gives the start of slot `i`.
    pub array_data: Vec<f64>,
    /// Start offset of each array in `array_data`.
    pub array_offsets: Vec<usize>,

    /// CSR sparse matrices for matmul instructions.
    pub csr_data: Vec<CsrData>,

    /// 1-D linear interpolation tables.
    pub interpolants: Vec<InterpolantEntry>,

    /// 1-D cubic (pchip) interpolation tables.
    pub cubic_interpolants: Vec<CubicInterpolantEntry>,

    /// N-D tensor-product interpolation tables.
    pub nd_interpolants: Vec<NdInterpolantEntry>,

    /// Concat source slot ranges: `(offset, len)` pairs.
    pub concat_sources: Vec<(u32, u32)>,

    /// Conditional branch slot ranges: `(offset, len)` pairs.
    pub branch_offsets: Vec<(u32, u32)>,

    /// `Dispatch` branch block ranges: `(rel_start, len)` instruction offsets
    /// relative to the owning `Dispatch`.
    pub branch_blocks: Vec<(u32, u32)>,

    /// N-D interpolant child slot ranges: `(offset, len)` per input.
    pub interp_nd_sources: Vec<(u32, u32)>,
}

impl ConstPool {
    /// An empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert array data, returning its index.
    pub fn add_array(&mut self, data: &[f64]) -> u32 {
        let idx = self.array_offsets.len() as u32;
        self.array_offsets.push(self.array_data.len());
        self.array_data.extend_from_slice(data);
        idx
    }

    /// Retrieve a slice of array data by index and length.
    pub fn get_array(&self, idx: u32, len: u32) -> &[f64] {
        let start = self.array_offsets[idx as usize];
        &self.array_data[start..start + len as usize]
    }

    /// Insert a CSR matrix, returning its index.
    pub fn add_csr(&mut self, csr: CsrData) -> u32 {
        let idx = self.csr_data.len() as u32;
        self.csr_data.push(csr);
        idx
    }

    /// Insert a 1-D linear interpolation table, returning its index.
    pub fn add_interpolant(&mut self, x_data: Vec<f64>, y_data: Vec<f64>) -> u32 {
        let idx = self.interpolants.len() as u32;
        self.interpolants.push(InterpolantEntry { x_data, y_data });
        idx
    }

    /// Insert a 1-D cubic interpolation table, returning its index.
    pub fn add_cubic_interpolant(&mut self, breakpoints: Vec<f64>, coeffs: Vec<[f64; 4]>) -> u32 {
        let idx = self.cubic_interpolants.len() as u32;
        self.cubic_interpolants.push(CubicInterpolantEntry {
            breakpoints,
            coeffs,
        });
        idx
    }

    /// Insert an N-D interpolation table, returning its index.
    pub fn add_nd_interpolant(
        &mut self,
        breakpoints: Vec<Vec<f64>>,
        coeffs: Vec<f64>,
        order: u32,
    ) -> u32 {
        let idx = self.nd_interpolants.len() as u32;
        self.nd_interpolants.push(NdInterpolantEntry {
            breakpoints,
            coeffs,
            order,
        });
        idx
    }
}

/// Metadata for a partitioned primal/tangent instruction stream.
///
/// When present, the instruction stream is ordered so all primal instructions
/// precede all tangent instructions, and buffer slots are partitioned into
/// two non-overlapping pools. This allows evaluating the primal section once
/// and re-running only the tangent section with different seed vectors.
#[derive(Debug, Clone, Copy)]
pub struct SplitEvalInfo {
    /// Index into the instruction stream where tangent instructions begin.
    pub primal_end: usize,
    /// Buffer region `[0, primal_buffer_size)` holds primal values.
    pub primal_buffer_size: usize,
}

/// Typed Intermediate Representation
///
/// A compiled representation of an expression DAG with explicit shapes
/// for all intermediate values. This IR can be:
/// - Interpreted directly by `CompiledExpr`
/// - Transformed by symbolic differentiation
#[derive(Debug, Clone)]
pub struct TypedIr {
    /// Linear sequence of instructions in topological order
    instructions: Vec<Instruction>,

    /// Root output slot
    root_slot: Slot,

    /// Total buffer size needed for evaluation
    buffer_size: usize,

    /// Constant pool for large data
    consts: ConstPool,

    /// Metadata for differentiation
    n_states: usize,
    n_params: usize,
    uses_state_dot: bool,

    /// Partitioning for split evaluation, `None` for standard IR.
    split_eval_info: Option<SplitEvalInfo>,
}

impl TypedIr {
    /// Compile an expression DAG into the IR.
    ///
    /// Runs [`privatise_conditionals`] first so branch cones `cse` made shared
    /// become exclusive and can be short-circuited. Every `from_arena*` entry
    /// point does this, which is why no caller has to.
    pub fn from_arena(arena: &Arena, root: NodeId) -> Self {
        Self::privatise_then(arena, root, IRBuilder::build)
    }

    /// [`from_arena`](Self::from_arena) without branch privatisation. Test-only
    /// escape hatch for comparing against an unprivatised tape.
    #[cfg(test)]
    pub fn from_arena_raw(arena: &Arena, root: NodeId) -> Self {
        IRBuilder::build(arena, root)
    }

    /// Run [`privatise_conditionals`] on `arena`/`root`, then hand the (possibly
    /// rewritten) arena to `build`. Shared by every `from_arena*` constructor so
    /// privatisation always happens exactly once, right before lowering.
    fn privatise_then(
        arena: &Arena,
        root: NodeId,
        build: impl FnOnce(&Arena, NodeId) -> Self,
    ) -> Self {
        match privatise_conditionals(arena, root) {
            Some((owned, owned_root)) => build(&owned, owned_root),
            None => build(arena, root),
        }
    }

    /// Output length of the root expression.
    #[inline]
    pub const fn output_len(&self) -> usize {
        self.root_slot.len as usize
    }

    /// Instruction stream in evaluation order.
    #[inline]
    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    /// Per-branch block lengths in tape order, one entry per branch of every
    /// `Dispatch`. Empty when no conditional was short-circuited.
    #[must_use]
    pub fn branch_block_lens(&self) -> Vec<u32> {
        let mut lens = Vec::new();
        for instr in &self.instructions {
            if let Instruction::Dispatch {
                blocks_idx,
                blocks_len,
                ..
            } = *instr
            {
                for b in 0..blocks_len as usize {
                    lens.push(self.consts.branch_blocks[blocks_idx as usize + b].1);
                }
            }
        }
        lens
    }

    /// How many `Dispatch` instructions the tape carries, one per
    /// short-circuited conditional, in each half of a split-eval tape.
    ///
    /// The part of [`common_instruction_count`](Self::common_instruction_count)
    /// that exists only because a conditional was short-circuited, which
    /// `branch_block_lens` cannot recover: it is flattened across dispatches.
    #[must_use]
    pub fn dispatch_count(&self) -> usize {
        self.instructions
            .iter()
            .filter(|instr| matches!(instr, Instruction::Dispatch { .. }))
            .count()
    }

    /// Instruction count excluding conditional branch blocks: the common tape
    /// plus one `Dispatch` per short-circuited conditional.
    ///
    /// This is the quantity `casadi.Function.n_instructions()` reports for a
    /// Switch-lowered node, so the two backends are comparable. Use
    /// <code>[instructions](Self::instructions).len()</code> for the raw tape
    /// length.
    #[must_use]
    pub fn common_instruction_count(&self) -> usize {
        self.instructions.len() - self.branch_block_lens().iter().sum::<u32>() as usize
    }

    /// Output slot of the root expression.
    #[inline]
    pub const fn root_slot(&self) -> Slot {
        self.root_slot
    }

    /// Total evaluation buffer size in elements.
    #[inline]
    pub const fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Reference to the constant pool.
    #[inline]
    pub const fn consts(&self) -> &ConstPool {
        &self.consts
    }

    /// Number of state variables referenced.
    #[inline]
    pub const fn n_states(&self) -> usize {
        self.n_states
    }

    /// Number of input parameters referenced.
    #[inline]
    pub const fn n_params(&self) -> usize {
        self.n_params
    }

    /// Whether the expression references state derivatives.
    #[inline]
    pub const fn uses_state_dot(&self) -> bool {
        self.uses_state_dot
    }

    /// Partitioning for split evaluation, `None` for standard IR.
    #[inline]
    pub const fn split_eval_info(&self) -> Option<SplitEvalInfo> {
        self.split_eval_info
    }

    /// Compute buffer slot allocation quality metrics.
    pub fn slot_stats(arena: &Arena, root: NodeId) -> SlotStats {
        let eval_order = arena.topological_order(root);
        let sizes = infer_sizes(arena, &eval_order);
        let naive_size: usize = eval_order.iter().map(|n| sizes[n.index()]).sum();
        let ir = Self::from_arena(arena, root);
        let reuse_ratio = if naive_size == 0 {
            1.0
        } else {
            ir.buffer_size as f64 / naive_size as f64
        };
        SlotStats {
            buffer_size: ir.buffer_size,
            naive_size,
            num_instructions: ir.instructions.len(),
            reuse_ratio,
        }
    }

    /// Compile with primal/tangent split for efficient Jacobian assembly.
    ///
    /// The instruction stream is partitioned so primal instructions precede
    /// tangent instructions, with buffer slots in two disjoint pools.
    /// Primal work is evaluated once; only the tangent section is re-run
    /// per color during coloring-based Jacobian assembly.
    pub fn from_arena_split_eval(arena: &Arena, root: NodeId) -> Self {
        Self::privatise_then(arena, root, IRBuilder::build_split_eval)
    }

    /// Compile with a no-reuse (SSA) slot layout for reverse-mode AD.
    /// Every intermediate keeps a unique slot, so after one primal eval the
    /// scratch buffer is the reverse value tape.
    pub fn from_arena_pinned(arena: &Arena, root: NodeId) -> Self {
        Self::privatise_then(arena, root, IRBuilder::build_pinned)
    }
}

/// Builder for constructing `TypedIr` from an expression DAG.
struct IRBuilder {
    /// Emitted instruction stream in evaluation order.
    instructions: Vec<Instruction>,
    consts: ConstPool,
    /// `(offset, len)` for each `NodeId` in the arena.
    slots: Vec<(usize, usize)>,
    /// Maximum state index referenced.
    n_states: usize,
    /// Number of input parameters referenced.
    n_params: usize,
    /// Whether the expression uses state derivatives.
    uses_state_dot: bool,
}

impl IRBuilder {
    fn build(arena: &Arena, root: NodeId) -> TypedIr {
        let base_order = arena.topological_order(root);
        let schedule = schedule_regions(arena, &base_order);
        let sizes = infer_sizes(arena, &schedule.order);
        let (slots, total_size) = assign_slots(arena, &sizes, &schedule.order, &[root]);

        let mut builder = Self {
            instructions: Vec::with_capacity(schedule.order.len()),
            consts: ConstPool::new(),
            slots,
            n_states: 0,
            n_params: 0,
            uses_state_dot: false,
        };
        builder.emit_scheduled(arena, &schedule.order, &schedule.groups);
        builder.assert_block_slots_private(total_size);

        let (root_offset, root_len) = builder.slots[root.index()];

        TypedIr {
            instructions: builder.instructions,
            root_slot: Slot::new(root_offset as u32, root_len as u32),
            buffer_size: total_size,
            consts: builder.consts,
            n_states: builder.n_states,
            n_params: builder.n_params,
            uses_state_dot: builder.uses_state_dot,
            split_eval_info: None,
        }
    }

    /// Build a primal `TypedIr` with a no-reuse (SSA) slot layout. See
    /// [`assign_slots_pinned`]. Mirrors [`Self::build`] but never recycles
    /// a slot, so the reverse-AD backward pass can read every intermediate.
    fn build_pinned(arena: &Arena, root: NodeId) -> TypedIr {
        let base_order = arena.topological_order(root);
        let schedule = schedule_regions(arena, &base_order);
        let sizes = infer_sizes(arena, &schedule.order);
        let (slots, total_size) = assign_slots_pinned(&sizes, &schedule.order);

        let mut builder = Self {
            instructions: Vec::with_capacity(schedule.order.len()),
            consts: ConstPool::new(),
            slots,
            n_states: 0,
            n_params: 0,
            uses_state_dot: false,
        };
        builder.emit_scheduled(arena, &schedule.order, &schedule.groups);
        builder.assert_block_slots_private(total_size);

        let (root_offset, root_len) = builder.slots[root.index()];

        TypedIr {
            instructions: builder.instructions,
            root_slot: Slot::new(root_offset as u32, root_len as u32),
            buffer_size: total_size,
            consts: builder.consts,
            n_states: builder.n_states,
            n_params: builder.n_params,
            uses_state_dot: builder.uses_state_dot,
            split_eval_info: None,
        }
    }

    fn build_split_eval(arena: &Arena, root: NodeId) -> TypedIr {
        let eval_order = arena.topological_order(root);
        let sizes = infer_sizes(arena, &eval_order);
        let is_tangent = classify_tangent_nodes(arena, &eval_order);

        // Stable-partition: primal nodes first, tangent nodes second
        let mut primal_order = Vec::new();
        let mut tangent_order = Vec::new();
        for &nid in &eval_order {
            if is_tangent[nid.index()] {
                tangent_order.push(nid);
            } else {
                primal_order.push(nid);
            }
        }
        // Each half schedules on its own, so a `Dispatch` and the blocks it
        // guards land in one half; a cone spanning the split yields one each.
        let primal_schedule = schedule_regions_partitioned(arena, &primal_order, &eval_order);
        let tangent_schedule = schedule_regions_partitioned(arena, &tangent_order, &eval_order);
        let partitioned_order: Vec<NodeId> = primal_schedule
            .order
            .iter()
            .chain(tangent_schedule.order.iter())
            .copied()
            .collect();

        let n = sizes.len();
        let mut last_use = vec![0_usize; n];
        for (pos, &nid) in partitioned_order.iter().enumerate() {
            arena.get(nid).for_each_child(|c| {
                if pos > last_use[c.index()] {
                    last_use[c.index()] = pos;
                }
            });
        }
        // Pin root
        last_use[root.index()] = usize::MAX;

        // Pass 1: allocate primal slots in pool [0, primal_hw)
        let (primal_slots, primal_hw) =
            assign_slots_configured(&sizes, &primal_schedule.order, &last_use, 0);

        // Pass 2: allocate tangent slots in pool [primal_hw, total_hw)
        let (tangent_slots, total_hw) =
            assign_slots_configured(&sizes, &tangent_schedule.order, &last_use, primal_hw);

        // Merge slot maps
        let mut slots = vec![(0_usize, 0_usize); n];
        for &nid in &primal_schedule.order {
            slots[nid.index()] = primal_slots[nid.index()];
        }
        for &nid in &tangent_schedule.order {
            slots[nid.index()] = tangent_slots[nid.index()];
        }

        let mut builder = Self {
            instructions: Vec::with_capacity(partitioned_order.len()),
            consts: ConstPool::new(),
            slots,
            n_states: 0,
            n_params: 0,
            uses_state_dot: false,
        };

        builder.emit_scheduled(arena, &primal_schedule.order, &primal_schedule.groups);
        let primal_end = builder.instructions.len();
        builder.emit_scheduled(arena, &tangent_schedule.order, &tangent_schedule.groups);
        // After both halves, so the tangent half's blocks are checked too.
        builder.assert_no_block_straddles(primal_end);
        builder.assert_block_slots_private(total_hw);

        let (root_offset, root_len) = builder.slots[root.index()];

        TypedIr {
            instructions: builder.instructions,
            root_slot: Slot::new(root_offset as u32, root_len as u32),
            buffer_size: total_hw,
            consts: builder.consts,
            n_states: builder.n_states,
            n_params: builder.n_params,
            uses_state_dot: builder.uses_state_dot,
            split_eval_info: Some(SplitEvalInfo {
                primal_end,
                primal_buffer_size: primal_hw,
            }),
        }
    }

    fn slot_for(&self, id: NodeId) -> (u32, u32) {
        let (off, len) = self.slots[id.index()];
        (off as u32, len as u32)
    }

    /// Emit `order`, wrapping each annotated group range in a `Dispatch`.
    ///
    /// Group anchors are distinct (an empty group is never recorded) and their
    /// ranges are disjoint, so one lookup slot per position suffices.
    /// Emit `order`, wrapping each annotated group range in a `Dispatch`.
    ///
    /// Group anchors are distinct (an empty group is never recorded) and their
    /// ranges are disjoint, so one lookup slot per position suffices.
    fn emit_scheduled(&mut self, arena: &Arena, order: &[NodeId], groups: &[RegionGroup]) {
        let mut group_at: Vec<Option<&RegionGroup>> = vec![None; order.len() + 1];
        for g in groups {
            assert!(
                group_at[g.anchor].is_none(),
                "two region groups share anchor {}",
                g.anchor
            );
            group_at[g.anchor] = Some(g);
        }

        let mut pos = 0;
        while pos < order.len() {
            if let Some(group) = group_at[pos] {
                let total: usize = group.branch_lens.iter().sum();
                assert!(total > 0, "an empty region group must not be recorded");
                self.emit_dispatch(arena, group, &order[pos..pos + total]);
                pos += total;
                continue;
            }
            self.emit_node(arena, order[pos]);
            pos += 1;
        }
    }

    /// Emit a `Dispatch` followed by one contiguous block per branch, recording
    /// each block's `(rel_start, len)` relative to the `Dispatch`'s own index.
    ///
    /// `nodes` is the group's slice of the emission order, branch runs back to
    /// back. Block lengths are measured in *instructions*, not nodes, because
    /// `Node::SparseMatrix` emits none.
    fn emit_dispatch(&mut self, arena: &Arena, group: &RegionGroup, nodes: &[NodeId]) {
        let Node::Conditional { selector, .. } = arena.get(group.cond) else {
            unreachable!("a region group is always anchored on a Conditional")
        };
        let selector_slot = self.slot_for(*selector).0;
        let dispatch_at = self.instructions.len();
        let blocks_idx = self.consts.branch_blocks.len() as u32;
        // Reserve the table so the blocks can be emitted before it is filled.
        self.consts
            .branch_blocks
            .resize(blocks_idx as usize + group.branch_lens.len(), (0, 0));
        self.instructions.push(Instruction::Dispatch {
            selector: selector_slot,
            blocks_idx,
            blocks_len: group.branch_lens.len() as u32,
        });

        let mut taken = 0;
        for (i, &n_nodes) in group.branch_lens.iter().enumerate() {
            let start = self.instructions.len();
            for &node_id in &nodes[taken..taken + n_nodes] {
                self.emit_node(arena, node_id);
            }
            taken += n_nodes;
            let len = self.instructions.len() - start;
            self.consts.branch_blocks[blocks_idx as usize + i] =
                ((start - dispatch_at) as u32, len as u32);
        }
        assert_eq!(taken, nodes.len(), "group range and branch_lens disagree");
    }

    /// Assert that every `Dispatch` span lies wholly on one side of `split`.
    /// Otherwise `run_tangent_section`, which executes `instructions[split..]`,
    /// would enter a block without its `Dispatch` and run it unconditionally.
    /// Ships alongside [`Self::assert_block_slots_private`], for the same reason:
    /// the failure is silently wrong values, not a crash.
    ///
    /// # Panics
    /// Panics if a block starts before `split` and ends after it.
    fn assert_no_block_straddles(&self, split: usize) {
        for (pc, instr) in self.instructions.iter().enumerate() {
            if let Instruction::Dispatch {
                blocks_idx,
                blocks_len,
                ..
            } = *instr
            {
                for b in 0..blocks_len as usize {
                    let (rel, len) = self.consts.branch_blocks[blocks_idx as usize + b];
                    let start = pc + rel as usize;
                    let end = start + len as usize;
                    assert!(
                        split <= start || split >= end,
                        "split point {split} straddles block [{start}, {end})"
                    );
                }
            }
        }
    }

    /// Which instructions sit inside a branch block.
    ///
    /// # Panics
    /// Panics if a block contains a `Dispatch`. Every evaluator skips one level
    /// of block only, and `reverse.rs`'s span table keys one owner per span end,
    /// so a nested span would be silently mis-walked; `build_span_owner`'s
    /// overlap check cannot see a span fully contained in another.
    fn block_instruction_mask(&self) -> Vec<bool> {
        let mut in_block = vec![false; self.instructions.len()];
        for (pc, instr) in self.instructions.iter().enumerate() {
            if let Instruction::Dispatch {
                blocks_idx,
                blocks_len,
                ..
            } = *instr
            {
                for b in 0..blocks_len as usize {
                    let (rel, len) = self.consts.branch_blocks[blocks_idx as usize + b];
                    for k in 0..len as usize {
                        let at = pc + rel as usize + k;
                        assert!(
                            !matches!(self.instructions[at], Instruction::Dispatch { .. }),
                            "instruction {at} is a Dispatch inside the block of the Dispatch at \
                             {pc}: branch blocks must be flat"
                        );
                        in_block[at] = true;
                    }
                }
            }
        }
        in_block
    }

    /// Assert that no instruction outside a branch block reads a value a block
    /// defined. This is the one miscompilation this plan produces silently, where
    /// the reader sees whatever a previous solve left in the recycled slot. Runs
    /// once per compile, in release too.
    ///
    /// Tracks the *last writer* of each buffer element, not set membership:
    /// recycling makes "something outside also writes here" routinely true.
    /// Extents are element-wise, so a partial overlap still trips. A `Dispatch`
    /// selector that is never written has no writer to blame, so its definition
    /// is asserted directly.
    ///
    /// # Panics
    /// Panics if an outside instruction's read resolves to a block-owned
    /// definition, naming the reader, the element and the defining instruction,
    /// or if a `Dispatch`'s selector has no earlier definition.
    fn assert_block_slots_private(&self, buffer_size: usize) {
        // A tape with no blocks cannot violate this, and most tapes have none.
        if self.consts.branch_blocks.is_empty() {
            return;
        }
        let in_block = self.block_instruction_mask();
        let mut last_writer: Vec<Option<usize>> = vec![None; buffer_size];

        for (pc, instr) in self.instructions.iter().enumerate() {
            if !in_block[pc] {
                // The selector decides which block runs, so an undefined one is
                // a miscompilation the last-writer scan below cannot name.
                if let Instruction::Dispatch { selector, .. } = *instr {
                    assert!(
                        last_writer[selector as usize].is_some(),
                        "Dispatch at {pc} selects on buffer element {selector}, which no earlier \
                         instruction defines"
                    );
                }
                for src in instruction_src_extents(instr, &self.consts) {
                    // A `Conditional`'s branch slots are the one legitimate
                    // outside read of a block-owned value.
                    if src.is_branch_slot {
                        continue;
                    }
                    let start = src.offset as usize;
                    let end = start + src.len as usize;
                    for (index, writer) in last_writer[start..end].iter().enumerate() {
                        if let Some(writer) = *writer
                            && in_block[writer]
                        {
                            panic!(
                                "instruction {pc} ({instr:?}) outside a branch block reads buffer \
                                 element {}, defined by block-private instruction {writer} ({:?})",
                                start + index,
                                self.instructions[writer]
                            );
                        }
                    }
                }
            }
            if let Some((offset, len)) = instruction_dst_extent(instr, &self.consts) {
                let start = offset as usize;
                for writer in &mut last_writer[start..start + len as usize] {
                    *writer = Some(pc);
                }
            }
        }
    }

    fn emit_node(&mut self, arena: &Arena, node_id: NodeId) {
        let (dst, out_len) = self.slots[node_id.index()];
        let dst = dst as u32;
        let out_len_u32 = out_len as u32;

        let instr = match &arena[node_id] {
            Node::Scalar(v) => Instruction::LoadScalar { value: *v, dst },
            Node::Time => Instruction::LoadTime { dst },
            Node::Array(arr) => {
                let data_idx = self.consts.add_array(&arr.data);
                Instruction::LoadArray {
                    data_idx,
                    len: arr.data.len() as u32,
                    dst,
                }
            },
            Node::ZeroVector { len } => Instruction::FillZero {
                dst,
                len: *len as u32,
            },
            Node::StateVector { start, end } => {
                self.n_states = self.n_states.max(*end);
                Instruction::LoadStateVector {
                    start: *start as u32,
                    end: *end as u32,
                    dst,
                }
            },
            Node::StateVectorDot { start, end } => {
                self.uses_state_dot = true;
                self.n_states = self.n_states.max(*end);
                Instruction::LoadStateVectorDot {
                    start: *start as u32,
                    end: *end as u32,
                    dst,
                }
            },
            Node::InputParameter {
                index,
                offset,
                width,
                ..
            } => {
                self.n_params = self.n_params.max(*index + 1);
                Instruction::LoadInputParameter {
                    offset: *offset as u32,
                    width: *width as u32,
                    dst,
                }
            },
            Node::TangentStateVector { start, end } => {
                self.n_states = self.n_states.max(*end);
                Instruction::LoadTangentState {
                    start: *start as u32,
                    end: *end as u32,
                    dst,
                }
            },
            Node::TangentParameter { index } => {
                self.n_params = self.n_params.max(*index + 1);
                Instruction::LoadTangentParameter {
                    index: *index as u32,
                    dst,
                }
            },
            Node::SparseMatrix(_) => return, // Not emitted as instruction

            // Binary operations
            Node::Add(a, b) => self.emit_binary(BinaryOp::Add, *a, *b, dst, out_len_u32),
            Node::Sub(a, b) => self.emit_binary(BinaryOp::Sub, *a, *b, dst, out_len_u32),
            Node::Mul(a, b) => self.emit_binary(BinaryOp::Mul, *a, *b, dst, out_len_u32),
            Node::Div(a, b) => self.emit_binary(BinaryOp::Div, *a, *b, dst, out_len_u32),
            Node::Pow(a, b) => self.emit_binary(BinaryOp::Pow, *a, *b, dst, out_len_u32),
            Node::Minimum(a, b) => self.emit_binary(BinaryOp::Minimum, *a, *b, dst, out_len_u32),
            Node::Maximum(a, b) => self.emit_binary(BinaryOp::Maximum, *a, *b, dst, out_len_u32),
            Node::Modulo(a, b) => self.emit_binary(BinaryOp::Modulo, *a, *b, dst, out_len_u32),
            Node::Hypot(a, b) => self.emit_binary(BinaryOp::Hypot, *a, *b, dst, out_len_u32),
            Node::EqualHeaviside(a, b) => {
                self.emit_binary(BinaryOp::EqualHeaviside, *a, *b, dst, out_len_u32)
            },
            Node::NotEqualHeaviside(a, b) => {
                self.emit_binary(BinaryOp::NotEqualHeaviside, *a, *b, dst, out_len_u32)
            },
            Node::Equality(a, b) => self.emit_binary(BinaryOp::Equality, *a, *b, dst, out_len_u32),

            // Unary operations
            Node::Neg(a) => self.emit_unary(UnaryOp::Neg, *a, dst, out_len_u32),
            Node::Abs(a) => self.emit_unary(UnaryOp::Abs, *a, dst, out_len_u32),
            Node::Sqrt(a) => self.emit_unary(UnaryOp::Sqrt, *a, dst, out_len_u32),
            Node::Exp(a) => self.emit_unary(UnaryOp::Exp, *a, dst, out_len_u32),
            Node::Log(a) => self.emit_unary(UnaryOp::Log, *a, dst, out_len_u32),
            Node::Sin(a) => self.emit_unary(UnaryOp::Sin, *a, dst, out_len_u32),
            Node::Cos(a) => self.emit_unary(UnaryOp::Cos, *a, dst, out_len_u32),
            Node::Tanh(a) => self.emit_unary(UnaryOp::Tanh, *a, dst, out_len_u32),
            Node::Sinh(a) => self.emit_unary(UnaryOp::Sinh, *a, dst, out_len_u32),
            Node::Cosh(a) => self.emit_unary(UnaryOp::Cosh, *a, dst, out_len_u32),
            Node::Arcsinh(a) => self.emit_unary(UnaryOp::Arcsinh, *a, dst, out_len_u32),
            Node::Arctan(a) => self.emit_unary(UnaryOp::Arctan, *a, dst, out_len_u32),
            Node::Erf(a) => self.emit_unary(UnaryOp::Erf, *a, dst, out_len_u32),
            Node::Sign(a) => self.emit_unary(UnaryOp::Sign, *a, dst, out_len_u32),
            Node::Floor(a) => self.emit_unary(UnaryOp::Floor, *a, dst, out_len_u32),
            Node::Ceiling(a) => self.emit_unary(UnaryOp::Ceiling, *a, dst, out_len_u32),

            // Reduction ops
            Node::MaxReduce(a) => {
                let (src, src_len) = self.slot_for(*a);
                Instruction::MaxReduce { src, src_len, dst }
            },
            Node::MinReduce(a) => {
                let (src, src_len) = self.slot_for(*a);
                Instruction::MinReduce { src, src_len, dst }
            },
            Node::ReduceArgSelect {
                basis,
                picker,
                is_max,
            } => {
                let (basis_src, len) = self.slot_for(*basis);
                let (picker_src, picker_len) = self.slot_for(*picker);
                // basis (a tangent) and its primal picker always share width; the
                // scan reads picker[0..len], so a mismatch would read the wrong slot.
                debug_assert_eq!(
                    len, picker_len,
                    "ReduceArgSelect basis/picker width mismatch"
                );
                Instruction::ReduceArgSelect {
                    basis_src,
                    picker_src,
                    len,
                    is_max: *is_max,
                    dst,
                }
            },

            // Structural nodes
            Node::Index { child, start, .. } => Instruction::Index {
                src: self.slot_for(*child).0,
                start: *start as u32,
                dst,
                len: out_len_u32,
            },
            Node::Concat(children) => {
                let sources_idx = self.consts.concat_sources.len() as u32;
                for c in children {
                    let (off, len) = self.slot_for(*c);
                    self.consts.concat_sources.push((off, len));
                }
                Instruction::Concat {
                    sources_idx,
                    sources_len: children.len() as u32,
                    dst,
                }
            },

            // Matrix operations
            Node::MatMul(a, b) => match &arena[*a] {
                Node::SparseMatrix(csr) => {
                    let csr_idx = self.consts.add_csr(csr.as_ref().clone());
                    Instruction::MatMul {
                        csr_idx,
                        vec_src: self.slot_for(*b).0,
                        dst,
                    }
                },
                Node::Array(arr) => Instruction::DenseMatMul {
                    mat_src: self.slot_for(*a).0,
                    rows: arr.shape.rows as u32,
                    cols: arr.shape.cols as u32,
                    vec_src: self.slot_for(*b).0,
                    dst,
                },
                _ => panic!("MatMul requires a constant matrix on the left"),
            },

            // Interpolation nodes
            Node::Interpolant1DLinear { data, child } => {
                let interp_idx = self
                    .consts
                    .add_interpolant(data.x_data.clone(), data.y_data.clone());
                Instruction::Interp1DLinear {
                    interp_idx,
                    src: self.slot_for(*child).0,
                    dst,
                    len: out_len_u32,
                }
            },
            Node::Interpolant1DLinearDeriv {
                slopes,
                x_data,
                child,
            } => {
                // Store x_data as breakpoints and slopes as y_data (the derivative values)
                let interp_idx = self
                    .consts
                    .add_interpolant(x_data.to_vec(), slopes.to_vec());
                Instruction::Interp1DLinearDeriv {
                    interp_idx,
                    src: self.slot_for(*child).0,
                    dst,
                    len: out_len_u32,
                }
            },
            Node::Interpolant1DCubic { data, child } => {
                let interp_idx = self
                    .consts
                    .add_cubic_interpolant(data.breakpoints.clone(), data.coeffs.clone());
                Instruction::Interp1DCubic {
                    interp_idx,
                    src: self.slot_for(*child).0,
                    dst,
                    len: out_len_u32,
                }
            },
            Node::Interpolant1DCubicDeriv { data, child } => {
                let interp_idx = self
                    .consts
                    .add_cubic_interpolant(data.breakpoints.clone(), data.coeffs.clone());
                Instruction::Interp1DCubicDeriv {
                    interp_idx,
                    src: self.slot_for(*child).0,
                    dst,
                    len: out_len_u32,
                }
            },
            Node::InterpolantNd { data, children } => {
                let interp_idx = self.consts.add_nd_interpolant(
                    data.breakpoints.clone(),
                    data.coeffs.clone(),
                    data.order,
                );
                let sources_idx = self.consts.interp_nd_sources.len() as u32;
                for c in children {
                    let (off, len) = self.slot_for(*c);
                    self.consts.interp_nd_sources.push((off, len));
                }
                Instruction::InterpNd {
                    interp_idx,
                    sources_idx,
                    dst,
                    len: out_len_u32,
                }
            },
            Node::InterpolantNdPartial {
                data,
                children,
                axis,
            } => {
                let interp_idx = self.consts.add_nd_interpolant(
                    data.breakpoints.clone(),
                    data.coeffs.clone(),
                    data.order,
                );
                let sources_idx = self.consts.interp_nd_sources.len() as u32;
                for c in children {
                    let (off, len) = self.slot_for(*c);
                    self.consts.interp_nd_sources.push((off, len));
                }
                Instruction::InterpNdPartial {
                    interp_idx,
                    sources_idx,
                    axis: *axis,
                    dst,
                    len: out_len_u32,
                }
            },

            Node::Conditional { selector, branches } => {
                let branches_idx = self.consts.branch_offsets.len() as u32;
                for b in branches {
                    let (off, len) = self.slot_for(*b);
                    self.consts.branch_offsets.push((off, len));
                }
                Instruction::Conditional {
                    selector: self.slot_for(*selector).0,
                    branches_idx,
                    branches_len: branches.len() as u32,
                    dst,
                    out_len: out_len_u32,
                }
            },
        };

        self.instructions.push(instr);
    }

    fn emit_binary(&self, op: BinaryOp, a: NodeId, b: NodeId, dst: u32, len: u32) -> Instruction {
        let (a_off, a_len) = self.slot_for(a);
        let (b_off, b_len) = self.slot_for(b);
        Instruction::Binary {
            op,
            a: a_off,
            b: b_off,
            dst,
            len,
            kind: BroadcastKind::from_lens(a_len as usize, b_len as usize),
        }
    }

    fn emit_unary(&self, op: UnaryOp, a: NodeId, dst: u32, len: u32) -> Instruction {
        let (src, _) = self.slot_for(a);
        Instruction::Unary { op, src, dst, len }
    }
}

#[cfg(test)]
impl IRBuilder {
    /// Hand-build a tape that violates `assert_block_slots_private`: an
    /// instruction outside the one `Dispatch` block reads the slot that
    /// block's only instruction wrote. No real scheduler produces this; it
    /// exists to exercise the guard, which otherwise ships untested.
    fn test_tape_reading_across_block_boundary() -> Self {
        let mut consts = ConstPool::new();
        consts.branch_blocks.push((1, 1)); // one block: 1 instruction, starting right after the Dispatch

        let instructions = vec![
            Instruction::LoadScalar { value: 1.0, dst: 0 }, // defines the selector
            Instruction::Dispatch {
                selector: 0,
                blocks_idx: 0,
                blocks_len: 1,
            },
            Instruction::LoadScalar { value: 1.0, dst: 5 }, // block-private write to slot 5
            Instruction::Unary {
                op: UnaryOp::Neg,
                src: 5, // outside the block, illegally reads the block's slot
                dst: 6,
                len: 1,
            },
        ];

        Self {
            instructions,
            consts,
            slots: Vec::new(),
            n_states: 0,
            n_params: 0,
            uses_state_dot: false,
        }
    }

    /// Hand-build a tape whose `Dispatch` selects on a slot no instruction ever
    /// writes. The last-writer scan has no writer to blame here, so the guard
    /// asserts the definition itself; no real scheduler produces this either.
    fn test_tape_dispatching_on_an_undefined_selector() -> Self {
        let mut consts = ConstPool::new();
        consts.branch_blocks.push((1, 1));

        let instructions = vec![
            Instruction::Dispatch {
                selector: 0, // never written
                blocks_idx: 0,
                blocks_len: 1,
            },
            Instruction::LoadScalar { value: 1.0, dst: 5 },
        ];

        Self {
            instructions,
            consts,
            slots: Vec::new(),
            n_states: 0,
            n_params: 0,
            uses_state_dot: false,
        }
    }
}

/// The earliest lowering blocker reachable from `root`, if any.
///
/// Checked before lowering so unsupported inputs surface as a Python error
/// instead of an FFI panic. Currently flags a `MatMul` whose left operand is
/// not a constant (`SparseMatrix` or `Array`).
pub fn first_unsupported(arena: &Arena, root: NodeId) -> Option<String> {
    for id in arena.topological_order(root) {
        if let Node::MatMul(a, _) = &arena[id] {
            match &arena[*a] {
                Node::SparseMatrix(_) | Node::Array(_) => {},
                other => {
                    return Some(format!(
                        "MatMul left operand must be a constant matrix, got {other:?}"
                    ));
                },
            }
        }
    }
    None
}

/// The earliest invalid evaluator shape relationship reachable from `root`.
///
/// Checked at Python entry points before lowering so malformed expression
/// graphs raise `ValueError` instead of reading adjacent evaluator scratch.
pub fn first_invalid(arena: &Arena, root: NodeId) -> Option<String> {
    let eval_order = arena.topological_order(root);
    if eval_order.iter().any(|&id| {
        matches!(
            &arena[id],
            Node::MatMul(a, _)
                if !matches!(&arena[*a], Node::SparseMatrix(_) | Node::Array(_))
        )
    }) {
        return None;
    }
    infer_sizes_checked(arena, &eval_order).err()
}

fn infer_sizes_checked(arena: &Arena, eval_order: &[NodeId]) -> Result<Vec<usize>, String> {
    let mut sizes = vec![0usize; arena.len()];

    for &node_id in eval_order {
        let size = match &arena[node_id] {
            Node::Scalar(_)
            | Node::Time
            | Node::TangentParameter { .. }
            | Node::MaxReduce(_)
            | Node::MinReduce(_)
            | Node::ReduceArgSelect { .. } => 1,
            Node::InputParameter { width, .. } => *width,
            Node::Array(arr) => arr.data.len(),
            Node::ZeroVector { len } => *len,
            Node::StateVector { start, end }
            | Node::StateVectorDot { start, end }
            | Node::TangentStateVector { start, end }
            | Node::Index { start, end, .. } => end.checked_sub(*start).ok_or_else(|| {
                format!("node {node_id:?} has an inverted extent: start={start}, end={end}")
            })?,
            Node::Add(a, b)
            | Node::Sub(a, b)
            | Node::Mul(a, b)
            | Node::Div(a, b)
            | Node::Pow(a, b)
            | Node::Minimum(a, b)
            | Node::Maximum(a, b)
            | Node::Modulo(a, b)
            | Node::Hypot(a, b)
            | Node::EqualHeaviside(a, b)
            | Node::NotEqualHeaviside(a, b)
            | Node::Equality(a, b) => {
                let (a_len, b_len) = (sizes[a.index()], sizes[b.index()]);
                if !broadcast_widths_compatible(a_len, b_len) {
                    return Err(format!(
                        "binary node {node_id:?} has incompatible operand widths {a_len} and \
                         {b_len}; widths must match or one operand must be scalar"
                    ));
                }
                a_len.max(b_len)
            },
            Node::MatMul(a, b) => {
                let (rows, cols) = match &arena[*a] {
                    Node::SparseMatrix(csr) => (csr.shape.rows, csr.shape.cols),
                    Node::Array(arr) => (arr.shape.rows, arr.shape.cols),
                    _ => {
                        return Err("MatMul requires a constant matrix on the left".to_string());
                    },
                };
                let b_len = sizes[b.index()];
                if b_len != cols {
                    return Err(format!(
                        "MatMul node {node_id:?} has {cols} columns but its vector operand has \
                         width {b_len}"
                    ));
                }
                rows
            },
            Node::Concat(children) => children.iter().map(|c| sizes[c.index()]).sum(),
            Node::InterpolantNd { data, children }
            | Node::InterpolantNdPartial { data, children, .. } => {
                if children.len() != data.breakpoints.len() {
                    return Err(format!(
                        "N-D interpolant node {node_id:?} has {} children for {} axes",
                        children.len(),
                        data.breakpoints.len()
                    ));
                }
                let out_len = children.iter().map(|c| sizes[c.index()]).max().unwrap_or(1);
                if let Some(child) = children
                    .iter()
                    .find(|c| !matches!(sizes[c.index()], 1) && sizes[c.index()] != out_len)
                {
                    return Err(format!(
                        "N-D interpolant node {node_id:?} has child widths that cannot broadcast: \
                         {} and {out_len}",
                        sizes[child.index()]
                    ));
                }
                out_len
            },
            Node::Neg(a)
            | Node::Abs(a)
            | Node::Sqrt(a)
            | Node::Exp(a)
            | Node::Log(a)
            | Node::Sin(a)
            | Node::Cos(a)
            | Node::Tanh(a)
            | Node::Sinh(a)
            | Node::Cosh(a)
            | Node::Arcsinh(a)
            | Node::Arctan(a)
            | Node::Erf(a)
            | Node::Sign(a)
            | Node::Floor(a)
            | Node::Ceiling(a) => sizes[a.index()],
            Node::Interpolant1DLinear { child, .. }
            | Node::Interpolant1DLinearDeriv { child, .. }
            | Node::Interpolant1DCubic { child, .. }
            | Node::Interpolant1DCubicDeriv { child, .. } => sizes[child.index()],
            Node::Conditional { selector, branches } => {
                let selector_len = sizes[selector.index()];
                if selector_len != 1 {
                    return Err(format!(
                        "Conditional node {node_id:?} requires a scalar selector, got width \
                         {selector_len}"
                    ));
                }
                let out_len = branches.first().map_or(1, |branch| sizes[branch.index()]);
                if let Some(branch) = branches
                    .iter()
                    .find(|branch| sizes[branch.index()] != out_len)
                {
                    return Err(format!(
                        "Conditional node {node_id:?} has branch widths {out_len} and {}",
                        sizes[branch.index()]
                    ));
                }
                out_len
            },
            Node::SparseMatrix(_) => 0,
        };

        match &arena[node_id] {
            Node::Index { child, end, .. } if *end > sizes[child.index()] => {
                return Err(format!(
                    "Index node {node_id:?} ends at {end}, beyond child width {}",
                    sizes[child.index()]
                ));
            },
            Node::ReduceArgSelect { basis, picker, .. } => {
                let (basis_len, picker_len) = (sizes[basis.index()], sizes[picker.index()]);
                if basis_len == 0 {
                    return Err(format!(
                        "ReduceArgSelect node {node_id:?} requires non-empty operands"
                    ));
                }
                if basis_len != picker_len {
                    return Err(format!(
                        "ReduceArgSelect node {node_id:?} has basis width {basis_len} and picker \
                         width {picker_len}"
                    ));
                }
            },
            Node::InterpolantNdPartial { data, axis, .. }
                if *axis as usize >= data.breakpoints.len() =>
            {
                return Err(format!(
                    "N-D interpolant partial node {node_id:?} has axis {axis} for {} axes",
                    data.breakpoints.len()
                ));
            },
            _ => {},
        }
        sizes[node_id.index()] = size;
    }
    Ok(sizes)
}

/// Infer the output size of each node in topological evaluation order.
pub fn infer_sizes(arena: &Arena, eval_order: &[NodeId]) -> Vec<usize> {
    infer_sizes_checked(arena, eval_order)
        .unwrap_or_else(|message| panic!("invalid expression graph: {message}"))
}

/// One slot range an instruction reads.
#[derive(Clone, Copy, Debug)]
struct SrcExtent {
    offset: u32,
    len: u32,
    /// A `Conditional`'s branch slot: the only read that may legitimately
    /// resolve to a definition owned by a branch block.
    is_branch_slot: bool,
}

impl SrcExtent {
    const fn read(offset: u32, len: u32) -> Self {
        Self {
            offset,
            len,
            is_branch_slot: false,
        }
    }

    const fn branch_slot(offset: u32, len: u32) -> Self {
        Self {
            offset,
            len,
            is_branch_slot: true,
        }
    }
}

/// Destination slot range `(offset, len)` an instruction writes, or `None` for
/// `Dispatch`, which writes nothing.
///
/// Exact, not an over-approximation: [`IRBuilder::assert_block_slots_private`]
/// tracks per-element last writers, so a short extent would leave stale entries
/// and a long one would mask a real violation.
fn instruction_dst_extent(instr: &Instruction, consts: &ConstPool) -> Option<(u32, u32)> {
    let extent = match *instr {
        Instruction::LoadScalar { dst, .. }
        | Instruction::LoadTime { dst }
        | Instruction::LoadTangentParameter { dst, .. }
        | Instruction::MaxReduce { dst, .. }
        | Instruction::MinReduce { dst, .. }
        | Instruction::ReduceArgSelect { dst, .. } => (dst, 1),
        Instruction::LoadArray { dst, len, .. }
        | Instruction::FillZero { dst, len }
        | Instruction::Binary { dst, len, .. }
        | Instruction::Unary { dst, len, .. }
        | Instruction::Index { dst, len, .. }
        | Instruction::Interp1DLinear { dst, len, .. }
        | Instruction::Interp1DLinearDeriv { dst, len, .. }
        | Instruction::Interp1DCubic { dst, len, .. }
        | Instruction::Interp1DCubicDeriv { dst, len, .. }
        | Instruction::InterpNd { dst, len, .. }
        | Instruction::InterpNdPartial { dst, len, .. } => (dst, len),
        Instruction::LoadStateVector { start, end, dst }
        | Instruction::LoadStateVectorDot { start, end, dst }
        | Instruction::LoadTangentState { start, end, dst } => (dst, end - start),
        Instruction::LoadInputParameter { width, dst, .. } => (dst, width),
        Instruction::Concat {
            sources_idx,
            sources_len,
            dst,
        } => {
            let width = (0..sources_len as usize)
                .map(|i| consts.concat_sources[sources_idx as usize + i].1)
                .sum();
            (dst, width)
        },
        Instruction::MatMul { csr_idx, dst, .. } => {
            (dst, consts.csr_data[csr_idx as usize].shape.rows as u32)
        },
        Instruction::DenseMatMul { rows, dst, .. } => (dst, rows),
        Instruction::Conditional { dst, out_len, .. } => (dst, out_len),
        Instruction::Dispatch { .. } => return None,
    };
    Some(extent)
}

/// Slot ranges an instruction reads, with exact widths.
fn instruction_src_extents(instr: &Instruction, consts: &ConstPool) -> Vec<SrcExtent> {
    match *instr {
        Instruction::LoadScalar { .. }
        | Instruction::LoadTime { .. }
        | Instruction::LoadArray { .. }
        | Instruction::FillZero { .. }
        | Instruction::LoadStateVector { .. }
        | Instruction::LoadStateVectorDot { .. }
        | Instruction::LoadInputParameter { .. }
        | Instruction::LoadTangentState { .. }
        | Instruction::LoadTangentParameter { .. } => Vec::new(),
        Instruction::Binary {
            a, b, len, kind, ..
        } => {
            // The broadcast pattern decides which operand is the scalar one.
            let (a_len, b_len) = match kind {
                BroadcastKind::ScalarScalar => (1, 1),
                BroadcastKind::ScalarVector => (1, len),
                BroadcastKind::VectorScalar => (len, 1),
                BroadcastKind::VectorVector => (len, len),
            };
            vec![SrcExtent::read(a, a_len), SrcExtent::read(b, b_len)]
        },
        Instruction::Unary { src, len, .. }
        | Instruction::Interp1DLinear { src, len, .. }
        | Instruction::Interp1DLinearDeriv { src, len, .. }
        | Instruction::Interp1DCubic { src, len, .. }
        | Instruction::Interp1DCubicDeriv { src, len, .. } => vec![SrcExtent::read(src, len)],
        Instruction::MaxReduce { src, src_len, .. }
        | Instruction::MinReduce { src, src_len, .. } => {
            vec![SrcExtent::read(src, src_len)]
        },
        Instruction::ReduceArgSelect {
            basis_src,
            picker_src,
            len,
            ..
        } => vec![
            SrcExtent::read(basis_src, len),
            SrcExtent::read(picker_src, len),
        ],
        Instruction::Index {
            src, start, len, ..
        } => vec![SrcExtent::read(src + start, len)],
        Instruction::Concat {
            sources_idx,
            sources_len,
            ..
        } => (0..sources_len as usize)
            .map(|i| {
                let (off, len) = consts.concat_sources[sources_idx as usize + i];
                SrcExtent::read(off, len)
            })
            .collect(),
        Instruction::MatMul {
            csr_idx, vec_src, ..
        } => vec![SrcExtent::read(
            vec_src,
            consts.csr_data[csr_idx as usize].shape.cols as u32,
        )],
        Instruction::DenseMatMul {
            mat_src,
            rows,
            cols,
            vec_src,
            ..
        } => vec![
            SrcExtent::read(mat_src, rows * cols),
            SrcExtent::read(vec_src, cols),
        ],
        Instruction::InterpNd {
            interp_idx,
            sources_idx,
            ..
        }
        | Instruction::InterpNdPartial {
            interp_idx,
            sources_idx,
            ..
        } => {
            // The axis count lives on the interpolant table, not the instruction.
            let ndim = consts.nd_interpolants[interp_idx as usize]
                .breakpoints
                .len();
            (0..ndim)
                .map(|a| {
                    let (off, len) = consts.interp_nd_sources[sources_idx as usize + a];
                    SrcExtent::read(off, len)
                })
                .collect()
        },
        Instruction::Conditional {
            selector,
            branches_idx,
            branches_len,
            out_len,
            ..
        } => {
            let mut srcs = vec![SrcExtent::read(selector, 1)];
            srcs.extend((0..branches_len as usize).map(|i| {
                let (off, _) = consts.branch_offsets[branches_idx as usize + i];
                SrcExtent::branch_slot(off, out_len)
            }));
            srcs
        },
        Instruction::Dispatch { selector, .. } => vec![SrcExtent::read(selector, 1)],
    }
}

/// Classify nodes as primal or tangent via forward taint propagation.
fn classify_tangent_nodes(arena: &Arena, eval_order: &[NodeId]) -> Vec<bool> {
    let mut is_tangent = vec![false; arena.len()];

    for &nid in eval_order {
        let node = arena.get(nid);
        let tainted = match node {
            Node::TangentStateVector { .. } | Node::TangentParameter { .. } => true,
            _ => {
                let mut any_child_tangent = false;
                node.for_each_child(|c| {
                    if is_tangent[c.index()] {
                        any_child_tangent = true;
                    }
                });
                any_child_tangent
            },
        };
        is_tangent[nid.index()] = tainted;
    }

    is_tangent
}

/// Assign buffer slots, computing last-use positions from the evaluation order.
fn assign_slots(
    arena: &Arena,
    sizes: &[usize],
    eval_order: &[NodeId],
    roots: &[NodeId],
) -> (Vec<(usize, usize)>, usize) {
    let n = sizes.len();

    // Compute last_use[node] from the eval_order
    let mut last_use = vec![0_usize; n];
    for (pos, &node_id) in eval_order.iter().enumerate() {
        arena.get(node_id).for_each_child(|c| {
            let cur = last_use[c.index()];
            if pos > cur {
                last_use[c.index()] = pos;
            }
        });
    }
    for &r in roots {
        last_use[r.index()] = usize::MAX;
    }

    assign_slots_configured(sizes, eval_order, &last_use, 0)
}

/// Slot allocation with **no reuse**: every node's slot is pinned for the
/// whole evaluation, so the primal scratch preserves every intermediate.
/// Required by the reverse-AD value tape, whose backward pass reads each
/// operand's recorded value by its (now stable) slot.
fn assign_slots_pinned(sizes: &[usize], eval_order: &[NodeId]) -> (Vec<(usize, usize)>, usize) {
    let last_use = vec![usize::MAX; sizes.len()];
    assign_slots_configured(sizes, eval_order, &last_use, 0)
}

/// Sweep-line buffer slot allocator with externalized lifetime control.
///
/// Assigns buffer offsets in `eval_order`, reusing freed regions.
/// `last_use[node_id]` is the last position that reads from the node;
/// set to `usize::MAX` to pin a slot permanently (e.g. roots).
/// `initial_high_water` offsets all allocations (for dual-pool partitioning).
fn assign_slots_configured(
    sizes: &[usize],
    eval_order: &[NodeId],
    last_use: &[usize],
    initial_high_water: usize,
) -> (Vec<(usize, usize)>, usize) {
    use std::collections::{BTreeMap, BTreeSet};

    let n = sizes.len();
    let mut slots = vec![(0_usize, 0_usize); n];

    let mut by_offset: BTreeMap<usize, usize> = BTreeMap::new();
    let mut by_size: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut high_water: usize = initial_high_water;

    let mut release_at: Vec<Vec<NodeId>> = vec![Vec::new(); eval_order.len() + 1];
    for &node_id in eval_order {
        let lu = last_use[node_id.index()];
        if lu != usize::MAX && lu < eval_order.len() {
            release_at[lu + 1].push(node_id);
        }
    }

    for (pos, &node_id) in eval_order.iter().enumerate() {
        // Release phase
        for &n_to_free in &release_at[pos] {
            let (off, len) = slots[n_to_free.index()];
            if len == 0 {
                continue;
            }

            let mut new_off = off;
            let mut new_len = len;
            if let Some(&right_len) = by_offset.get(&(off + len)) {
                by_offset.remove(&(off + len));
                by_size.remove(&(right_len, off + len));
                new_len += right_len;
            }
            if let Some((&left_off, &left_len)) = by_offset.range(..off).next_back()
                && left_off + left_len == off
            {
                by_offset.remove(&left_off);
                by_size.remove(&(left_len, left_off));
                new_off = left_off;
                new_len += left_len;
            }
            by_offset.insert(new_off, new_len);
            by_size.insert((new_len, new_off));
        }

        // Allocate phase
        let len = sizes[node_id.index()];
        if len == 0 {
            slots[node_id.index()] = (0, 0);
            continue;
        }

        let chosen = by_size.range((len, 0)..).next().copied();
        let off = if let Some((region_len, region_off)) = chosen {
            by_size.remove(&(region_len, region_off));
            by_offset.remove(&region_off);
            if region_len > len {
                let rem_off = region_off + len;
                let rem_len = region_len - len;
                by_offset.insert(rem_off, rem_len);
                by_size.insert((rem_len, rem_off));
            }
            region_off
        } else {
            let off = high_water;
            high_water += len;
            off
        };
        slots[node_id.index()] = (off, len);
    }

    (slots, high_water)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::CompiledExpr;
    use crate::node::{ArrayData, NdInterpolantData, Shape};

    #[test]
    #[should_panic(expected = "outside a branch block reads buffer element")]
    fn assert_block_slots_private_catches_a_cross_block_read() {
        let builder = IRBuilder::test_tape_reading_across_block_boundary();
        builder.assert_block_slots_private(7);
    }

    #[test]
    #[should_panic(expected = "which no earlier instruction defines")]
    fn assert_block_slots_private_catches_an_undefined_selector() {
        let builder = IRBuilder::test_tape_dispatching_on_an_undefined_selector();
        builder.assert_block_slots_private(7);
    }

    #[test]
    fn test_typed_ir_from_expression() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let two = arena.alloc(Node::Scalar(2.0));
        let expr = arena.alloc(Node::Mul(two, y));

        let ir = TypedIr::from_arena(&arena, expr);

        assert_eq!(ir.output_len(), 3);
        assert_eq!(ir.instructions().len(), 3); // Scalar, StateVector, Mul
    }

    #[test]
    fn test_slot_struct() {
        let slot = Slot::new(10, 5);
        assert_eq!(slot.offset, 10);
        assert_eq!(slot.len, 5);
        assert_eq!(slot.offset_usize(), 10);
        assert_eq!(slot.len_usize(), 5);
        assert!(!slot.is_scalar());

        let scalar_slot = Slot::new(0, 1);
        assert!(scalar_slot.is_scalar());
    }

    #[test]
    fn test_broadcast_kind_inference() {
        assert_eq!(BroadcastKind::from_lens(1, 1), BroadcastKind::ScalarScalar);
        assert_eq!(BroadcastKind::from_lens(1, 5), BroadcastKind::ScalarVector);
        assert_eq!(BroadcastKind::from_lens(5, 1), BroadcastKind::VectorScalar);
        assert_eq!(BroadcastKind::from_lens(5, 5), BroadcastKind::VectorVector);
    }

    #[test]
    fn test_ir_metadata() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 10 });
        let p = arena.alloc(Node::InputParameter {
            name: "param".to_string(),
            index: 2,
            offset: 2,
            width: 1,
        });
        let expr = arena.alloc(Node::Mul(y, p));

        let ir = TypedIr::from_arena(&arena, expr);

        assert_eq!(ir.n_states(), 10);
        assert_eq!(ir.n_params(), 3); // index 2 means 3 params (0, 1, 2)
        assert!(!ir.uses_state_dot());
    }

    #[test]
    fn test_ir_with_state_dot() {
        let mut arena = Arena::new();
        let y_dot = arena.alloc(Node::StateVectorDot { start: 0, end: 5 });
        let two = arena.alloc(Node::Scalar(2.0));
        let expr = arena.alloc(Node::Mul(two, y_dot));

        let ir = TypedIr::from_arena(&arena, expr);

        assert!(ir.uses_state_dot());
        assert_eq!(ir.n_states(), 5);
    }

    #[test]
    fn test_ir_nested_expression() {
        let mut arena = Arena::new();
        // Build: sin(x * 2 + 1) where x is a state vector
        let x = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let two = arena.alloc(Node::Scalar(2.0));
        let one = arena.alloc(Node::Scalar(1.0));
        let mul = arena.alloc(Node::Mul(x, two));
        let add = arena.alloc(Node::Add(mul, one));
        let sin = arena.alloc(Node::Sin(add));

        let ir = TypedIr::from_arena(&arena, sin);

        assert_eq!(ir.output_len(), 3);
        // Instructions: StateVector, Scalar(2), Scalar(1), Mul, Add, Sin
        assert_eq!(ir.instructions().len(), 6);
    }

    #[test]
    fn test_ir_concat() {
        let mut arena = Arena::new();
        let a = arena.alloc(Node::Scalar(1.0));
        let b = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![2.0, 3.0],
            shape: Shape::vector(2),
        })));
        let c = arena.alloc(Node::Scalar(4.0));
        let concat = arena.alloc(Node::Concat(vec![a, b, c]));

        let ir = TypedIr::from_arena(&arena, concat);

        assert_eq!(ir.output_len(), 4);
        assert_eq!(ir.consts().concat_sources.len(), 3);
    }

    #[test]
    fn test_first_unsupported_flags_non_constant_matmul_lhs() {
        let mut arena = Arena::new();
        let s = arena.alloc(Node::Scalar(2.0));
        let v = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0],
            shape: Shape::vector(1),
        })));
        let mm = arena.alloc(Node::MatMul(s, v));
        assert!(first_unsupported(&arena, mm).is_some());
        assert!(first_unsupported(&arena, v).is_none());
    }

    #[test]
    fn test_symbolic_jacobian_width_mismatch_rejected_before_eval_overlap() {
        let mut arena = Arena::new();
        let short = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let long = arena.alloc(Node::StateVector { start: 2, end: 5 });
        let add = arena.alloc(Node::Add(short, long));
        let scalar_entry = arena.alloc(Node::Scalar(1.0));
        let symbolic_jacobian = arena.alloc(Node::Concat(vec![add, scalar_entry]));

        // This layout previously reached split_dst_two_src with len=3 for the
        // width-2 operand, making its fictitious read window overlap dst.
        let message =
            first_invalid(&arena, symbolic_jacobian).expect("binary widths must be rejected");
        assert!(message.contains("incompatible operand widths 2 and 3"));
    }

    #[test]
    fn test_first_invalid_flags_other_evaluator_width_hazards() {
        let mut matmul_arena = Arena::new();
        let matrix = matmul_arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0; 6],
            shape: Shape::matrix(2, 3),
        })));
        let short = matmul_arena.alloc(Node::StateVector { start: 0, end: 2 });
        let matmul = matmul_arena.alloc(Node::MatMul(matrix, short));
        assert!(
            first_invalid(&matmul_arena, matmul)
                .expect("MatMul width must be rejected")
                .contains("has 3 columns but its vector operand has width 2")
        );

        let mut index_arena = Arena::new();
        let child = index_arena.alloc(Node::StateVector { start: 0, end: 2 });
        let index = index_arena.alloc(Node::Index {
            child,
            start: 1,
            end: 3,
        });
        assert!(
            first_invalid(&index_arena, index)
                .expect("Index bounds must be rejected")
                .contains("beyond child width 2")
        );

        let mut conditional_arena = Arena::new();
        let selector = conditional_arena.alloc(Node::Scalar(1.0));
        let short = conditional_arena.alloc(Node::StateVector { start: 0, end: 2 });
        let long = conditional_arena.alloc(Node::StateVector { start: 2, end: 5 });
        let conditional = conditional_arena.alloc(Node::Conditional {
            selector,
            branches: vec![short, long],
        });
        assert!(
            first_invalid(&conditional_arena, conditional)
                .expect("Conditional widths must be rejected")
                .contains("branch widths 2 and 3")
        );

        let mut interpolant_arena = Arena::new();
        let short = interpolant_arena.alloc(Node::StateVector { start: 0, end: 2 });
        let long = interpolant_arena.alloc(Node::StateVector { start: 2, end: 5 });
        let interpolant = interpolant_arena.alloc(Node::InterpolantNd {
            data: Box::new(NdInterpolantData {
                breakpoints: vec![vec![0.0, 1.0], vec![0.0, 1.0]],
                coeffs: vec![0.0; 4],
                order: 2,
            }),
            children: vec![short, long],
        });
        assert!(
            first_invalid(&interpolant_arena, interpolant)
                .expect("N-D interpolant widths must be rejected")
                .contains("cannot broadcast: 2 and 3")
        );

        let mut reduce_arena = Arena::new();
        let basis = reduce_arena.alloc(Node::StateVector { start: 0, end: 2 });
        let picker = reduce_arena.alloc(Node::StateVector { start: 2, end: 5 });
        let reduce = reduce_arena.alloc(Node::ReduceArgSelect {
            basis,
            picker,
            is_max: true,
        });
        assert!(
            first_invalid(&reduce_arena, reduce)
                .expect("ReduceArgSelect widths must be rejected")
                .contains("basis width 2 and picker width 3")
        );

        let mut empty_reduce_arena = Arena::new();
        let basis = empty_reduce_arena.alloc(Node::ZeroVector { len: 0 });
        let picker = empty_reduce_arena.alloc(Node::ZeroVector { len: 0 });
        let reduce = empty_reduce_arena.alloc(Node::ReduceArgSelect {
            basis,
            picker,
            is_max: true,
        });
        assert!(
            first_invalid(&empty_reduce_arena, reduce)
                .expect("empty ReduceArgSelect must be rejected")
                .contains("requires non-empty operands")
        );
    }

    #[test]
    fn test_ir_dense_matmul_output_len() {
        let mut arena = Arena::new();
        let a = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // row-major 2x3
            shape: Shape::matrix(2, 3),
        })));
        let v = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0, 1.0, 1.0],
            shape: Shape::vector(3),
        })));
        let matmul = arena.alloc(Node::MatMul(a, v));
        let ir = TypedIr::from_arena(&arena, matmul);
        assert_eq!(ir.output_len(), 2);
    }

    #[test]
    fn test_ir_interpolant() {
        use crate::node::InterpolantData;

        let mut arena = Arena::new();
        let x = arena.alloc(Node::Scalar(1.5));
        let interp = arena.alloc(Node::Interpolant1DLinear {
            data: Box::new(InterpolantData {
                x_data: vec![0.0, 1.0, 2.0],
                y_data: vec![0.0, 10.0, 20.0],
            }),
            child: x,
        });

        let ir = TypedIr::from_arena(&arena, interp);

        assert_eq!(ir.output_len(), 1);
        assert_eq!(ir.consts().interpolants.len(), 1);
        assert_eq!(ir.consts().interpolants[0].x_data, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_const_pool() {
        let mut pool = ConstPool::new();

        // Test array storage
        let idx1 = pool.add_array(&[1.0, 2.0, 3.0]);
        let idx2 = pool.add_array(&[4.0, 5.0]);
        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(pool.get_array(idx1, 3), &[1.0, 2.0, 3.0]);
        assert_eq!(pool.get_array(idx2, 2), &[4.0, 5.0]);

        // Test interpolant storage
        let interp_idx = pool.add_interpolant(vec![0.0, 1.0], vec![0.0, 10.0]);
        assert_eq!(interp_idx, 0);
        assert_eq!(pool.interpolants[0].x_data, vec![0.0, 1.0]);
    }

    #[test]
    fn test_assign_slots_reuses_freed_slot_in_simple_chain() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let a = arena.alloc(Node::Sin(y));
        let b = arena.alloc(Node::Cos(a));
        let root = arena.alloc(Node::Neg(b));

        let eval_order = arena.topological_order(root);
        let sizes = infer_sizes(&arena, &eval_order);
        let (slots, total) = assign_slots(&arena, &sizes, &eval_order, &[root]);

        assert!(
            total <= 6,
            "expected slot reuse to reduce buffer below 6, got {total}"
        );

        for &(off, len) in &slots {
            assert!(
                off + len <= total,
                "invalid slot ({off}, {len}) > total {total}"
            );
        }
    }

    #[test]
    fn test_assign_slots_coalesces_adjacent_free_regions() {
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y1 = arena.alloc(Node::StateVector { start: 2, end: 4 });
        let s0 = arena.alloc(Node::Sin(y0));
        let s1 = arena.alloc(Node::Sin(y1));
        let sum = arena.alloc(Node::Add(s0, s1));
        let cat = arena.alloc(Node::Concat(vec![y0, y1]));
        let big = arena.alloc(Node::Sin(cat));
        let repeated_sum = arena.alloc(Node::Concat(vec![sum, sum]));
        let root = arena.alloc(Node::Add(big, repeated_sum));

        let eval_order = arena.topological_order(root);
        let sizes = infer_sizes(&arena, &eval_order);
        let (_slots, total) = assign_slots(&arena, &sizes, &eval_order, &[root]);

        let no_reuse: usize = eval_order.iter().map(|n| sizes[n.index()]).sum();
        assert!(
            total < no_reuse,
            "expected slot reuse total ({total}) < sequential sum ({no_reuse})"
        );
    }

    #[test]
    fn test_pinned_layout_disables_reuse_and_preserves_intermediates() {
        // f = (y0*y1) + y2 ; the product `a` is dead after the add, so the
        // reuse allocator recycles its slot, but the pinned layout must not.
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let y2 = arena.alloc(Node::StateVector { start: 2, end: 3 });
        let a = arena.alloc(Node::Mul(y0, y1));
        let root = arena.alloc(Node::Add(a, y2));

        let reuse = TypedIr::from_arena(&arena, root);
        let pinned = TypedIr::from_arena_pinned(&arena, root);

        // No reuse => strictly larger (or equal) buffer; here strictly larger.
        assert!(pinned.buffer_size() > reuse.buffer_size());

        // After a pinned eval, the intermediate product a = y0*y1 = 2*3 = 6 is
        // still live at its own slot (reuse layout would have overwritten it).
        let compiled = CompiledExpr::from_ir(pinned);
        let mut s = vec![0.0; compiled.scratch_len()];
        let out = compiled.eval(&mut s, 0.0, &[2.0, 3.0, 5.0], &[], &[]);
        assert_eq!(out, &[11.0]);
        // Every node has a unique slot: the number of distinct slot offsets in
        // the instruction stream equals the instruction count (SSA).
        let ir = compiled.ir();
        let mut offsets: Vec<usize> = Vec::new();
        for instr in ir.instructions() {
            let dst = instruction_dst(instr);
            if dst != usize::MAX {
                offsets.push(dst);
            }
        }
        let unique: std::collections::BTreeSet<usize> = offsets.iter().copied().collect();
        assert_eq!(unique.len(), offsets.len(), "pinned layout reused a slot");
    }

    // Small test-only helper: the dst offset of any instruction.
    fn instruction_dst(instr: &Instruction) -> usize {
        match *instr {
            Instruction::LoadScalar { dst, .. }
            | Instruction::LoadTime { dst }
            | Instruction::LoadArray { dst, .. }
            | Instruction::FillZero { dst, .. }
            | Instruction::LoadStateVector { dst, .. }
            | Instruction::LoadStateVectorDot { dst, .. }
            | Instruction::LoadInputParameter { dst, .. }
            | Instruction::LoadTangentState { dst, .. }
            | Instruction::LoadTangentParameter { dst, .. }
            | Instruction::Binary { dst, .. }
            | Instruction::Unary { dst, .. }
            | Instruction::MaxReduce { dst, .. }
            | Instruction::MinReduce { dst, .. }
            | Instruction::ReduceArgSelect { dst, .. }
            | Instruction::Index { dst, .. }
            | Instruction::Concat { dst, .. }
            | Instruction::MatMul { dst, .. }
            | Instruction::DenseMatMul { dst, .. }
            | Instruction::Interp1DLinear { dst, .. }
            | Instruction::Interp1DLinearDeriv { dst, .. }
            | Instruction::Interp1DCubic { dst, .. }
            | Instruction::Interp1DCubicDeriv { dst, .. }
            | Instruction::InterpNd { dst, .. }
            | Instruction::InterpNdPartial { dst, .. }
            | Instruction::Conditional { dst, .. } => dst as usize,
            Instruction::Dispatch { .. } => usize::MAX,
        }
    }
}
