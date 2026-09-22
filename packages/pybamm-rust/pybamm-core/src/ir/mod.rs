//! Lowering from the expression DAG to a flat [`TypedIr`] instruction tape, with
//! operands as [`Slot`]s into one scratch buffer and constants in a [`ConstPool`].

pub mod ids;

use crate::arena::{Arena, DagRef, NodeId};
use crate::branch_regions::{
    RegionGroup, privatise_conditionals, region_owners, schedule_regions,
    schedule_regions_partitioned,
};
use crate::ir::ids::{
    ArrayIdx, Axis, BlockSpan, BlockSpanIdx, BufSlot, Cols, CsrIdx, CubicInterpolantIdx,
    EntryCount, InputOffset, InstructionCount, InterpolantIdx, NdInterpolantIdx, Rows, SlotOffset,
    SlotTableIdx, StateIdx, Width,
};
use crate::node::{CsrData, CubicInterpolantData, InterpolantData, NdInterpolantData, Node};
use crate::params::ParamIdx;

/// A contiguous window of the evaluation buffer: where one value lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub offset: BufSlot,
    pub len: Width,
}

impl Slot {
    #[inline]
    pub const fn new(offset: BufSlot, len: Width) -> Self {
        Self { offset, len }
    }

    #[inline]
    pub(crate) fn from_usize(offset: usize, len: usize) -> Self {
        Self {
            offset: BufSlot::from_usize(offset),
            len: Width::from_usize(len),
        }
    }

    #[inline]
    pub(crate) const fn offset_usize(self) -> usize {
        self.offset.index()
    }

    #[inline]
    pub(crate) const fn len_usize(self) -> usize {
        self.len.index()
    }

    #[inline]
    pub const fn bounds(self) -> std::ops::Range<usize> {
        self.offset_usize()..self.offset_usize() + self.len_usize()
    }

    /// Whether this slot holds a single element (and so broadcasts).
    #[inline]
    pub(crate) const fn is_scalar(self) -> bool {
        self.len.get() == 1
    }
}

/// Compile-time broadcast pattern for binary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BroadcastKind {
    ScalarScalar = 0,
    /// Left operand is a single element, broadcast across the right.
    ScalarVector = 1,
    /// Right operand is a single element, broadcast across the left.
    VectorScalar = 2,
    VectorVector = 3,
}

impl BroadcastKind {
    /// Classify a pair of operand widths; panics on a mismatch [`validate`] rejects.
    #[inline]
    pub(crate) fn from_lens(a_len: usize, b_len: usize) -> Self {
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

/// Element-wise binary operation, without domain guards; comparisons return 1.0 or 0.0.
///
/// IEEE 754's exactly specified operations take `f64`'s values, the rest the `libm` crate's,
/// so every value is bitwise the same on every target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    /// IEEE 754-2019 `minimumNumber`: the non-NaN operand if exactly one is NaN, and
    /// `-0 < +0`.
    Minimum,
    /// IEEE 754-2019 `maximumNumber`: the non-NaN operand if exactly one is NaN, and
    /// `-0 < +0`.
    Maximum,
    /// Truncated remainder (`%`): the sign follows the dividend, unlike Python's `%`.
    Modulo,
    Hypot,
    /// `a <= b`, the branch-inclusive Heaviside: 1.0 when the operands are equal.
    EqualHeaviside,
    /// `a < b`, the strict Heaviside: 0.0 when the operands are equal.
    NotEqualHeaviside,
    /// `|a - b| < 1e-14`, a tolerance test rather than an exact comparison.
    Equality,
}

/// Element-wise unary `f64` function, without domain guards.
///
/// IEEE 754's exactly specified operations take `f64`'s values, the rest the `libm` crate's,
/// so every value is bitwise the same on every target.
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
    Erf,
    /// `sign(0) = 0`, unlike `f64::signum`.
    Sign,
    Floor,
    Ceiling,
}

/// Single evaluation step.
///
/// `dst`/`src`/`a`/`b` are buffer offsets, `len` counts `f64` elements, and `*_idx`
/// addresses the [`ConstPool`]. Write windows never overlap read windows.
#[derive(Clone, Copy, Debug)]
pub enum Instruction {
    LoadScalar {
        value: f64,
        dst: BufSlot,
    },
    LoadTime {
        dst: BufSlot,
    },
    LoadArray {
        data_idx: ArrayIdx,
        len: Width,
        dst: BufSlot,
    },
    FillZero {
        dst: BufSlot,
        len: Width,
    },
    LoadStateVector {
        start: StateIdx,
        width: Width,
        dst: BufSlot,
    },
    LoadStateVectorDot {
        start: StateIdx,
        width: Width,
        dst: BufSlot,
    },
    LoadInputParameter {
        offset: InputOffset,
        width: Width,
        dst: BufSlot,
    },
    LoadTangentState {
        start: StateIdx,
        width: Width,
        dst: BufSlot,
    },
    /// Load the parameter seed `dp[index]` into `dst`.
    LoadTangentParameter {
        index: ParamIdx,
        dst: BufSlot,
    },
    /// Element-wise binary operation: `dst = a <op> b` with broadcast.
    Binary {
        op: BinaryOp,
        a: BufSlot,
        b: BufSlot,
        dst: BufSlot,
        len: Width,
        kind: BroadcastKind,
    },
    /// Element-wise unary operation: `dst = op(src)`.
    Unary {
        op: UnaryOp,
        src: BufSlot,
        dst: BufSlot,
        len: Width,
    },
    /// Reduce to a scalar: `dst = max(src[..src_len])`.
    MaxReduce {
        src: BufSlot,
        src_len: Width,
        dst: BufSlot,
    },
    /// Reduce to a scalar: `dst = min(src[..src_len])`.
    MinReduce {
        src: BufSlot,
        src_len: Width,
        dst: BufSlot,
    },
    /// Reduce subgradient: `dst = basis[k]`, `k` = argmax/argmin of
    /// `picker[..len]` (first occurrence, strict comparison).
    ReduceArgSelect {
        basis_src: BufSlot,
        picker_src: BufSlot,
        len: Width,
        is_max: bool,
        dst: BufSlot,
    },
    /// Slice: `dst = src[start..start+len]`.
    Index {
        src: BufSlot,
        start: SlotOffset,
        dst: BufSlot,
        len: Width,
    },
    /// Concatenate `sources_len` slot ranges into `dst`.
    Concat {
        sources_idx: SlotTableIdx,
        sources_len: EntryCount,
        dst: BufSlot,
    },
    /// Sparse matrix-vector multiply: `dst = CSR(csr_idx) @ vec(vec_src)`.
    MatMul {
        csr_idx: CsrIdx,
        vec_src: BufSlot,
        dst: BufSlot,
    },
    /// Dense matrix-vector multiply: `dst = A @ v`, A row-major at slot `mat_src` (rows × cols).
    DenseMatMul {
        mat_src: BufSlot,
        rows: Rows,
        cols: Cols,
        vec_src: BufSlot,
        dst: BufSlot,
    },
    /// 1-D linear interpolation: `dst[i] = interp(src[i])`.
    Interp1DLinear {
        interp_idx: InterpolantIdx,
        src: BufSlot,
        dst: BufSlot,
        len: Width,
    },
    /// 1-D linear interpolation derivative, read from the table's slopes.
    Interp1DLinearDeriv {
        interp_idx: InterpolantIdx,
        src: BufSlot,
        dst: BufSlot,
        len: Width,
    },
    /// 1-D cubic (pchip) interpolation.
    Interp1DCubic {
        interp_idx: CubicInterpolantIdx,
        src: BufSlot,
        dst: BufSlot,
        len: Width,
    },
    /// 1-D cubic (pchip) interpolation derivative.
    Interp1DCubicDeriv {
        interp_idx: CubicInterpolantIdx,
        src: BufSlot,
        dst: BufSlot,
        len: Width,
    },
    /// N-D tensor-product interpolation.
    InterpNd {
        interp_idx: NdInterpolantIdx,
        sources_idx: SlotTableIdx,
        dst: BufSlot,
        len: Width,
    },
    /// N-D tensor-product interpolation partial derivative along `axis`.
    InterpNdPartial {
        interp_idx: NdInterpolantIdx,
        sources_idx: SlotTableIdx,
        axis: Axis,
        dst: BufSlot,
        len: Width,
    },
    /// Piecewise conditional: select branches by `selector` value.
    Conditional {
        selector: BufSlot,
        branches_idx: SlotTableIdx,
        branches_len: EntryCount,
        dst: BufSlot,
        out_len: Width,
    },
    /// Skip all but the active branch's instruction block.
    ///
    /// Precedes the blocks it guards; their [`BlockSpan`]s are relative to this instruction.
    Dispatch {
        selector: BufSlot,
        blocks_idx: BlockSpanIdx,
        blocks_len: EntryCount,
        /// Instructions this `Dispatch` and its blocks occupy, so evaluation resumes past them.
        span_len: InstructionCount,
    },
}

// The interpreter streams the tape; keep each instruction to 24 bytes.
const _: () = assert!(size_of::<Instruction>() == 24);

/// Side tables an instruction addresses by index instead of carrying inline.
#[derive(Debug, Default, Clone)]
pub struct ConstPool {
    /// Flattened array data; `array_offsets[i]` gives the start of array `i`.
    pub array_data: Vec<f64>,
    pub array_offsets: Vec<usize>,

    pub csr_data: Vec<CsrData>,

    pub interpolants: Vec<InterpolantData>,

    pub cubic_interpolants: Vec<CubicInterpolantData>,

    pub nd_interpolants: Vec<NdInterpolantData>,

    /// Concat source windows, one per operand.
    pub concat_sources: Vec<Slot>,

    /// Conditional branch windows, one per branch.
    pub branch_offsets: Vec<Slot>,

    /// `Dispatch` branch blocks, in instruction space.
    pub branch_blocks: Vec<BlockSpan>,

    /// N-D interpolant child windows, one per input axis.
    pub interp_nd_sources: Vec<Slot>,
}

impl ConstPool {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn add_array(&mut self, data: &[f64]) -> ArrayIdx {
        let idx = ArrayIdx::of_next(&self.array_offsets);
        self.array_offsets.push(self.array_data.len());
        self.array_data.extend_from_slice(data);
        idx
    }

    pub(crate) fn get_array(&self, idx: ArrayIdx, len: Width) -> &[f64] {
        let start = self.array_offsets[idx.index()];
        &self.array_data[start..start + len.index()]
    }

    pub(crate) fn add_csr(&mut self, csr: &CsrData) -> CsrIdx {
        let idx = CsrIdx::of_next(&self.csr_data);
        self.csr_data.push(csr.clone());
        idx
    }

    pub(crate) fn add_interpolant(&mut self, data: &InterpolantData) -> InterpolantIdx {
        let idx = InterpolantIdx::of_next(&self.interpolants);
        self.interpolants.push(data.clone());
        idx
    }

    pub(crate) fn add_cubic_interpolant(
        &mut self,
        data: &CubicInterpolantData,
    ) -> CubicInterpolantIdx {
        let idx = CubicInterpolantIdx::of_next(&self.cubic_interpolants);
        self.cubic_interpolants.push(data.clone());
        idx
    }

    pub(crate) fn add_nd_interpolant(&mut self, data: &NdInterpolantData) -> NdInterpolantIdx {
        let idx = NdInterpolantIdx::of_next(&self.nd_interpolants);
        self.nd_interpolants.push(data.clone());
        idx
    }
}

/// Partition of a split-eval tape: primal instructions and slots precede the tangent ones.
#[derive(Debug, Clone, Copy)]
pub struct SplitEvalInfo {
    /// Index into the instruction stream where tangent instructions begin.
    pub primal_end: usize,
    /// Buffer region `[0, primal_buffer_size)` holds primal values.
    pub primal_buffer_size: usize,
}

/// Typed intermediate representation: a flat instruction tape with resolved shapes.
#[derive(Debug, Clone)]
pub struct TypedIr {
    instructions: Vec<Instruction>,

    root_slot: Slot,

    buffer_size: usize,

    consts: ConstPool,
}

impl TypedIr {
    /// Compile an expression DAG into the IR, privatising conditional branch cones first.
    pub fn from_dag(dag: DagRef<'_>) -> Self {
        Self::privatise_then(dag, IRBuilder::build)
    }

    /// [`from_dag`](Self::from_dag) without branch privatisation.
    #[cfg(test)]
    pub(crate) fn from_dag_raw(dag: DagRef<'_>) -> Self {
        IRBuilder::build(dag.arena(), dag.root())
    }

    /// Run [`privatise_conditionals`] on `dag`, then lower the result with `build`.
    fn privatise_then<T>(dag: DagRef<'_>, build: impl FnOnce(&Arena, NodeId) -> T) -> T {
        match privatise_conditionals(dag) {
            Some(privatised) => build(privatised.arena(), privatised.root()),
            None => build(dag.arena(), dag.root()),
        }
    }

    /// Output length of the root expression.
    #[inline]
    pub const fn output_len(&self) -> usize {
        self.root_slot.len.index()
    }

    /// Instruction stream in evaluation order.
    #[inline]
    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    /// Per-branch block lengths in tape order, one per branch of every `Dispatch`.
    #[must_use]
    pub fn branch_block_lens(&self) -> Vec<usize> {
        self.block_lens().collect()
    }

    /// Lowering emits one `branch_blocks` entry per branch of each `Dispatch`, in tape order.
    fn block_lens(&self) -> impl Iterator<Item = usize> + '_ {
        self.consts
            .branch_blocks
            .iter()
            .copied()
            .map(BlockSpan::len)
    }

    /// Number of `Dispatch` instructions, one per short-circuited conditional.
    #[must_use]
    pub fn dispatch_count(&self) -> usize {
        self.instructions
            .iter()
            .filter(|instr| matches!(instr, Instruction::Dispatch { .. }))
            .count()
    }

    /// Instruction count excluding conditional branch blocks (the `Dispatch`es included).
    #[must_use]
    pub fn common_instruction_count(&self) -> usize {
        self.instructions.len() - self.block_lens().sum::<usize>()
    }

    /// Output slot of the root expression.
    #[inline]
    pub(crate) const fn root_slot(&self) -> Slot {
        self.root_slot
    }

    /// Total evaluation buffer size in elements.
    #[inline]
    pub const fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    #[inline]
    pub const fn consts(&self) -> &ConstPool {
        &self.consts
    }

    /// Compile with a primal/tangent split, so only the tangent section reruns per colour.
    pub(crate) fn from_dag_split_eval(dag: DagRef<'_>) -> (Self, SplitEvalInfo) {
        Self::privatise_then(dag, IRBuilder::build_split_eval)
    }

    /// Compile with a no-reuse (SSA) slot layout, so the scratch is the reverse-AD value tape.
    pub fn from_dag_pinned(dag: DagRef<'_>) -> Self {
        Self::privatise_then(dag, IRBuilder::build_pinned)
    }
}

/// Builder for constructing `TypedIr` from an expression DAG.
struct IRBuilder {
    instructions: Vec<Instruction>,
    consts: ConstPool,
    /// The buffer window of each `NodeId` in the arena.
    slots: Vec<Slot>,
}

impl IRBuilder {
    /// A builder over the `(offset, len)` windows slot assignment produced.
    fn new(slots: Vec<(usize, usize)>, n_nodes: usize) -> Self {
        Self {
            instructions: Vec::with_capacity(n_nodes),
            consts: ConstPool::new(),
            slots: slots
                .into_iter()
                .map(|(offset, len)| Slot::from_usize(offset, len))
                .collect(),
        }
    }

    fn build(arena: &Arena, root: NodeId) -> TypedIr {
        Self::build_scheduled(arena, root, |sizes, order| {
            assign_slots(arena, sizes, order, &[root])
        })
    }

    /// Like [`Self::build`] but never recycles a slot (see [`assign_slots_pinned`]).
    fn build_pinned(arena: &Arena, root: NodeId) -> TypedIr {
        Self::build_scheduled(arena, root, assign_slots_pinned)
    }

    /// Lower in one region schedule, with slots from `assign(sizes, order)`.
    fn build_scheduled(
        arena: &Arena,
        root: NodeId,
        assign: impl FnOnce(&[usize], &[NodeId]) -> (Vec<(usize, usize)>, usize),
    ) -> TypedIr {
        let base_order = arena.topological_order(root);
        let schedule = schedule_regions(arena, &base_order);
        let sizes = infer_sizes(arena, &schedule.order);
        let (slots, buffer_size) = assign(&sizes, &schedule.order);

        let mut builder = Self::new(slots, schedule.order.len());
        builder.emit_scheduled(arena, &schedule.order, &schedule.groups);
        builder.assert_block_slots_private(buffer_size);
        builder.finish(root, buffer_size)
    }

    fn build_split_eval(arena: &Arena, root: NodeId) -> (TypedIr, SplitEvalInfo) {
        let eval_order = arena.topological_order(root);
        let sizes = infer_sizes(arena, &eval_order);
        let is_tangent = classify_tangent_nodes(arena, &eval_order);

        let mut primal_order = Vec::new();
        let mut tangent_order = Vec::new();
        for &nid in &eval_order {
            if is_tangent[nid.index()] {
                tangent_order.push(nid);
            } else {
                primal_order.push(nid);
            }
        }
        // Each half schedules alone, so a `Dispatch` and its blocks share a half.
        let owners = region_owners(arena, &eval_order);
        let schedule = |half: &[NodeId]| {
            schedule_regions_partitioned(arena, half, &eval_order, owners.as_deref())
        };
        let (primal_schedule, tangent_schedule) =
            (schedule(&primal_order), schedule(&tangent_order));
        let partitioned_order: Vec<NodeId> = primal_schedule
            .order
            .iter()
            .chain(tangent_schedule.order.iter())
            .copied()
            .collect();
        // Whole-tape positions pin primal values the tangent half reads; tangent ones are local.
        let primal_last_use = last_use_positions(arena, &partitioned_order, &[root]);
        let tangent_last_use = last_use_positions(arena, &tangent_schedule.order, &[root]);

        // Primal slots occupy `[0, primal_hw)`, tangent slots `[primal_hw, total_hw)`.
        let (primal_slots, primal_hw) =
            assign_slots_configured(&sizes, &primal_schedule.order, &primal_last_use, 0);

        let (tangent_slots, total_hw) = assign_slots_configured(
            &sizes,
            &tangent_schedule.order,
            &tangent_last_use,
            primal_hw,
        );

        let mut slots = vec![(0_usize, 0_usize); sizes.len()];
        for &nid in &primal_schedule.order {
            slots[nid.index()] = primal_slots[nid.index()];
        }
        for &nid in &tangent_schedule.order {
            slots[nid.index()] = tangent_slots[nid.index()];
        }

        let mut builder = Self::new(slots, partitioned_order.len());
        builder.emit_scheduled(arena, &primal_schedule.order, &primal_schedule.groups);
        let primal_end = builder.instructions.len();
        builder.emit_scheduled(arena, &tangent_schedule.order, &tangent_schedule.groups);
        builder.assert_no_block_straddles(primal_end);
        builder.assert_block_slots_private(total_hw);
        let split = SplitEvalInfo {
            primal_end,
            primal_buffer_size: primal_hw,
        };
        (builder.finish(root, total_hw), split)
    }

    /// The finished tape, whose result is `root`'s slot.
    fn finish(self, root: NodeId, buffer_size: usize) -> TypedIr {
        TypedIr {
            root_slot: self.slot_for(root),
            instructions: self.instructions,
            buffer_size,
            consts: self.consts,
        }
    }

    /// The buffer window allocated for a node's value.
    fn slot_for(&self, id: NodeId) -> Slot {
        self.slots[id.index()]
    }

    /// Emit `order`, wrapping each annotated group range in a `Dispatch`.
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

    /// Emit a `Dispatch` followed by one contiguous block per branch.
    ///
    /// Block lengths count instructions, not nodes, since `Node::SparseMatrix` emits none.
    fn emit_dispatch(&mut self, arena: &Arena, group: &RegionGroup, nodes: &[NodeId]) {
        let Node::Conditional { selector, .. } = arena.get(group.cond) else {
            unreachable!("a region group is always anchored on a Conditional")
        };
        let selector_slot = self.slot_for(*selector).offset;
        let dispatch_at = self.instructions.len();
        let blocks_idx = BlockSpanIdx::of_next(&self.consts.branch_blocks);
        // Reserve the table so the blocks can be emitted before it is filled.
        self.consts.branch_blocks.resize(
            blocks_idx.index() + group.branch_lens.len(),
            BlockSpan::new(0, 0),
        );
        self.instructions.push(Instruction::Dispatch {
            selector: selector_slot,
            blocks_idx,
            blocks_len: EntryCount::from_usize(group.branch_lens.len()),
            span_len: InstructionCount::new(1),
        });

        let mut taken = 0;
        for (i, &n_nodes) in group.branch_lens.iter().enumerate() {
            let start = self.instructions.len();
            for &node_id in &nodes[taken..taken + n_nodes] {
                self.emit_node(arena, node_id);
            }
            taken += n_nodes;
            let len = self.instructions.len() - start;
            self.consts.branch_blocks[blocks_idx.index() + i] =
                BlockSpan::from_usize(start - dispatch_at, len);
        }
        assert_eq!(taken, nodes.len(), "group range and branch_lens disagree");
        let end = self.instructions.len();
        let Instruction::Dispatch { span_len, .. } = &mut self.instructions[dispatch_at] else {
            unreachable!("the Dispatch was pushed above")
        };
        *span_len = InstructionCount::from_usize(end - dispatch_at);
    }

    /// Assert every `Dispatch` block lies wholly on one side of `split`.
    ///
    /// A straddling block would run unconditionally in the tangent section.
    fn assert_no_block_straddles(&self, split: usize) {
        for (pc, instr) in self.instructions.iter().enumerate() {
            if let Instruction::Dispatch {
                blocks_idx,
                blocks_len,
                ..
            } = *instr
            {
                for b in 0..blocks_len.index() {
                    let block = self.consts.branch_blocks[blocks_idx.index() + b].bounds(pc);
                    assert!(
                        split <= block.start || split >= block.end,
                        "split point {split} straddles block [{}, {})",
                        block.start,
                        block.end
                    );
                }
            }
        }
    }

    /// Which instructions sit inside a branch block.
    ///
    /// Panics on a nested `Dispatch`, as evaluators skip only one level of block.
    fn block_instruction_mask(&self) -> Vec<bool> {
        let mut in_block = vec![false; self.instructions.len()];
        for (pc, instr) in self.instructions.iter().enumerate() {
            if let Instruction::Dispatch {
                blocks_idx,
                blocks_len,
                ..
            } = *instr
            {
                for b in 0..blocks_len.index() {
                    let block = self.consts.branch_blocks[blocks_idx.index() + b].bounds(pc);
                    for at in block {
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

    /// Assert no instruction outside a branch block reads a value a block defined.
    ///
    /// That would silently read a stale recycled slot, so this runs in release too.
    fn assert_block_slots_private(&self, buffer_size: usize) {
        if self.consts.branch_blocks.is_empty() {
            return;
        }
        let in_block = self.block_instruction_mask();
        let mut last_writer: Vec<Option<usize>> = vec![None; buffer_size];

        for (pc, instr) in self.instructions.iter().enumerate() {
            if !in_block[pc] {
                if let Instruction::Dispatch { selector, .. } = *instr {
                    assert!(
                        last_writer[selector.index()].is_some(),
                        "Dispatch at {pc} selects on buffer element {}, which no earlier \
                         instruction defines",
                        selector.index()
                    );
                }
                for_each_src_extent(instr, &self.consts, |src| {
                    // A `Conditional`'s branch slots are the one legitimate outside read.
                    if src.is_branch_slot {
                        return;
                    }
                    let start = src.slot.offset_usize();
                    let end = start + src.slot.len_usize();
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
                });
            }
            if let Some(dst) = instruction_dst_extent(instr, &self.consts) {
                for writer in &mut last_writer[dst.bounds()] {
                    *writer = Some(pc);
                }
            }
        }
    }

    fn emit_node(&mut self, arena: &Arena, node_id: NodeId) {
        let Slot {
            offset: dst,
            len: out_len,
        } = self.slot_for(node_id);

        let instr = match &arena[node_id] {
            Node::Scalar(v) => Instruction::LoadScalar { value: *v, dst },
            Node::Time => Instruction::LoadTime { dst },
            Node::Array(arr) => {
                let data_idx = self.consts.add_array(&arr.data);
                Instruction::LoadArray {
                    data_idx,
                    len: Width::from_usize(arr.data.len()),
                    dst,
                }
            },
            Node::ZeroVector { len } => Instruction::FillZero {
                dst,
                len: Width::from_usize(*len),
            },
            Node::StateVector { start, end } => Instruction::LoadStateVector {
                start: StateIdx::from_usize(*start),
                width: state_width(*start, *end),
                dst,
            },
            Node::StateVectorDot { start, end } => Instruction::LoadStateVectorDot {
                start: StateIdx::from_usize(*start),
                width: state_width(*start, *end),
                dst,
            },
            Node::InputParameter { offset, width, .. } => Instruction::LoadInputParameter {
                offset: InputOffset::from_usize(*offset),
                width: Width::from_usize(*width),
                dst,
            },
            Node::TangentStateVector { start, end } => Instruction::LoadTangentState {
                start: StateIdx::from_usize(*start),
                width: state_width(*start, *end),
                dst,
            },
            Node::TangentParameter { index } => Instruction::LoadTangentParameter {
                index: ParamIdx::from_usize(*index),
                dst,
            },
            Node::SparseMatrix(_) => return, // Not emitted as instruction

            Node::Add(a, b) => self.emit_binary(BinaryOp::Add, *a, *b, dst, out_len),
            Node::Sub(a, b) => self.emit_binary(BinaryOp::Sub, *a, *b, dst, out_len),
            Node::Mul(a, b) => self.emit_binary(BinaryOp::Mul, *a, *b, dst, out_len),
            Node::Div(a, b) => self.emit_binary(BinaryOp::Div, *a, *b, dst, out_len),
            Node::Pow(a, b) => self.emit_binary(BinaryOp::Pow, *a, *b, dst, out_len),
            Node::Minimum(a, b) => self.emit_binary(BinaryOp::Minimum, *a, *b, dst, out_len),
            Node::Maximum(a, b) => self.emit_binary(BinaryOp::Maximum, *a, *b, dst, out_len),
            Node::Modulo(a, b) => self.emit_binary(BinaryOp::Modulo, *a, *b, dst, out_len),
            Node::Hypot(a, b) => self.emit_binary(BinaryOp::Hypot, *a, *b, dst, out_len),
            Node::EqualHeaviside(a, b) => {
                self.emit_binary(BinaryOp::EqualHeaviside, *a, *b, dst, out_len)
            },
            Node::NotEqualHeaviside(a, b) => {
                self.emit_binary(BinaryOp::NotEqualHeaviside, *a, *b, dst, out_len)
            },
            Node::Equality(a, b) => self.emit_binary(BinaryOp::Equality, *a, *b, dst, out_len),

            Node::Neg(a) => self.emit_unary(UnaryOp::Neg, *a, dst, out_len),
            Node::Abs(a) => self.emit_unary(UnaryOp::Abs, *a, dst, out_len),
            Node::Sqrt(a) => self.emit_unary(UnaryOp::Sqrt, *a, dst, out_len),
            Node::Exp(a) => self.emit_unary(UnaryOp::Exp, *a, dst, out_len),
            Node::Log(a) => self.emit_unary(UnaryOp::Log, *a, dst, out_len),
            Node::Sin(a) => self.emit_unary(UnaryOp::Sin, *a, dst, out_len),
            Node::Cos(a) => self.emit_unary(UnaryOp::Cos, *a, dst, out_len),
            Node::Tanh(a) => self.emit_unary(UnaryOp::Tanh, *a, dst, out_len),
            Node::Sinh(a) => self.emit_unary(UnaryOp::Sinh, *a, dst, out_len),
            Node::Cosh(a) => self.emit_unary(UnaryOp::Cosh, *a, dst, out_len),
            Node::Arcsinh(a) => self.emit_unary(UnaryOp::Arcsinh, *a, dst, out_len),
            Node::Arctan(a) => self.emit_unary(UnaryOp::Arctan, *a, dst, out_len),
            Node::Erf(a) => self.emit_unary(UnaryOp::Erf, *a, dst, out_len),
            Node::Sign(a) => self.emit_unary(UnaryOp::Sign, *a, dst, out_len),
            Node::Floor(a) => self.emit_unary(UnaryOp::Floor, *a, dst, out_len),
            Node::Ceiling(a) => self.emit_unary(UnaryOp::Ceiling, *a, dst, out_len),

            Node::MaxReduce(a) => {
                let src = self.slot_for(*a);
                Instruction::MaxReduce {
                    src: src.offset,
                    src_len: src.len,
                    dst,
                }
            },
            Node::MinReduce(a) => {
                let src = self.slot_for(*a);
                Instruction::MinReduce {
                    src: src.offset,
                    src_len: src.len,
                    dst,
                }
            },
            Node::ReduceArgSelect {
                basis,
                picker,
                is_max,
            } => {
                // `infer_sizes` checked the scan's `picker[0..len]` matches the basis width.
                let basis_slot = self.slot_for(*basis);
                Instruction::ReduceArgSelect {
                    basis_src: basis_slot.offset,
                    picker_src: self.slot_for(*picker).offset,
                    len: basis_slot.len,
                    is_max: *is_max,
                    dst,
                }
            },

            Node::Index { child, start, .. } => Instruction::Index {
                src: self.slot_for(*child).offset,
                start: SlotOffset::from_usize(*start),
                dst,
                len: out_len,
            },
            Node::Concat(children) => {
                let sources_idx = SlotTableIdx::of_next(&self.consts.concat_sources);
                for c in children {
                    let slot = self.slot_for(*c);
                    self.consts.concat_sources.push(slot);
                }
                Instruction::Concat {
                    sources_idx,
                    sources_len: EntryCount::from_usize(children.len()),
                    dst,
                }
            },

            Node::MatMul(a, b) => match &arena[*a] {
                Node::SparseMatrix(csr) => {
                    let csr_idx = self.consts.add_csr(csr);
                    Instruction::MatMul {
                        csr_idx,
                        vec_src: self.slot_for(*b).offset,
                        dst,
                    }
                },
                Node::Array(arr) => Instruction::DenseMatMul {
                    mat_src: self.slot_for(*a).offset,
                    rows: Rows::from_usize(arr.shape.rows),
                    cols: Cols::from_usize(arr.shape.cols),
                    vec_src: self.slot_for(*b).offset,
                    dst,
                },
                _ => panic!("MatMul requires a constant matrix on the left"),
            },

            Node::Interpolant1DLinear { data, child } => {
                let interp_idx = self.consts.add_interpolant(data);
                Instruction::Interp1DLinear {
                    interp_idx,
                    src: self.slot_for(*child).offset,
                    dst,
                    len: out_len,
                }
            },
            Node::Interpolant1DLinearDeriv { data, child } => {
                let interp_idx = self.consts.add_interpolant(data);
                Instruction::Interp1DLinearDeriv {
                    interp_idx,
                    src: self.slot_for(*child).offset,
                    dst,
                    len: out_len,
                }
            },
            Node::Interpolant1DCubic { data, child } => {
                let interp_idx = self.consts.add_cubic_interpolant(data);
                Instruction::Interp1DCubic {
                    interp_idx,
                    src: self.slot_for(*child).offset,
                    dst,
                    len: out_len,
                }
            },
            Node::Interpolant1DCubicDeriv { data, child } => {
                let interp_idx = self.consts.add_cubic_interpolant(data);
                Instruction::Interp1DCubicDeriv {
                    interp_idx,
                    src: self.slot_for(*child).offset,
                    dst,
                    len: out_len,
                }
            },
            Node::InterpolantNd { data, children } => {
                let interp_idx = self.consts.add_nd_interpolant(data);
                let sources_idx = SlotTableIdx::of_next(&self.consts.interp_nd_sources);
                for c in children {
                    let slot = self.slot_for(*c);
                    self.consts.interp_nd_sources.push(slot);
                }
                Instruction::InterpNd {
                    interp_idx,
                    sources_idx,
                    dst,
                    len: out_len,
                }
            },
            Node::InterpolantNdPartial {
                data,
                children,
                axis,
            } => {
                let interp_idx = self.consts.add_nd_interpolant(data);
                let sources_idx = SlotTableIdx::of_next(&self.consts.interp_nd_sources);
                for c in children {
                    let slot = self.slot_for(*c);
                    self.consts.interp_nd_sources.push(slot);
                }
                Instruction::InterpNdPartial {
                    interp_idx,
                    sources_idx,
                    axis: Axis::new(*axis),
                    dst,
                    len: out_len,
                }
            },

            Node::Conditional { selector, branches } => {
                let branches_idx = SlotTableIdx::of_next(&self.consts.branch_offsets);
                for b in branches {
                    let slot = self.slot_for(*b);
                    self.consts.branch_offsets.push(slot);
                }
                Instruction::Conditional {
                    selector: self.slot_for(*selector).offset,
                    branches_idx,
                    branches_len: EntryCount::from_usize(branches.len()),
                    dst,
                    out_len,
                }
            },
        };

        self.instructions.push(instr);
    }

    fn emit_binary(
        &self,
        op: BinaryOp,
        a: NodeId,
        b: NodeId,
        dst: BufSlot,
        len: Width,
    ) -> Instruction {
        let a_slot = self.slot_for(a);
        let b_slot = self.slot_for(b);
        Instruction::Binary {
            op,
            a: a_slot.offset,
            b: b_slot.offset,
            dst,
            len,
            kind: BroadcastKind::from_lens(a_slot.len_usize(), b_slot.len_usize()),
        }
    }

    fn emit_unary(&self, op: UnaryOp, a: NodeId, dst: BufSlot, len: Width) -> Instruction {
        let src = self.slot_for(a).offset;
        Instruction::Unary { op, src, dst, len }
    }
}

#[cfg(test)]
impl IRBuilder {
    /// A tape where an instruction outside the `Dispatch` block reads that block's slot.
    fn test_tape_reading_across_block_boundary() -> Self {
        let mut consts = ConstPool::new();
        consts.branch_blocks.push(BlockSpan::new(1, 1));

        let instructions = vec![
            Instruction::LoadScalar {
                value: 1.0,
                dst: BufSlot::new(0),
            },
            Instruction::Dispatch {
                selector: BufSlot::new(0),
                blocks_idx: BlockSpanIdx::new(0),
                blocks_len: EntryCount::new(1),
                span_len: InstructionCount::new(2),
            },
            Instruction::LoadScalar {
                value: 1.0,
                dst: BufSlot::new(5),
            },
            Instruction::Unary {
                op: UnaryOp::Neg,
                src: BufSlot::new(5), // outside the block, illegally reads the block's slot
                dst: BufSlot::new(6),
                len: Width::new(1),
            },
        ];

        Self {
            instructions,
            consts,
            ..Self::new(Vec::new(), 0)
        }
    }

    /// A tape whose `Dispatch` selects on a slot no instruction writes.
    fn test_tape_dispatching_on_an_undefined_selector() -> Self {
        let mut consts = ConstPool::new();
        consts.branch_blocks.push(BlockSpan::new(1, 1));

        let instructions = vec![
            Instruction::Dispatch {
                selector: BufSlot::new(0), // never written
                blocks_idx: BlockSpanIdx::new(0),
                blocks_len: EntryCount::new(1),
                span_len: InstructionCount::new(2),
            },
            Instruction::LoadScalar {
                value: 1.0,
                dst: BufSlot::new(5),
            },
        ];

        Self {
            instructions,
            consts,
            ..Self::new(Vec::new(), 0)
        }
    }
}

/// Width of the state slice `[start, end)`; an inverted extent here is a lowering bug.
fn state_width(start: usize, end: usize) -> Width {
    Width::from_usize(
        end.checked_sub(start)
            .expect("inverted state extent reached lowering"),
    )
}

/// Why [`validate`] rejects a graph.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GraphError {
    /// A construct lowering cannot handle: a non-constant `MatMul` left operand.
    #[error("{0}")]
    Unsupported(String),
    /// A width relationship the evaluator relies on is broken.
    #[error("{0}")]
    Invalid(String),
}

/// Check a caller-built graph at `root` can be lowered, in one walk of its topological order.
///
/// The earliest unsupported node wins over any invalid one; otherwise the earliest invalid.
/// A node kind only the rewrite passes build is invalid here: the reads a forged tangent
/// load makes are not part of the extents callers check their buffers against.
pub fn validate(arena: &Arena, root: NodeId) -> Result<(), GraphError> {
    let mut sizes = vec![0_usize; arena.len()];
    let mut invalid = None;
    for node_id in arena.topological_order(root) {
        let node = &arena[node_id];
        if let Node::MatMul(a, _) = node
            && !matches!(&arena[*a], Node::SparseMatrix(_) | Node::Array(_))
        {
            return Err(GraphError::Unsupported(format!(
                "MatMul left operand must be a constant matrix, got {:?}",
                arena[*a]
            )));
        }
        // Widths are meaningless past an invalid node, but a later unsupported one still wins.
        if invalid.is_none() {
            let width = caller_buildable(node_id, node)
                .and_then(|()| checked_width(arena, node_id, node, &sizes));
            match width {
                Ok(width) => sizes[node_id.index()] = width,
                Err(message) => invalid = Some(message),
            }
        }
    }
    invalid.map_or(Ok(()), |message| Err(GraphError::Invalid(message)))
}

/// Refuse a node kind only the rewrite passes build.
fn caller_buildable(node_id: NodeId, node: &Node) -> Result<(), String> {
    let kind = match node {
        Node::ZeroVector { .. } => "ZeroVector",
        Node::ReduceArgSelect { .. } => "ReduceArgSelect",
        Node::TangentStateVector { .. } => "TangentStateVector",
        Node::TangentParameter { .. } => "TangentParameter",
        Node::Interpolant1DLinearDeriv { .. } => "Interpolant1DLinearDeriv",
        Node::Interpolant1DCubicDeriv { .. } => "Interpolant1DCubicDeriv",
        Node::InterpolantNdPartial { .. } => "InterpolantNdPartial",
        _ => return Ok(()),
    };
    Err(format!(
        "node {node_id:?} is an internal {kind} node, which no caller builds"
    ))
}

fn infer_sizes_checked(arena: &Arena, eval_order: &[NodeId]) -> Result<Vec<usize>, String> {
    let mut sizes = vec![0usize; arena.len()];
    for &node_id in eval_order {
        sizes[node_id.index()] = checked_width(arena, node_id, &arena[node_id], &sizes)?;
    }
    Ok(sizes)
}

/// `node`'s width, once its relationships hold and it fits the tape's `u32` indices.
fn checked_width(
    arena: &Arena,
    node_id: NodeId,
    node: &Node,
    sizes: &[usize],
) -> Result<usize, String> {
    check_widths(arena, node_id, node, sizes)?;
    let width = node_width(arena, node, |child| sizes[child.index()]);
    if u32::try_from(width).is_err() {
        return Err(format!(
            "node {node_id:?} has width {width}, past the tape's u32 index range"
        ));
    }
    Ok(width)
}

/// The first width relationship `node` breaks, given its children's `sizes`.
fn check_widths(
    arena: &Arena,
    node_id: NodeId,
    node: &Node,
    sizes: &[usize],
) -> Result<(), String> {
    let width = |id: &NodeId| sizes[id.index()];
    match node {
        Node::StateVector { start, end }
        | Node::StateVectorDot { start, end }
        | Node::TangentStateVector { start, end }
        | Node::Index { start, end, .. }
            if start > end =>
        {
            Err(format!(
                "node {node_id:?} has an inverted extent: start={start}, end={end}"
            ))
        },
        Node::StateVector { end, .. }
        | Node::StateVectorDot { end, .. }
        | Node::TangentStateVector { end, .. }
            if u32::try_from(*end).is_err() =>
        {
            Err(format!(
                "node {node_id:?} reads states up to {end}, past the tape's u32 index range"
            ))
        },
        Node::InputParameter { offset, width, .. }
            if offset
                .checked_add(*width)
                .is_none_or(|end| u32::try_from(end).is_err()) =>
        {
            Err(format!(
                "node {node_id:?} reads inputs from offset {offset} with width {width}, past \
                 the tape's u32 index range"
            ))
        },
        Node::Index { child, end, .. } if *end > width(child) => Err(format!(
            "Index node {node_id:?} ends at {end}, beyond child width {}",
            width(child)
        )),
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
        | Node::Equality(a, b)
            if !broadcast_widths_compatible(width(a), width(b)) =>
        {
            Err(format!(
                "binary node {node_id:?} has incompatible operand widths {} and {}; widths \
                 must match or one operand must be scalar",
                width(a),
                width(b)
            ))
        },
        Node::MatMul(a, b) => {
            let cols = match &arena[*a] {
                Node::SparseMatrix(csr) => csr.shape.cols,
                Node::Array(arr) => arr.shape.cols,
                _ => return Err("MatMul requires a constant matrix on the left".to_string()),
            };
            if width(b) == cols {
                Ok(())
            } else {
                Err(format!(
                    "MatMul node {node_id:?} has {cols} columns but its vector operand has \
                     width {}",
                    width(b)
                ))
            }
        },
        Node::InterpolantNd { data, children }
        | Node::InterpolantNdPartial { data, children, .. } => {
            let axes = data.breakpoints.len();
            if children.len() != axes {
                return Err(format!(
                    "N-D interpolant node {node_id:?} has {} children for {axes} axes",
                    children.len()
                ));
            }
            let out_len = children.iter().map(width).max().unwrap_or(1);
            if let Some(child) = children
                .iter()
                .find(|c| width(c) != 1 && width(c) != out_len)
            {
                return Err(format!(
                    "N-D interpolant node {node_id:?} has child widths that cannot broadcast: \
                     {} and {out_len}",
                    width(child)
                ));
            }
            match node {
                Node::InterpolantNdPartial { axis, .. } if *axis as usize >= axes => Err(format!(
                    "N-D interpolant partial node {node_id:?} has axis {axis} for {axes} axes"
                )),
                _ => Ok(()),
            }
        },
        Node::Conditional { selector, branches } => {
            if width(selector) != 1 {
                return Err(format!(
                    "Conditional node {node_id:?} requires a scalar selector, got width {}",
                    width(selector)
                ));
            }
            let out_len = branches.first().map_or(1, width);
            branches
                .iter()
                .find(|branch| width(branch) != out_len)
                .map_or(Ok(()), |branch| {
                    Err(format!(
                        "Conditional node {node_id:?} has branch widths {out_len} and {}",
                        width(branch)
                    ))
                })
        },
        Node::ReduceArgSelect { basis, picker, .. } => {
            if width(basis) == 0 {
                Err(format!(
                    "ReduceArgSelect node {node_id:?} requires non-empty operands"
                ))
            } else if width(basis) != width(picker) {
                Err(format!(
                    "ReduceArgSelect node {node_id:?} has basis width {} and picker width {}",
                    width(basis),
                    width(picker)
                ))
            } else {
                Ok(())
            }
        },
        _ => Ok(()),
    }
}

/// `node`'s output width from its children's: the one rule lowering and every rewrite share.
///
/// Assumes the relationships [`validate`] checks, e.g. a non-inverted extent.
pub(crate) fn node_width(
    arena: &Arena,
    node: &Node,
    mut width_of: impl FnMut(NodeId) -> usize,
) -> usize {
    match node {
        Node::Scalar(_)
        | Node::Time
        | Node::TangentParameter { .. }
        | Node::MaxReduce(_)
        | Node::MinReduce(_)
        | Node::ReduceArgSelect { .. } => 1,
        Node::InputParameter { width, .. } => *width,
        Node::Array(arr) => arr.data.len(),
        Node::ZeroVector { len } => *len,
        Node::SparseMatrix(_) => 0,
        Node::StateVector { start, end }
        | Node::StateVectorDot { start, end }
        | Node::TangentStateVector { start, end }
        | Node::Index { start, end, .. } => end - start,
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
        | Node::Equality(a, b) => width_of(*a).max(width_of(*b)),
        Node::MatMul(a, b) => match arena.get(*a) {
            Node::SparseMatrix(csr) => csr.shape.rows,
            Node::Array(arr) => arr.shape.rows,
            _ => width_of(*b),
        },
        Node::Concat(children) => children.iter().map(|&c| width_of(c)).sum(),
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
        | Node::Ceiling(a)
        | Node::Interpolant1DLinear { child: a, .. }
        | Node::Interpolant1DLinearDeriv { child: a, .. }
        | Node::Interpolant1DCubic { child: a, .. }
        | Node::Interpolant1DCubicDeriv { child: a, .. } => width_of(*a),
        Node::InterpolantNd { children, .. }
        | Node::InterpolantNdPartial { children, .. }
        | Node::Conditional {
            branches: children, ..
        } => children.iter().map(|&c| width_of(c)).max().unwrap_or(1),
    }
}

/// Infer the output size of each node in topological evaluation order.
pub fn infer_sizes(arena: &Arena, eval_order: &[NodeId]) -> Vec<usize> {
    infer_sizes_checked(arena, eval_order)
        .unwrap_or_else(|message| panic!("invalid expression graph: {message}"))
}

/// One slot range an instruction reads.
#[derive(Clone, Copy, Debug)]
struct SrcExtent {
    slot: Slot,
    /// A `Conditional`'s branch slot, which may legitimately read block-owned values.
    is_branch_slot: bool,
}

impl SrcExtent {
    const fn read(offset: BufSlot, len: Width) -> Self {
        Self {
            slot: Slot::new(offset, len),
            is_branch_slot: false,
        }
    }

    const fn of(slot: Slot) -> Self {
        Self {
            slot,
            is_branch_slot: false,
        }
    }

    const fn branch_slot(offset: BufSlot, len: Width) -> Self {
        Self {
            slot: Slot::new(offset, len),
            is_branch_slot: true,
        }
    }
}

/// The exact buffer window an instruction writes, or `None` for `Dispatch`.
fn instruction_dst_extent(instr: &Instruction, consts: &ConstPool) -> Option<Slot> {
    let extent = match *instr {
        Instruction::LoadScalar { dst, .. }
        | Instruction::LoadTime { dst }
        | Instruction::LoadTangentParameter { dst, .. }
        | Instruction::MaxReduce { dst, .. }
        | Instruction::MinReduce { dst, .. }
        | Instruction::ReduceArgSelect { dst, .. } => Slot::new(dst, Width::new(1)),
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
        | Instruction::InterpNdPartial { dst, len, .. } => Slot::new(dst, len),
        Instruction::LoadStateVector { width, dst, .. }
        | Instruction::LoadStateVectorDot { width, dst, .. }
        | Instruction::LoadTangentState { width, dst, .. }
        | Instruction::LoadInputParameter { width, dst, .. } => Slot::new(dst, width),
        Instruction::Concat {
            sources_idx,
            sources_len,
            dst,
        } => {
            // Sum in `usize`: a `u32` sum of `u32` widths wraps in release.
            let width: usize = (0..sources_len.index())
                .map(|i| consts.concat_sources[sources_idx.index() + i].len.index())
                .sum();
            Slot::new(dst, Width::from_usize(width))
        },
        Instruction::MatMul { csr_idx, dst, .. } => Slot::new(
            dst,
            Width::from_usize(consts.csr_data[csr_idx.index()].shape.rows),
        ),
        Instruction::DenseMatMul { rows, dst, .. } => Slot::new(dst, rows.to_width()),
        Instruction::Conditional { dst, out_len, .. } => Slot::new(dst, out_len),
        Instruction::Dispatch { .. } => return None,
    };
    Some(extent)
}

/// Hand each slot range `instr` reads to `visit`, with exact widths.
fn for_each_src_extent(instr: &Instruction, consts: &ConstPool, mut visit: impl FnMut(SrcExtent)) {
    match *instr {
        Instruction::LoadScalar { .. }
        | Instruction::LoadTime { .. }
        | Instruction::LoadArray { .. }
        | Instruction::FillZero { .. }
        | Instruction::LoadStateVector { .. }
        | Instruction::LoadStateVectorDot { .. }
        | Instruction::LoadInputParameter { .. }
        | Instruction::LoadTangentState { .. }
        | Instruction::LoadTangentParameter { .. } => {},
        Instruction::Binary {
            a, b, len, kind, ..
        } => {
            let one = Width::new(1);
            let (a_len, b_len) = match kind {
                BroadcastKind::ScalarScalar => (one, one),
                BroadcastKind::ScalarVector => (one, len),
                BroadcastKind::VectorScalar => (len, one),
                BroadcastKind::VectorVector => (len, len),
            };
            visit(SrcExtent::read(a, a_len));
            visit(SrcExtent::read(b, b_len));
        },
        Instruction::Unary { src, len, .. }
        | Instruction::Interp1DLinear { src, len, .. }
        | Instruction::Interp1DLinearDeriv { src, len, .. }
        | Instruction::Interp1DCubic { src, len, .. }
        | Instruction::Interp1DCubicDeriv { src, len, .. } => visit(SrcExtent::read(src, len)),
        Instruction::MaxReduce { src, src_len, .. }
        | Instruction::MinReduce { src, src_len, .. } => visit(SrcExtent::read(src, src_len)),
        Instruction::ReduceArgSelect {
            basis_src,
            picker_src,
            len,
            ..
        } => {
            visit(SrcExtent::read(basis_src, len));
            visit(SrcExtent::read(picker_src, len));
        },
        Instruction::Index {
            src, start, len, ..
        } => visit(SrcExtent::read(
            BufSlot::from_usize(src.offset_by(start)),
            len,
        )),
        Instruction::Concat {
            sources_idx,
            sources_len,
            ..
        } => {
            for &slot in &consts.concat_sources[sources_idx.index()..][..sources_len.index()] {
                visit(SrcExtent::of(slot));
            }
        },
        Instruction::MatMul {
            csr_idx, vec_src, ..
        } => visit(SrcExtent::read(
            vec_src,
            Width::from_usize(consts.csr_data[csr_idx.index()].shape.cols),
        )),
        Instruction::DenseMatMul {
            mat_src,
            rows,
            cols,
            vec_src,
            ..
        } => {
            visit(SrcExtent::read(mat_src, rows.times(cols)));
            visit(SrcExtent::read(vec_src, cols.to_width()));
        },
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
            let ndim = consts.nd_interpolants[interp_idx.index()].breakpoints.len();
            for &slot in &consts.interp_nd_sources[sources_idx.index()..][..ndim] {
                visit(SrcExtent::of(slot));
            }
        },
        Instruction::Conditional {
            selector,
            branches_idx,
            branches_len,
            out_len,
            ..
        } => {
            visit(SrcExtent::read(selector, Width::new(1)));
            for slot in &consts.branch_offsets[branches_idx.index()..][..branches_len.index()] {
                visit(SrcExtent::branch_slot(slot.offset, out_len));
            }
        },
        Instruction::Dispatch { selector, .. } => visit(SrcExtent::read(selector, Width::new(1))),
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

/// The last position in `order` reading each node, with every node in `roots` pinned.
fn last_use_positions(arena: &Arena, order: &[NodeId], roots: &[NodeId]) -> Vec<usize> {
    let mut last_use = vec![0_usize; arena.len()];
    for (pos, &node_id) in order.iter().enumerate() {
        arena
            .get(node_id)
            .for_each_child(|c| last_use[c.index()] = pos);
    }
    for &r in roots {
        last_use[r.index()] = usize::MAX;
    }
    last_use
}

/// Assign buffer slots, computing last-use positions from the evaluation order.
fn assign_slots(
    arena: &Arena,
    sizes: &[usize],
    eval_order: &[NodeId],
    roots: &[NodeId],
) -> (Vec<(usize, usize)>, usize) {
    let last_use = last_use_positions(arena, eval_order, roots);
    assign_slots_configured(sizes, eval_order, &last_use, 0)
}

/// Slot allocation with no reuse, so the primal scratch preserves every intermediate.
fn assign_slots_pinned(sizes: &[usize], eval_order: &[NodeId]) -> (Vec<(usize, usize)>, usize) {
    let last_use = vec![usize::MAX; sizes.len()];
    assign_slots_configured(sizes, eval_order, &last_use, 0)
}

/// Sweep-line buffer slot allocator, reusing freed regions.
///
/// `last_use[id]` is the last position reading the node (`usize::MAX` pins it), and
/// `initial_high_water` offsets every allocation.
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
    use crate::node::{ArrayData, Shape};

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

        let ir = TypedIr::from_dag(DagRef::new(&arena, expr));

        assert_eq!(ir.output_len(), 3);
        assert_eq!(ir.instructions().len(), 3); // Scalar, StateVector, Mul
    }

    #[test]
    fn test_slot_struct() {
        let slot = Slot::from_usize(10, 5);
        assert_eq!(slot.offset, BufSlot::new(10));
        assert_eq!(slot.len, Width::new(5));
        assert_eq!(slot.offset_usize(), 10);
        assert_eq!(slot.len_usize(), 5);
        assert!(!slot.is_scalar());

        let scalar_slot = Slot::from_usize(0, 1);
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
    fn test_ir_nested_expression() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let two = arena.alloc(Node::Scalar(2.0));
        let one = arena.alloc(Node::Scalar(1.0));
        let mul = arena.alloc(Node::Mul(x, two));
        let add = arena.alloc(Node::Add(mul, one));
        let sin = arena.alloc(Node::Sin(add));

        let ir = TypedIr::from_dag(DagRef::new(&arena, sin));

        assert_eq!(ir.output_len(), 3);
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

        let ir = TypedIr::from_dag(DagRef::new(&arena, concat));

        assert_eq!(ir.output_len(), 4);
        assert_eq!(ir.consts().concat_sources.len(), 3);
    }

    /// The message [`validate`] rejects `root` with as invalid.
    fn invalid_message(arena: &Arena, root: NodeId) -> String {
        match validate(arena, root) {
            Err(GraphError::Invalid(message)) => message,
            other => panic!("expected an invalid graph, got {other:?}"),
        }
    }

    #[test]
    fn validate_flags_a_non_constant_matmul_lhs_as_unsupported() {
        let mut arena = Arena::new();
        let s = arena.alloc(Node::Scalar(2.0));
        let v = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0],
            shape: Shape::vector(1),
        })));
        let mm = arena.alloc(Node::MatMul(s, v));
        assert_eq!(
            validate(&arena, mm),
            Err(GraphError::Unsupported(
                "MatMul left operand must be a constant matrix, got Scalar(2.0)".to_string()
            ))
        );
        assert_eq!(validate(&arena, v), Ok(()));
    }

    #[test]
    fn validate_reports_unsupported_over_an_earlier_invalid_node() {
        // The inverted extent comes first in topological order; its width is never computed.
        let mut arena = Arena::new();
        let inverted = arena.alloc(Node::StateVector { start: 3, end: 1 });
        let s = arena.alloc(Node::Scalar(2.0));
        let v = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let mm = arena.alloc(Node::MatMul(s, v));
        let root = arena.alloc(Node::Concat(vec![inverted, mm]));
        assert!(matches!(
            validate(&arena, root),
            Err(GraphError::Unsupported(message)) if message.contains("constant matrix")
        ));

        let only_invalid = arena.alloc(Node::Concat(vec![inverted, v]));
        assert!(invalid_message(&arena, only_invalid).contains("inverted extent"));
    }

    #[test]
    fn test_symbolic_jacobian_width_mismatch_rejected_before_eval_overlap() {
        let mut arena = Arena::new();
        let short = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let long = arena.alloc(Node::StateVector { start: 2, end: 5 });
        let add = arena.alloc(Node::Add(short, long));
        let scalar_entry = arena.alloc(Node::Scalar(1.0));
        let symbolic_jacobian = arena.alloc(Node::Concat(vec![add, scalar_entry]));

        // Unrejected, the width-2 operand's read window would overlap `dst`.
        let message = invalid_message(&arena, symbolic_jacobian);
        assert!(message.contains("incompatible operand widths 2 and 3"));
    }

    #[test]
    fn validate_flags_other_evaluator_width_hazards() {
        let mut matmul_arena = Arena::new();
        let matrix = matmul_arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0; 6],
            shape: Shape::matrix(2, 3),
        })));
        let short = matmul_arena.alloc(Node::StateVector { start: 0, end: 2 });
        let matmul = matmul_arena.alloc(Node::MatMul(matrix, short));
        assert!(
            invalid_message(&matmul_arena, matmul)
                .contains("has 3 columns but its vector operand has width 2")
        );

        let mut index_arena = Arena::new();
        let child = index_arena.alloc(Node::StateVector { start: 0, end: 2 });
        let index = index_arena.alloc(Node::Index {
            child,
            start: 1,
            end: 3,
        });
        assert!(invalid_message(&index_arena, index).contains("beyond child width 2"));

        let mut conditional_arena = Arena::new();
        let selector = conditional_arena.alloc(Node::Scalar(1.0));
        let short = conditional_arena.alloc(Node::StateVector { start: 0, end: 2 });
        let long = conditional_arena.alloc(Node::StateVector { start: 2, end: 5 });
        let conditional = conditional_arena.alloc(Node::Conditional {
            selector,
            branches: vec![short, long],
        });
        assert!(invalid_message(&conditional_arena, conditional).contains("branch widths 2 and 3"));

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
            invalid_message(&interpolant_arena, interpolant).contains("cannot broadcast: 2 and 3")
        );

        // Only the rewrite passes build these, so lowering's own width check is what guards them.
        let lowering_message = |arena: &Arena, root| {
            infer_sizes_checked(arena, &arena.topological_order(root))
                .expect_err("lowering must refuse the graph")
        };
        let mut reduce_arena = Arena::new();
        let basis = reduce_arena.alloc(Node::StateVector { start: 0, end: 2 });
        let picker = reduce_arena.alloc(Node::StateVector { start: 2, end: 5 });
        let reduce = reduce_arena.alloc(Node::ReduceArgSelect {
            basis,
            picker,
            is_max: true,
        });
        assert!(
            lowering_message(&reduce_arena, reduce).contains("basis width 2 and picker width 3")
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
            lowering_message(&empty_reduce_arena, reduce).contains("requires non-empty operands")
        );
    }

    #[test]
    fn validate_refuses_extents_and_widths_past_the_tapes_u32_indices() {
        // Unrefused, the first three lower into a panic narrowing an index to u32.
        let mut arena = Arena::new();
        let late = arena.alloc(Node::StateVector {
            start: 1 << 32,
            end: (1 << 32) + 1,
        });
        assert!(invalid_message(&arena, late).contains("reads states up to 4294967297"));
        let late_dot = arena.alloc(Node::StateVectorDot {
            start: 0,
            end: (1 << 32) + 1,
        });
        assert!(invalid_message(&arena, late_dot).contains("reads states up to 4294967297"));
        let late_input = arena.alloc(Node::InputParameter {
            name: "far".to_string(),
            index: 0,
            offset: 1 << 32,
            width: 1,
        });
        assert!(invalid_message(&arena, late_input).contains("offset 4294967296 with width 1"));
        // A slot ending past u32::MAX, which the parameter registry refuses too.
        let overrunning_input = arena.alloc(Node::InputParameter {
            name: "long".to_string(),
            index: 1,
            offset: u32::MAX as usize,
            width: 2,
        });
        assert!(invalid_message(&arena, overrunning_input).contains("past the tape's u32"));

        let half = arena.alloc(Node::StateVector {
            start: 0,
            end: (1 << 31) + 1,
        });
        assert_eq!(validate(&arena, half), Ok(()));
        let doubled = arena.alloc(Node::Concat(vec![half, half]));
        assert!(invalid_message(&arena, doubled).contains("has width 4294967298"));
    }

    #[test]
    fn validate_refuses_every_node_kind_only_the_rewrite_passes_build() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let x = arena.alloc(Node::Scalar(0.5));
        let linear = InterpolantData::try_new(vec![0.0, 1.0], vec![0.0, 1.0]).expect("valid");
        let cubic = CubicInterpolantData::try_new(vec![0.0, 1.0], vec![[0.0; 4]]).expect("valid");
        let nd =
            NdInterpolantData::try_new(vec![vec![0.0, 1.0]; 2], vec![0.0; 4], 2).expect("valid");
        for (kind, node) in [
            ("ZeroVector", Node::ZeroVector { len: 1 }),
            (
                "ReduceArgSelect",
                Node::ReduceArgSelect {
                    basis: y,
                    picker: y,
                    is_max: true,
                },
            ),
            (
                "TangentStateVector",
                Node::TangentStateVector { start: 0, end: 1 },
            ),
            ("TangentParameter", Node::TangentParameter { index: 0 }),
            (
                "Interpolant1DLinearDeriv",
                Node::Interpolant1DLinearDeriv {
                    data: Box::new(linear),
                    child: x,
                },
            ),
            (
                "Interpolant1DCubicDeriv",
                Node::Interpolant1DCubicDeriv {
                    data: Box::new(cubic),
                    child: x,
                },
            ),
            (
                "InterpolantNdPartial",
                Node::InterpolantNdPartial {
                    data: Box::new(nd),
                    children: vec![x, x],
                    axis: 0,
                },
            ),
        ] {
            let internal = arena.alloc(node);
            let root = arena.alloc(Node::Neg(internal));
            assert!(
                invalid_message(&arena, root).contains(&format!("internal {kind} node")),
                "{kind} must be refused"
            );
        }
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
        let ir = TypedIr::from_dag(DagRef::new(&arena, matmul));
        assert_eq!(ir.output_len(), 2);
    }

    #[test]
    fn test_ir_interpolant() {
        use crate::node::InterpolantData;

        let mut arena = Arena::new();
        let x = arena.alloc(Node::Scalar(1.5));
        let interp = arena.alloc(Node::Interpolant1DLinear {
            data: Box::new(
                InterpolantData::try_new(vec![0.0, 1.0, 2.0], vec![0.0, 10.0, 20.0])
                    .expect("valid interpolant"),
            ),
            child: x,
        });

        let ir = TypedIr::from_dag(DagRef::new(&arena, interp));

        assert_eq!(ir.output_len(), 1);
        assert_eq!(ir.consts().interpolants.len(), 1);
        assert_eq!(ir.consts().interpolants[0].x_data, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn slope_lookups_lower_against_the_value_pool() {
        use crate::node::InterpolantData;

        let values = InterpolantData::try_new(vec![0.0, 1.0, 3.0], vec![0.0, 10.0, 40.0])
            .expect("valid interpolant");
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let root = arena.alloc(Node::Interpolant1DLinearDeriv {
            data: Box::new(values),
            child: x,
        });
        let ir = TypedIr::from_dag(DagRef::new(&arena, root));

        assert_eq!(ir.consts().interpolants.len(), 1);
        let table = &ir.consts().interpolants[0];
        assert_eq!(table.x_data.len(), table.slopes.len() + 1);
        assert_eq!(table.slopes, &[10.0, 15.0]);
        assert!(matches!(
            ir.instructions()[1],
            Instruction::Interp1DLinearDeriv { interp_idx, .. }
                if interp_idx == InterpolantIdx::new(0)
        ));
    }

    #[test]
    fn test_const_pool() {
        let mut pool = ConstPool::new();

        let idx1 = pool.add_array(&[1.0, 2.0, 3.0]);
        let idx2 = pool.add_array(&[4.0, 5.0]);
        assert_eq!(idx1, ArrayIdx::new(0));
        assert_eq!(idx2, ArrayIdx::new(1));
        assert_eq!(pool.get_array(idx1, Width::new(3)), &[1.0, 2.0, 3.0]);
        assert_eq!(pool.get_array(idx2, Width::new(2)), &[4.0, 5.0]);

        let data =
            InterpolantData::try_new(vec![0.0, 1.0], vec![0.0, 10.0]).expect("valid interpolant");
        let interp_idx = pool.add_interpolant(&data);
        assert_eq!(interp_idx, InterpolantIdx::new(0));
        assert_eq!(pool.interpolants[0].x_data, vec![0.0, 1.0]);
        assert_eq!(pool.interpolants[0].slopes, vec![10.0]);
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
        // The product `a` is dead after the add, so only the reuse layout recycles its slot.
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let y2 = arena.alloc(Node::StateVector { start: 2, end: 3 });
        let a = arena.alloc(Node::Mul(y0, y1));
        let root = arena.alloc(Node::Add(a, y2));

        let reuse = TypedIr::from_dag(DagRef::new(&arena, root));
        let pinned = TypedIr::from_dag_pinned(DagRef::new(&arena, root));

        assert!(pinned.buffer_size() > reuse.buffer_size());

        let compiled = CompiledExpr::from_ir(pinned);
        let mut s = vec![0.0; compiled.scratch_len()];
        let out = compiled.eval(&mut s, 0.0, &[2.0, 3.0, 5.0], &[], &[]);
        assert_eq!(out, &[11.0]);
        // SSA: each instruction writes a distinct slot.
        let ir = compiled.ir();
        let mut offsets: Vec<usize> = Vec::new();
        for instr in ir.instructions() {
            if let Some(dst) = instruction_dst_extent(instr, &ir.consts) {
                offsets.push(dst.offset_usize());
            }
        }
        let unique: std::collections::BTreeSet<usize> = offsets.iter().copied().collect();
        assert_eq!(unique.len(), offsets.len(), "pinned layout reused a slot");
    }

    #[test]
    fn a_split_tape_recycles_dead_tangent_slots() {
        // d(sin^8 y) is a chain of `cos(x_k) * dx_k`: two width-4 tangents live at a time.
        let mut arena = Arena::new();
        let mut x = arena.alloc(Node::StateVector { start: 0, end: 4 });
        for _ in 0..8 {
            x = arena.alloc(Node::Sin(x));
        }
        let mut tangent = crate::arena::Dag::new(arena, x);
        tangent.rewrite_in_place(crate::tangent::tangent_wrt_states);
        let (ir, split) = TypedIr::from_dag_split_eval(tangent.as_ref());
        assert_eq!(ir.buffer_size() - split.primal_buffer_size, 8);

        let (y, dy) = ([0.1, 0.2, 0.3, 0.4], [1.0, 0.0, -1.0, 2.0]);
        let seeds = crate::eval::TangentInputs {
            dy: Some(&dy),
            dp: None,
        };
        let monolithic = CompiledExpr::new(tangent.as_ref());
        let mut scratch = vec![0.0; monolithic.scratch_len()];
        let expected = monolithic.eval_with_tangent(&mut scratch, 0.0, &y, &[], &[], &seeds);
        let expected: Vec<u64> = expected.iter().map(|v| v.to_bits()).collect();

        let split = crate::eval::SplitCompiledExpr::compile(tangent.as_ref());
        let mut scratch = vec![0.0; split.scratch_len()];
        let mut primal = split.eval_primal(&mut scratch, 0.0, &y, &[], &[]);
        let got: Vec<u64> = primal
            .eval_tangent(&seeds)
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(got, expected);
    }
}
