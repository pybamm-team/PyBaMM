//! Jacobian construction: symbolic derivative, sparsity, coloring, assembly.
//!
//! [`JacobianData::assemble_into`] is the *only* implementation of the sweep; a
//! consumer supplies a [`JacobianLayout`] and a [`JacobianScratch`].

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use crate::adjoint::AdjointTape;
use crate::arena::{Arena, Dag, DagRef, NodeId, NodeMap};
use crate::coloring::{ColumnColoring, EntryMask, color_columns_masked};
use crate::const_entries::classify_constant_entries;
use crate::eval::{PrimalCache, SplitCompiledExpr, TangentInputs};
use crate::node::Node;
use crate::row_extract::extract_scalar_rows;
use crate::simplify::{dce, node_len, simplify_pipeline};
use crate::sparsity::{SparsityPattern, detect_sparsity_per_output, transpose};
use crate::tangent::{DiffTarget, tangent_wrt_params, tangent_wrt_states, tangent_wrt_subset};
use crate::tangent_batch::{self, LanePlan, MAX_LANES};

/// `differentiate`'s graph of `root`, built over only the nodes `root` reaches.
///
/// Copying the reachable subgraph, not the whole arena, compiles to the same tapes.
pub(crate) fn tangent_dag(
    arena: &Arena,
    root: NodeId,
    differentiate: impl FnOnce(&mut Arena, NodeId) -> NodeId,
) -> Dag {
    let mut tangent = dce(DagRef::new(arena, root));
    tangent.rewrite_in_place(differentiate);
    tangent
}

/// CSC sparsity pattern, the ordering KLU expects.
///
/// Only ever a transposed [`SparsityPattern`], so each column's rows strictly ascend below
/// `nrows`, as KLU and faer's checked constructor require.
#[derive(Debug, Clone)]
pub struct CscPattern {
    nrows: usize,
    ncols: usize,
    colptr: Vec<usize>,
    rowind: Vec<usize>,
}

impl CscPattern {
    pub(crate) fn from_csr(csr: &SparsityPattern) -> Self {
        let (colptr, entries) = transpose(csr.ncols(), csr.indptr(), csr.indices());
        Self {
            nrows: csr.nrows(),
            ncols: csr.ncols(),
            colptr,
            rowind: entries.iter().map(|&(row, _)| row).collect(),
        }
    }

    #[inline]
    pub const fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub const fn ncols(&self) -> usize {
        self.ncols
    }

    /// Column pointers, length `ncols + 1`.
    #[inline]
    pub const fn colptr(&self) -> &[usize] {
        self.colptr.as_slice()
    }

    /// Row index of each non-zero.
    #[inline]
    pub const fn rowind(&self) -> &[usize] {
        self.rowind.as_slice()
    }

    #[inline]
    pub const fn nnz(&self) -> usize {
        self.rowind.len()
    }

    /// `(row, col)` of each entry, in CSC order.
    pub fn entries(&self) -> impl ExactSizeIterator<Item = (usize, usize)> + '_ {
        let mut col = 0;
        self.rowind.iter().enumerate().map(move |(entry, &row)| {
            // Skips empty columns; `colptr` ends at `nnz`, so some column holds `entry`.
            while self.colptr[col + 1] <= entry {
                col += 1;
            }
            (row, col)
        })
    }

    /// Slot of every entry of `csr`, which must be a sub-pattern of this one.
    pub(crate) fn slots_of(&self, csr: &SparsityPattern) -> Vec<usize> {
        RowSlots::new(self).slots_of(csr)
    }
}

/// A [`CscPattern`] read row by row: each row's `(column, slot)` pairs, columns ascending.
struct RowSlots {
    rowptr: Vec<usize>,
    entries: Vec<(usize, usize)>,
}

impl RowSlots {
    fn new(csc: &CscPattern) -> Self {
        let (rowptr, entries) = transpose(csc.nrows, &csc.colptr, &csc.rowind);
        Self { rowptr, entries }
    }

    fn row(&self, row: usize) -> &[(usize, usize)] {
        &self.entries[self.rowptr[row]..self.rowptr[row + 1]]
    }

    /// Slot of every entry of `csr`, which must be a sub-pattern of this one.
    fn slots_of(&self, csr: &SparsityPattern) -> Vec<usize> {
        let mut slots = Vec::with_capacity(csr.nnz());
        for (row, span) in csr.indptr().windows(2).enumerate() {
            let candidates = self.row(row);
            let mut pos = 0;
            for &col in &csr.indices()[span[0]..span[1]] {
                pos += candidates[pos..].partition_point(|&(candidate, _)| candidate < col);
                match candidates.get(pos) {
                    Some(&(found, slot)) if found == col => slots.push(slot),
                    _ => panic!("missing CSC slot for CSR entry ({row}, {col})"),
                }
            }
        }
        slots
    }
}

/// Where one entry of a color's JVP result belongs in a layout's value buffer.
#[derive(Debug, Clone, Copy)]
struct ColorScatterEntry {
    /// Slot of the value buffer to write.
    slot: usize,
    /// Row of the JVP result to read.
    row: usize,
}

/// Dense rows split out of the coloring, filled by one reverse (VJP) pass each.
#[derive(Debug, Clone)]
pub(crate) struct AdjointDenseRows {
    /// Parent output rows these gradients reconstruct, in tape-element order.
    rows: Vec<usize>,
    /// Reverse-AD tape whose element `i` is row `rows[i]`.
    tape: AdjointTape,
}

impl AdjointDenseRows {
    /// Group `rows` onto `tape`, whose element `i` must be row `rows[i]`.
    fn new(rows: Vec<usize>, tape: AdjointTape) -> Self {
        assert_eq!(
            tape.n_rows(),
            rows.len(),
            "the tape must hold one element per split row"
        );
        Self { rows, tape }
    }

    /// Rows filled by reverse mode, so reverse passes per assembly.
    #[inline]
    const fn n_rows(&self) -> usize {
        self.rows.len()
    }

    /// Fill every row's `scatter` targets in `out` from one shared forward pass.
    ///
    /// `scatter` is indexed by tape element; `grad` must span the tape's state dimension and
    /// be zero, as it is again on return.
    #[allow(clippy::too_many_arguments)]
    fn assemble_into(
        &self,
        scratch: &mut [f64],
        bar: &mut [f64],
        grad: &mut [f64],
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
        scatter: &[Vec<(usize, usize)>],
        out: &mut [f64],
    ) {
        self.tape.eval_forward(scratch, t, y, y_dot, inputs);
        for (row, targets) in scatter.iter().enumerate() {
            self.tape.assemble_row(scratch, bar, grad, row);
            for &(col, slot) in targets {
                out[slot] = grad[col];
            }
        }
        self.tape.clear_state_reads(grad);
    }
}

/// Where one [`JacobianData`]'s entries land in a consumer's value buffer.
///
/// Either the artifact's own CSC ([`JacobianData::layout`]) or a merged buffer
/// ([`JacobianData::layout_in`]).
#[derive(Debug)]
pub struct JacobianLayout {
    /// Per colour, the `(slot, row)` pairs that colour's sweep recovers.
    color_to_slots: Vec<Vec<ColorScatterEntry>>,
    /// Per dense-row group, per row of that group, `(column, slot)` targets.
    dense_row_slots: Vec<Vec<Vec<(usize, usize)>>>,
    /// Starting buffer for one assembly: constants in place, zero elsewhere.
    template: Vec<f64>,
}

impl JacobianLayout {
    /// Slots the target buffer must hold, which is its consumer's `nnz`.
    #[inline]
    pub const fn n_slots(&self) -> usize {
        self.template.len()
    }
}

/// Per-solve mutable buffers for one [`JacobianData`]'s assembly and actions.
///
/// Two concurrent assemblies of the same artifact need two.
#[derive(Debug)]
pub struct JacobianScratch {
    /// The parent tangent tape, then reused in turn by each dense-row sub-tape.
    tape: Vec<f64>,
    /// Scalar colour seed, spanning the artifact's seed dimension.
    seed: Vec<f64>,
    /// Adjoint `bar` buffer, parallel to the widest dense-row value tape.
    bar: Vec<f64>,
    lanes: usize,
    /// Lane-minor tangent region; empty when `lanes == 1`.
    tangent_lanes: Vec<f64>,
    /// Lane-minor colour seeds; empty when `lanes == 1`.
    seed_lanes: Vec<f64>,
    /// Buffers for [`JvpAt::apply_lanes`], allocated on first use.
    action_tangent_lanes: Vec<f64>,
    action_seed_lanes: Vec<f64>,
}

impl JacobianScratch {
    /// Buffers for `jac`, wide enough for the widest walk its assembly takes.
    pub fn new(jac: &JacobianData) -> Self {
        Self::with_lanes(jac, jac.lane_width())
    }

    /// Buffers pinned to the scalar one-colour-per-walk path.
    #[cfg(test)]
    pub(crate) fn scalar(jac: &JacobianData) -> Self {
        Self::with_lanes(jac, 1)
    }

    fn with_lanes(jac: &JacobianData, lanes: usize) -> Self {
        let batched = lanes > 1;
        Self {
            tape: vec![0.0; jac.max_scratch_len()],
            seed: vec![0.0; jac.seed_dim.max(1)],
            bar: vec![0.0; jac.max_adjoint_tape_len().max(1)],
            lanes,
            tangent_lanes: vec![
                0.0;
                if batched {
                    tangent_batch::tangent_scratch_len(jac.assembly_tape(), lanes)
                } else {
                    0
                }
            ],
            seed_lanes: vec![0.0; if batched { jac.seed_dim * lanes } else { 0 }],
            action_tangent_lanes: Vec::new(),
            action_seed_lanes: Vec::new(),
        }
    }

    /// Colours this scratch sweeps per walk of the tangent tape.
    #[cfg(test)]
    pub(crate) const fn lane_width(&self) -> usize {
        self.lanes
    }

    /// Overwrite every buffer with `value`, as an unwound call could leave them.
    #[cfg(test)]
    pub(crate) fn poison(&mut self, value: f64) {
        let fixed = [
            &mut self.tape,
            &mut self.seed,
            &mut self.bar,
            &mut self.tangent_lanes,
            &mut self.seed_lanes,
        ];
        for buffer in fixed {
            buffer.fill(value);
        }
        poison_grown(&mut self.action_tangent_lanes, value);
        poison_grown(&mut self.action_seed_lanes, value);
    }
}

/// Fill a buffer grown on demand with `value`, first growing it past any test model's need so
/// that the next call finds it long enough and must not trust its contents.
#[cfg(test)]
pub(crate) fn poison_grown(buffer: &mut Vec<f64>, value: f64) {
    buffer.resize(buffer.len().max(1 << 16), value);
    buffer.fill(value);
}

/// Everything `finish_with_options` switches on.
#[derive(Debug, Clone, Copy)]
struct BuildOptions {
    /// Fill wide rows by reverse mode; only a full-state build owns every column.
    split_dense_rows: bool,
    /// Sweep only entries not proven constant, writing the rest from a table.
    split_constants: bool,
}

/// What [`JacobianData::decide_coloring`] settled on.
struct ColoringDecision {
    coloring: ColumnColoring,
    swept: Vec<bool>,
    dense_rows: Vec<AdjointDenseRows>,
    /// Rows `detect_dense_rows` nominated, adopted or not.
    n_candidate_rows: usize,
}

/// Prepared derivative artifact for one (expression, wrt) pair.
///
/// Immutable after build and shared via `Arc`; each assembly brings its own
/// [`JacobianScratch`] and [`JacobianLayout`].
#[derive(Debug)]
pub struct JacobianData {
    jvp_expr: Arc<SplitCompiledExpr>,
    assembly_jvp_expr: Arc<SplitCompiledExpr>,
    sparsity: SparsityPattern,
    csc: CscPattern,
    coloring: ColumnColoring,
    wrt: DiffTarget,
    /// Global state index per local column for a subset build; empty means identity.
    col_to_global: Vec<usize>,
    /// Seed buffer length: `n_cols`, or the full state dimension for a subset build.
    seed_dim: usize,
    /// Dense rows split out of the coloring; empty unless the split lowered the colour count.
    dense_rows: Vec<AdjointDenseRows>,
    /// Rows `detect_dense_rows` nominated; above `n_dense_rows()` means a declined split.
    n_candidate_rows: usize,
    /// Per CSR entry of `sparsity`: whether a sweep must recover it.
    swept_entries: Vec<bool>,
    constant_csr_entries: Vec<(usize, f64)>,
    /// How colour sweeps walk `assembly_jvp_expr`.
    assembly_lanes: LanePlan,
    /// How [`JvpAt::apply_lanes`] walks `jvp_expr`.
    action_lanes: LanePlan,
    /// This artifact's own-CSC layout, built lazily and shared.
    own_layout: OnceLock<JacobianLayout>,
    /// The COO triplet for that layout, on the same terms.
    own_coo: OnceLock<(Vec<usize>, Vec<usize>)>,
}

impl JacobianData {
    /// `df/dy` over the full state vector, with dense-row splitting.
    pub fn new_wrt_states(arena: &Arena, root: NodeId, n_rows: usize, n_states: usize) -> Self {
        Self::new_wrt_states_inner(arena, root, n_rows, n_states, true)
    }

    /// As [`Self::new_wrt_states`], but sweeping every entry, constants included.
    pub fn new_wrt_states_sweeping_constants(
        arena: &Arena,
        root: NodeId,
        n_rows: usize,
        n_states: usize,
    ) -> Self {
        Self::new_wrt_states_inner(arena, root, n_rows, n_states, false)
    }

    fn new_wrt_states_inner(
        arena: &Arena,
        root: NodeId,
        n_rows: usize,
        n_states: usize,
        split_constants: bool,
    ) -> Self {
        let tangent = tangent_dag(arena, root, tangent_wrt_states);
        let sparsity = detect_sparsity_per_output(arena, root, n_rows, n_states);
        Self::finish_with_options(
            arena,
            root,
            tangent,
            sparsity,
            DiffTarget::States,
            Vec::new(),
            n_states,
            BuildOptions {
                split_dense_rows: true,
                split_constants,
            },
        )
    }

    /// `df/dp` over the full parameter vector, with the sparsity taken as dense.
    pub fn new_wrt_params(arena: &Arena, root: NodeId, n_rows: usize, n_params: usize) -> Self {
        let tangent = tangent_dag(arena, root, tangent_wrt_params);
        let sparsity = SparsityPattern::dense(n_rows, n_params);
        Self::finish_with_options(
            arena,
            root,
            tangent,
            sparsity,
            DiffTarget::Params,
            Vec::new(),
            n_params,
            BuildOptions {
                split_dense_rows: false,
                split_constants: false,
            },
        )
    }

    /// `dg/dy_subset` for an algebraic block, with local columns `0..subset.len()`.
    ///
    /// `subset` must be strictly ascending; the CSR/CSC column maps rely on it.
    pub fn new_wrt_state_subset(
        arena: &Arena,
        root: NodeId,
        n_rows: usize,
        n_states: usize,
        subset: &[usize],
    ) -> Self {
        assert!(
            subset.windows(2).all(|w| w[0] < w[1]),
            "subset must be strictly ascending"
        );
        let tangent = tangent_dag(arena, root, |diff_arena, root| {
            tangent_wrt_subset(diff_arena, root, subset)
        });
        let full = detect_sparsity_per_output(arena, root, n_rows, n_states);
        Self::finish_with_options(
            arena,
            root,
            tangent,
            restrict_columns(&full, subset),
            DiffTarget::States,
            subset.to_vec(),
            n_states,
            BuildOptions {
                // A dense row's adjoint tape spans every column, which a subset does not own.
                split_dense_rows: false,
                // Constants are keyed by global seed index, but these columns are local.
                split_constants: false,
            },
        )
    }

    /// Assemble the artifact.
    ///
    /// `arena`/`root` are the original primal graph, which dense-row splitting extracts from.
    #[allow(clippy::too_many_arguments)]
    fn finish_with_options(
        arena: &Arena,
        root: NodeId,
        tangent: Dag,
        sparsity: SparsityPattern,
        wrt: DiffTarget,
        col_to_global: Vec<usize>,
        seed_dim: usize,
        options: BuildOptions,
    ) -> Self {
        let mut tangent = simplify_pipeline(tangent);

        let csc = CscPattern::from_csr(&sparsity);

        // Classify on the simplified tape, so folds follow the sweep's operator order.
        let (swept, mut constant_csr_entries) = if options.split_constants {
            classify_constant_entries(tangent.arena(), tangent.root(), &sparsity)
        } else {
            (vec![true; sparsity.nnz()], Vec::new())
        };

        let ColoringDecision {
            coloring,
            swept: swept_entries,
            dense_rows,
            n_candidate_rows,
        } = Self::decide_coloring(arena, root, &sparsity, swept, options.split_dense_rows);
        let skip_rows: Vec<usize> = dense_rows
            .iter()
            .flat_map(|split| split.rows.iter().copied())
            .collect();
        retain_outside_rows(&sparsity, &skip_rows, &mut constant_csr_entries);

        let jvp_expr = Arc::new(SplitCompiledExpr::compile(tangent.as_ref()));
        // Masking is only pruning: nothing reads a split row's sweep output.
        let unmasked_root = tangent.root();
        tangent.rewrite_in_place(|diff_arena, root| {
            mask_scalar_rows(diff_arena, root, &skip_rows).unwrap_or(root)
        });
        let assembly_jvp_expr = if tangent.root() == unmasked_root {
            Arc::clone(&jvp_expr)
        } else {
            Arc::new(SplitCompiledExpr::compile(tangent.as_ref()))
        };
        let assembly_lanes = LanePlan::new(&assembly_jvp_expr);
        let action_lanes = LanePlan::new(&jvp_expr);
        Self {
            jvp_expr,
            assembly_jvp_expr,
            sparsity,
            csc,
            coloring,
            wrt,
            col_to_global,
            seed_dim,
            dense_rows,
            n_candidate_rows,
            swept_entries,
            constant_csr_entries,
            assembly_lanes,
            action_lanes,
            own_layout: OnceLock::new(),
            own_coo: OnceLock::new(),
        }
    }

    /// Choose the column coloring, and with it which rows leave the sweep.
    ///
    /// An adopted split returns `swept` with the split rows' entries cleared.
    fn decide_coloring(
        arena: &Arena,
        root: NodeId,
        sparsity: &SparsityPattern,
        swept: Vec<bool>,
        split_dense_rows: bool,
    ) -> ColoringDecision {
        // Sweep every row, colouring `swept` unless that colouring is already in hand.
        let keep_rows =
            |swept: Vec<bool>, full: Option<ColumnColoring>, n_candidate_rows| ColoringDecision {
                coloring: full.unwrap_or_else(|| {
                    color_columns_masked(sparsity, EntryMask::new(sparsity, &swept))
                }),
                swept,
                dense_rows: Vec::new(),
                n_candidate_rows,
            };
        if !split_dense_rows {
            return keep_rows(swept, None, 0);
        }
        // Measure by swept width: a row of known entries costs no colours.
        let widths = (0..sparsity.nrows())
            .map(|row| swept[sparsity.row_span(row)].iter().filter(|&&s| s).count())
            .collect::<Vec<_>>();
        let candidates = detect_dense_rows(&widths);
        if candidates.is_empty() {
            return keep_rows(swept, None, 0);
        }

        let mut narrowed = swept.clone();
        for &row in &candidates {
            narrowed[sparsity.row_span(row)].fill(false);
        }
        let reduced = color_columns_masked(sparsity, EntryMask::new(sparsity, &narrowed));
        // A row's swept columns pairwise conflict, so no full colouring beats the widest row;
        // below that the split wins without colouring dense rows (a near-complete graph).
        let widest = widths.iter().copied().max().unwrap_or(0);
        let full = (reduced.n_colors() >= widest)
            .then(|| color_columns_masked(sparsity, EntryMask::new(sparsity, &swept)));
        if full
            .as_ref()
            .is_some_and(|full| reduced.n_colors() >= full.n_colors())
        {
            return keep_rows(swept, full, candidates.len());
        }

        // Extraction is expensive, so it runs only once the colouring is known to improve.
        let dense_rows = candidates
            .chunks(ROWS_PER_TAPE)
            .map(|group| {
                let block = extract_scalar_rows(DagRef::new(arena, root), group);
                let tape = AdjointTape::new(block.dag.as_ref(), sparsity.ncols());
                AdjointDenseRows::new(block.rows, tape)
            })
            .collect();
        ColoringDecision {
            coloring: reduced,
            swept: narrowed,
            n_candidate_rows: candidates.len(),
            dense_rows,
        }
    }

    /// This artifact's entries laid out on its own CSC, built on first ask and shared.
    pub fn layout(&self) -> &JacobianLayout {
        self.own_layout.get_or_init(|| self.layout_in(&self.csc))
    }

    /// This artifact's entries laid out on `buffer`, a superset of [`Self::sparsity`].
    ///
    /// Slots with no entry here keep the template's zero for the consumer to fill.
    #[must_use]
    pub fn layout_in(&self, buffer: &CscPattern) -> JacobianLayout {
        let n_slots = buffer.nnz();
        let csr_to_slot = buffer.slots_of(&self.sparsity);
        let color_to_slots = build_color_scatter_entries(
            &self.sparsity,
            &csr_to_slot,
            &self.coloring,
            &self.swept_entries,
        );
        // Only the derivative's own entries: a merged buffer's other slots are not ours to write.
        let dense_row_slots = self
            .dense_rows
            .iter()
            .map(|split| {
                let targets = |row: usize| {
                    let span = self.sparsity.row_span(row);
                    let columns = self.sparsity.indices()[span.clone()].iter().copied();
                    columns.zip(csr_to_slot[span].iter().copied()).collect()
                };
                split.rows.iter().map(|&row| targets(row)).collect()
            })
            .collect();
        let mut template = vec![0.0; n_slots];
        for &(csr_idx, value) in &self.constant_csr_entries {
            template[csr_to_slot[csr_idx]] = value;
        }
        // Bucketing a split row would alias the sums of same-coloured columns.
        debug_assert!(
            {
                let split: HashSet<usize> = self
                    .dense_rows
                    .iter()
                    .flat_map(|group| group.rows.iter().copied())
                    .collect();
                color_to_slots
                    .iter()
                    .flatten()
                    .all(|entry| !split.contains(&entry.row))
            },
            "a split dense row must not be scattered by any colour"
        );
        JacobianLayout {
            color_to_slots,
            dense_row_slots,
            template,
        }
    }

    /// COO `(rows, cols)` in [`layout`](Self::layout) order, columns as global state indices.
    pub(crate) fn coo_global_indices(&self) -> (&[usize], &[usize]) {
        let (rows, cols) = self.own_coo.get_or_init(|| {
            self.csc
                .entries()
                .map(|(row, col)| (row, self.global_column(col)))
                .unzip()
        });
        (rows, cols)
    }

    /// Global state index of a local column; `col_to_global` empty is identity.
    #[inline]
    fn global_column(&self, col: usize) -> usize {
        self.col_to_global.get(col).copied().unwrap_or(col)
    }

    /// CSR pattern of the derivative alone: no mass term, no `cj`, pure `df/d·`.
    #[inline]
    pub const fn sparsity(&self) -> &SparsityPattern {
        &self.sparsity
    }

    /// The same entries in CSC (KLU/scipy) ordering.
    #[inline]
    pub const fn csc(&self) -> &CscPattern {
        &self.csc
    }

    /// Column coloring; with a dense-row split adopted it does not cover every row.
    #[inline]
    pub(crate) const fn coloring(&self) -> &ColumnColoring {
        &self.coloring
    }

    /// Sweeps of the tangent tape one assembly costs, before batching.
    #[inline]
    pub const fn n_colors(&self) -> usize {
        self.coloring.n_colors()
    }

    #[inline]
    pub const fn wrt(&self) -> DiffTarget {
        self.wrt
    }

    /// Rows of the derivative, matching the primal expression's output length.
    #[inline]
    pub const fn n_rows(&self) -> usize {
        self.sparsity.nrows()
    }

    /// Columns of the derivative: states, parameters, or the subset size.
    #[inline]
    pub const fn n_cols(&self) -> usize {
        self.sparsity.ncols()
    }

    /// `(csr_idx, value)` of every proven-constant entry; split dense rows are absent.
    #[inline]
    pub fn constant_csr_entries(&self) -> &[(usize, f64)] {
        &self.constant_csr_entries
    }

    /// Output rows filled by reverse mode rather than by the colour sweep, group by group.
    pub fn dense_row_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.dense_rows
            .iter()
            .flat_map(|split| split.rows.iter().copied())
    }

    /// Rows `detect_dense_rows` nominated, adopted or not.
    #[inline]
    pub const fn n_candidate_rows(&self) -> usize {
        self.n_candidate_rows
    }

    /// The unpruned tangent tape, which every matrix-free action walks.
    #[inline]
    pub const fn action_tape(&self) -> &Arc<SplitCompiledExpr> {
        &self.jvp_expr
    }

    /// The tape colored assembly walks, with split dense rows' outputs masked off.
    #[inline]
    pub const fn assembly_tape(&self) -> &Arc<SplitCompiledExpr> {
        &self.assembly_jvp_expr
    }

    /// Most colours one walk of the assembly tape carries, or 1 for the scalar path.
    #[inline]
    pub(crate) fn lane_width(&self) -> usize {
        self.assembly_lanes.lanes(self.coloring.n_colors())
    }

    /// Scratch length covering the parent tape and every dense-row value tape, which share it.
    fn max_scratch_len(&self) -> usize {
        self.jvp_expr
            .scratch_len()
            .max(self.assembly_jvp_expr.scratch_len())
            .max(self.max_adjoint_tape_len())
    }

    /// Length of the adjoint `bar` buffer: the largest dense-row value tape.
    pub(crate) fn max_adjoint_tape_len(&self) -> usize {
        self.dense_rows
            .iter()
            .map(|split| split.tape.scratch_len())
            .max()
            .unwrap_or(0)
    }

    /// Rows filled by reverse mode rather than by the colour sweep.
    pub fn n_dense_rows(&self) -> usize {
        self.dense_rows.iter().map(AdjointDenseRows::n_rows).sum()
    }

    /// Jacobian entries those rows account for.
    pub fn dense_row_entries(&self) -> usize {
        let widths = self.sparsity.indptr();
        self.dense_rows
            .iter()
            .flat_map(|split| &split.rows)
            .map(|&row| widths[row + 1] - widths[row])
            .sum()
    }

    /// Instructions across the split rows' tapes.
    pub fn dense_row_tape_instructions(&self) -> usize {
        self.dense_rows
            .iter()
            .map(|split| split.tape.instruction_count())
            .sum()
    }

    /// Number of non-zeros in the derivative sparsity pattern.
    pub const fn nnz(&self) -> usize {
        self.sparsity.nnz()
    }

    /// Assemble the derivative values into `out` at `layout`'s slots.
    ///
    /// Overwrites all `layout.n_slots()` slots: constants and swept entries at their values,
    /// every other slot zero, so `out` need not be zeroed. No mass term and no `cj`.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble_into(
        &self,
        scratch: &mut JacobianScratch,
        layout: &JacobianLayout,
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
        out: &mut [f64],
    ) {
        // Checked in release too: a foreign layout scatters into the wrong slots without panicking.
        assert_eq!(
            layout.color_to_slots.len(),
            self.coloring.n_colors(),
            "layout was built for a different jacobian"
        );
        assert_eq!(
            layout.dense_row_slots.len(),
            self.dense_rows.len(),
            "layout was built for a different jacobian"
        );
        let n_slots = layout.n_slots();
        assert!(
            out.len() >= n_slots,
            "jacobian buffer too small: need {n_slots}, got {}",
            out.len()
        );
        out[..n_slots].copy_from_slice(&layout.template);

        if scratch.lanes > 1 {
            self.sweep_colors_batched(scratch, layout, t, y, y_dot, inputs, out);
        } else {
            self.sweep_colors_scalar(scratch, layout, t, y, y_dot, inputs, out);
        }

        // After the colour sweeps, which free `tape` and dirty `seed`; zeroed once, `seed` is
        // every group's `grad`, and each group leaves it zero again.
        if !self.dense_rows.is_empty() {
            let JacobianScratch {
                tape, bar, seed, ..
            } = scratch;
            seed.fill(0.0);
            for (split, slots) in self.dense_rows.iter().zip(&layout.dense_row_slots) {
                split.assemble_into(tape, bar, seed, t, y, y_dot, inputs, slots, out);
            }
        }
    }

    /// One tangent-tape walk per colour.
    #[allow(clippy::too_many_arguments)]
    fn sweep_colors_scalar(
        &self,
        scratch: &mut JacobianScratch,
        layout: &JacobianLayout,
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
        out: &mut [f64],
    ) {
        if self.coloring.n_colors() == 0 {
            return;
        }
        let JacobianScratch { tape, seed, .. } = scratch;
        let mut primal = self
            .assembly_jvp_expr
            .eval_primal(tape, t, y, y_dot, inputs);
        for color in 0..self.coloring.n_colors() {
            self.sweep_one_color_scalar(&mut primal, seed, layout, color, out);
        }
    }

    /// Sweep colours in blocks by the assembly [`LanePlan`], one tangent-tape walk per block.
    ///
    /// Per-lane arithmetic is width-independent, so block widths cannot change the values.
    #[allow(clippy::too_many_arguments)]
    fn sweep_colors_batched(
        &self,
        scratch: &mut JacobianScratch,
        layout: &JacobianLayout,
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
        out: &mut [f64],
    ) {
        let n_colors = self.coloring.n_colors();
        if n_colors == 0 {
            return;
        }
        let JacobianScratch {
            tape,
            seed,
            tangent_lanes,
            seed_lanes,
            ..
        } = scratch;
        let mut primal = self
            .assembly_jvp_expr
            .eval_primal(tape, t, y, y_dot, inputs);

        // Zeroed once so a sweep unwound mid-block cannot leak seeds into the next; after that,
        // undo each block's seeds rather than memset: O(n_states), not O(n * blocks).
        seed_lanes.fill(0.0);
        let mut block = 0;
        while block < n_colors {
            let stride = self.assembly_lanes.lanes(n_colors - block);
            if stride == 1 {
                // Too few colours left for a batched walk to pay.
                self.sweep_one_color_scalar(&mut primal, seed, layout, block, out);
                block += 1;
                continue;
            }
            // Surplus lanes in a tail block are never seeded, so they scatter nothing.
            let carried = stride.min(n_colors - block);
            self.write_seed_lanes(seed_lanes, block, carried, stride, 1.0);

            let result = primal.eval_tangent_batch(stride, tangent_lanes, seed_lanes);

            for lane in 0..carried {
                for entry in &layout.color_to_slots[block + lane] {
                    out[entry.slot] = result[entry.row * stride + lane];
                }
            }
            self.write_seed_lanes(seed_lanes, block, carried, stride, 0.0);
            block += carried;
        }
    }

    /// One colour through the scalar tangent section, reusing the filled primal region.
    fn sweep_one_color_scalar(
        &self,
        primal: &mut PrimalCache<'_>,
        seed: &mut [f64],
        layout: &JacobianLayout,
        color: usize,
        out: &mut [f64],
    ) {
        seed.fill(0.0);
        for &col in self.coloring.columns_with_color(color) {
            seed[self.global_column(col)] = 1.0;
        }
        let result = primal.eval_tangent(&self.seed_as_tangent(seed));
        for entry in &layout.color_to_slots[color] {
            out[entry.slot] = result[entry.row];
        }
    }

    /// Write `value` into the seed lanes of the `carried` colours starting at `block`.
    ///
    /// Called with 1.0 before the sweep and 0.0 after, keeping `seed_lanes` zero between blocks.
    fn write_seed_lanes(
        &self,
        seed_lanes: &mut [f64],
        block: usize,
        carried: usize,
        stride: usize,
        value: f64,
    ) {
        for lane in 0..carried {
            for &col in self.coloring.columns_with_color(block + lane) {
                seed_lanes[self.global_column(col) * stride + lane] = value;
            }
        }
    }

    /// Point the seed buffer at whichever tangent input this artifact seeds.
    #[inline]
    const fn seed_as_tangent<'s>(&self, seed: &'s [f64]) -> TangentInputs<'s> {
        match self.wrt {
            DiffTarget::States => TangentInputs {
                dy: Some(seed),
                dp: None,
            },
            DiffTarget::Params => TangentInputs {
                dy: None,
                dp: Some(seed),
            },
        }
    }

    /// Matrix-free `J . v` into `out`, through the unpruned tangent tape.
    ///
    /// `v` holds one value per column; a subset build scatters it to global state positions.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn action_into(
        &self,
        scratch: &mut JacobianScratch,
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) {
        let JacobianScratch { tape, seed, .. } = scratch;
        let seed = self.direction_seed(v, seed);
        self.assert_output_fits(out);
        let result =
            self.jvp_expr
                .eval_with_tangent(tape, t, y, y_dot, inputs, &self.seed_as_tangent(seed));
        out[..result.len()].copy_from_slice(result);
    }

    /// Run the primal pass at `(t, y)`, returning a [`JvpAt`] for `J · v` there.
    ///
    /// Each product matches [`Self::action_into`] bit for bit.
    pub(crate) fn action_at<'a>(
        &'a self,
        scratch: &'a mut JacobianScratch,
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
    ) -> JvpAt<'a> {
        let JacobianScratch {
            tape,
            seed,
            action_tangent_lanes,
            action_seed_lanes,
            ..
        } = scratch;
        JvpAt {
            jac: self,
            primal: self.jvp_expr.eval_primal(tape, t, y, y_dot, inputs),
            seed,
            tangent_lanes: action_tangent_lanes,
            seed_lanes: action_seed_lanes,
        }
    }

    /// The tangent seed for `v`: `v` itself, or scattered to global indices for a subset.
    fn direction_seed<'s>(&self, v: &'s [f64], scratch: &'s mut [f64]) -> &'s [f64] {
        assert!(
            v.len() >= self.n_cols(),
            "jacobian tangent buffer too small: need {}, got {}",
            self.n_cols(),
            v.len()
        );
        if self.col_to_global.is_empty() {
            return v;
        }
        scratch.fill(0.0);
        for (col, &seed) in v.iter().enumerate().take(self.n_cols()) {
            scratch[self.global_column(col)] = seed;
        }
        scratch
    }

    fn assert_output_fits(&self, out: &[f64]) {
        assert!(
            out.len() >= self.n_rows(),
            "jacobian output buffer too small: need {}, got {}",
            self.n_rows(),
            out.len()
        );
    }
}

/// Matrix-free Jacobian-vector products at one point whose primal pass has run.
#[derive(Debug)]
pub(crate) struct JvpAt<'a> {
    jac: &'a JacobianData,
    /// The action tape with this point's primal section evaluated.
    primal: PrimalCache<'a>,
    seed: &'a mut [f64],
    tangent_lanes: &'a mut Vec<f64>,
    seed_lanes: &'a mut Vec<f64>,
}

impl JvpAt<'_> {
    /// Write `J · v` into `out`.
    pub fn apply(&mut self, v: &[f64], out: &mut [f64]) {
        let jac = self.jac;
        let seed = jac.direction_seed(v, self.seed);
        jac.assert_output_fits(out);
        let result = self.primal.eval_tangent(&jac.seed_as_tangent(seed));
        out[..result.len()].copy_from_slice(result);
    }

    /// Write `J · v` into the matching `outs` entry for each `v` in `directions`.
    ///
    /// Batched by the action tape's [`LanePlan`]; each result matches [`Self::apply`] bit for bit.
    pub fn apply_lanes<'d, 'o>(
        &mut self,
        directions: impl ExactSizeIterator<Item = &'d [f64]>,
        outs: impl ExactSizeIterator<Item = &'o mut [f64]>,
    ) {
        let mut remaining = directions.len();
        assert_eq!(
            remaining,
            outs.len(),
            "jacobian action needs one output per direction"
        );
        let mut pairs = directions.zip(outs);
        while remaining > 0 {
            let width = self.jac.action_lanes.lanes(remaining);
            if width == 1 {
                let Some((v, out)) = pairs.next() else {
                    return;
                };
                self.apply(v, out);
                remaining -= 1;
            } else {
                let carried = self.apply_block(width, &mut pairs);
                if carried == 0 {
                    return;
                }
                remaining -= carried;
            }
        }
    }

    /// Evaluate up to `width` directions in one batched pass, returning how many ran.
    fn apply_block<'d, 'o>(
        &mut self,
        width: usize,
        pairs: &mut impl Iterator<Item = (&'d [f64], &'o mut [f64])>,
    ) -> usize {
        let jac = self.jac;
        let tangent_len = tangent_batch::tangent_scratch_len(&jac.jvp_expr, width);
        if self.tangent_lanes.len() < tangent_len {
            self.tangent_lanes.resize(tangent_len, 0.0);
        }
        self.seed_lanes.clear();
        self.seed_lanes.resize(jac.seed_dim * width, 0.0);

        let mut outs: [Option<&'o mut [f64]>; MAX_LANES] = Default::default();
        let mut carried = 0;
        for (out_slot, (v, out)) in outs.iter_mut().zip(pairs.take(width)) {
            assert!(
                v.len() >= jac.n_cols(),
                "jacobian tangent buffer too small: need {}, got {}",
                jac.n_cols(),
                v.len()
            );
            jac.assert_output_fits(out);
            for (col, &value) in v.iter().enumerate().take(jac.n_cols()) {
                self.seed_lanes[jac.global_column(col) * width + carried] = value;
            }
            *out_slot = Some(out);
            carried += 1;
        }

        let result = self
            .primal
            .eval_tangent_batch(width, self.tangent_lanes, self.seed_lanes);
        for (lane, out) in outs.iter_mut().take(carried).enumerate() {
            let out = out.take().expect("every carried lane has an output");
            for (row, value) in out.iter_mut().take(result.len() / width).enumerate() {
                *value = result[row * width + lane];
            }
        }
        carried
    }
}

/// Per-color scatter entries, for only the entries `swept` marks.
///
/// Unswept entries must stay out: a split row would alias same-colour sums, and
/// a constant entry's sweep value is polluted by design.
fn build_color_scatter_entries(
    sparsity: &SparsityPattern,
    csr_to_slot: &[usize],
    coloring: &ColumnColoring,
    swept: &[bool],
) -> Vec<Vec<ColorScatterEntry>> {
    let mut color_to_slots = vec![Vec::new(); coloring.n_colors()];
    for (csr_idx, row) in sparsity.entry_rows().into_iter().enumerate() {
        if !swept[csr_idx] {
            continue;
        }
        let color = coloring.colors()[sparsity.indices()[csr_idx]];
        color_to_slots[color].push(ColorScatterEntry {
            slot: csr_to_slot[csr_idx],
            row,
        });
    }
    color_to_slots
}

/// Keep only the entries outside `rows`, which their own tapes fill wholesale.
fn retain_outside_rows(pattern: &SparsityPattern, rows: &[usize], entries: &mut Vec<(usize, f64)>) {
    if rows.is_empty() || entries.is_empty() {
        return;
    }
    let mut dropped = vec![false; pattern.nnz()];
    for &row in rows {
        dropped[pattern.row_span(row)].fill(true);
    }
    entries.retain(|&(csr_idx, _)| !dropped[csr_idx]);
}

/// `pattern` restricted to the columns in `subset`, renumbered to their positions there.
///
/// `subset` must ascend, so renumbered columns still ascend within each row.
fn restrict_columns(pattern: &SparsityPattern, subset: &[usize]) -> SparsityPattern {
    let mut global_to_local = vec![None; pattern.ncols()];
    for (local, &global) in subset.iter().enumerate() {
        global_to_local[global] = Some(local);
    }
    let mut indptr = Vec::with_capacity(pattern.nrows() + 1);
    let mut indices = Vec::new();
    indptr.push(0);
    for span in pattern.indptr().windows(2) {
        let kept = pattern.indices()[span[0]..span[1]]
            .iter()
            .filter_map(|&col| global_to_local[col]);
        indices.extend(kept);
        indptr.push(indices.len());
    }
    SparsityPattern::new_unchecked(pattern.nrows(), subset.len(), indptr, indices)
}

/// Minimum row width to split out; max row nnz is the coloring's lower bound.
const DENSE_ROW_MIN_NNZ: usize = 16;
/// Cap on reverse passes, so a pathological pattern cannot allocate unbounded tapes.
const MAX_DENSE_ROWS: usize = 1024;
/// Rows per shared adjoint tape, trading assembly time for compiled memory.
const ROWS_PER_TAPE: usize = 4;

/// The `k` widest rows, with `k` minimising the sweep count `k + nnz(wide[k])`.
///
/// `row_widths` counts only the entries a sweep must recover.
fn detect_dense_rows(row_widths: &[usize]) -> Vec<usize> {
    let nnz = |row: usize| row_widths[row];
    let mut wide: Vec<usize> = (0..row_widths.len())
        .filter(|&row| nnz(row) >= DENSE_ROW_MIN_NNZ)
        .collect();
    wide.sort_unstable_by_key(|&row| std::cmp::Reverse(nnz(row)));

    let bound_after = |k: usize| wide.get(k).map_or(0, |&row| nnz(row));
    let unsplit = bound_after(0);
    let cap = wide
        .len()
        .min(MAX_DENSE_ROWS)
        .min(unsplit.saturating_sub(1));
    let mut best_cost = unsplit;
    let mut best_k = 0;
    for k in 1..=cap {
        let cost = k + bound_after(k);
        if cost < best_cost {
            best_cost = cost;
            best_k = k;
        }
    }
    wide.truncate(best_k);
    wide.sort_unstable();
    wide
}

/// Replace selected scalar output rows with zeros while preserving output width.
///
/// Walks nested `Concat`s with an explicit stack, so their depth is bounded by the heap.
fn mask_scalar_rows(arena: &mut Arena, root: NodeId, rows: &[usize]) -> Option<NodeId> {
    if rows.is_empty() {
        return Some(root);
    }
    if !rows.windows(2).all(|pair| pair[0] < pair[1]) {
        return None;
    }
    let mut lens = NodeMap::new(arena.len());
    let mut open: Vec<MaskFrame> = Vec::new();
    let mut next = Some((root, rows.to_vec()));
    let mut finished = None;
    loop {
        if let Some((id, rows)) = next.take() {
            if rows.is_empty() {
                finished = Some(id);
            } else if let Node::Concat(children) = arena.get(id) {
                open.push(MaskFrame {
                    masked: Vec::with_capacity(children.len()),
                    children: children.clone(),
                    rows,
                    offset: 0,
                });
            } else if rows == [0] && node_len(arena, id, &mut lens) == 1 {
                finished = Some(arena.alloc(Node::Scalar(0.0)));
            } else {
                return None;
            }
        }
        let Some(frame) = open.last_mut() else {
            return finished;
        };
        if let Some(id) = finished.take() {
            frame.masked.push(id);
        }
        if let Some(&child) = frame.children.get(frame.masked.len()) {
            let (offset, len) = (frame.offset, node_len(arena, child, &mut lens));
            let start = frame.rows.partition_point(|&row| row < offset);
            let end = frame.rows.partition_point(|&row| row < offset + len);
            let local = frame.rows[start..end]
                .iter()
                .map(|&row| row - offset)
                .collect();
            next = Some((child, local));
            frame.offset += len;
        } else {
            let frame = open.pop().expect("a frame is open");
            if frame.rows.iter().any(|&row| row >= frame.offset) {
                return None;
            }
            finished = Some(arena.alloc(Node::Concat(frame.masked)));
        }
    }
}

/// A `Concat` part-way through [`mask_scalar_rows`], its children masked in order.
struct MaskFrame {
    children: Vec<NodeId>,
    /// Rows to zero, relative to this `Concat`'s output.
    rows: Vec<usize>,
    /// Output offset of the next child.
    offset: usize,
    masked: Vec<NodeId>,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::arena::Arena;
    use crate::coloring::color_columns;
    use crate::eval::CompiledExpr;
    use crate::ir::TypedIr;
    use crate::node::Node;
    use crate::sparsity::tests::pattern_from_rows;

    /// f(y, p) = [y0*y1 + p0, sin(y0) * p1].
    fn toy(arena: &mut Arena) -> NodeId {
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let p0 = arena.alloc(Node::InputParameter {
            name: "a".into(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let p1 = arena.alloc(Node::InputParameter {
            name: "b".into(),
            index: 1,
            offset: 1,
            width: 1,
        });
        let prod = arena.alloc(Node::Mul(y0, y1));
        let r0 = arena.alloc(Node::Add(prod, p0));
        let s = arena.alloc(Node::Sin(y0));
        let r1 = arena.alloc(Node::Mul(s, p1));
        arena.alloc(Node::Concat(vec![r0, r1]))
    }

    fn assemble_full(jac: &JacobianData, t: f64, y: &[f64], p: &[f64]) -> Vec<Vec<f64>> {
        let layout = jac.layout();
        let mut scratch = JacobianScratch::new(jac);
        let mut data = vec![0.0; layout.n_slots()];
        jac.assemble_into(&mut scratch, layout, t, y, &[], p, &mut data);
        let mut dense = vec![vec![0.0; jac.n_cols()]; jac.n_rows()];
        for (col, span) in jac.csc().colptr.windows(2).enumerate() {
            for k in span[0]..span[1] {
                dense[jac.csc().rowind[k]][col] = data[k];
            }
        }
        dense
    }

    /// Nonlinear banded expression over `n` states with half-width `w`, fully batchable.
    pub fn banded_nonlinear(n: usize, w: usize) -> (Arena, NodeId) {
        let mut arena = Arena::new();
        let states: Vec<_> = (0..n)
            .map(|i| {
                arena.alloc(Node::StateVector {
                    start: i,
                    end: i + 1,
                })
            })
            .collect();
        let rows: Vec<_> = (0..n)
            .map(|i| {
                let mut acc = arena.alloc(Node::Exp(states[i]));
                for offset in 1..=w {
                    let left = arena.alloc(Node::Sin(states[i.saturating_sub(offset)]));
                    let right = arena.alloc(Node::Cos(states[(i + offset).min(n - 1)]));
                    let pair = arena.alloc(Node::Mul(left, right));
                    acc = arena.alloc(Node::Add(acc, pair));
                }
                acc
            })
            .collect();
        let root = arena.alloc(Node::Concat(rows));
        (arena, root)
    }

    /// A `1 x n` matrix of ones, which sums a vector into one dense row.
    fn ones_row(arena: &mut Arena, n: usize) -> NodeId {
        arena.alloc(Node::SparseMatrix(Box::new(
            crate::node::CsrData::try_new(
                vec![0, n],
                (0..n).collect(),
                vec![1.0; n],
                crate::node::Shape::matrix(1, n),
            )
            .expect("valid row matrix"),
        )))
    }

    fn assemble_with(jac: &JacobianData, scratch: &mut JacobianScratch, y: &[f64]) -> Vec<f64> {
        let layout = jac.layout();
        let mut out = vec![f64::NAN; layout.n_slots()];
        jac.assemble_into(scratch, layout, 0.5, y, &[], &[], &mut out);
        out
    }

    #[allow(clippy::float_cmp)]
    fn assert_batched_matches_scalar(jac: &JacobianData, y: &[f64]) {
        assert!(jac.lane_width() > 1, "the fixture must batch");
        let batched = assemble_with(jac, &mut JacobianScratch::new(jac), y);
        let scalar = assemble_with(jac, &mut JacobianScratch::scalar(jac), y);
        assert_eq!(batched, scalar);
    }

    fn batching_probe_states(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64).mul_add(0.037, 0.41)).collect()
    }

    #[test]
    fn a_batched_sweep_matches_the_scalar_one_bit_for_bit() {
        let (n, w) = (64, 3);
        let (arena, root) = banded_nonlinear(n, w);
        let n_rows = CompiledExpr::new(DagRef::new(&arena, root)).output_len();
        let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n);
        assert_batched_matches_scalar(&jac, &batching_probe_states(n));
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn a_batched_sweep_recovers_from_seed_lanes_left_set() {
        // As if a sweep had unwound between seeding a block and undoing it.
        let (n, w) = (64, 3);
        let (arena, root) = banded_nonlinear(n, w);
        let n_rows = CompiledExpr::new(DagRef::new(&arena, root)).output_len();
        let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n);
        let y = batching_probe_states(n);
        let mut scratch = JacobianScratch::new(&jac);
        assert!(scratch.lane_width() > 1, "the fixture must batch");
        let clean = assemble_with(&jac, &mut scratch, &y);
        scratch.seed_lanes.fill(1.0);
        assert_eq!(assemble_with(&jac, &mut scratch, &y), clean);
    }

    /// A Jacobian whose actions batch at [`MAX_LANES`], with its state and row counts.
    fn batching_action_jac() -> (JacobianData, usize, usize) {
        let (n, w) = (64, 3);
        let (arena, root) = banded_nonlinear(n, w);
        let n_rows = CompiledExpr::new(DagRef::new(&arena, root)).output_len();
        let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n);
        assert_eq!(
            jac.action_lanes.widest(),
            MAX_LANES,
            "the fixture must batch"
        );
        (jac, n, n_rows)
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn batched_actions_match_one_whole_tape_action_each() {
        // 11 directions: one 8-wide batch, then a 4-wide batch with a spare lane.
        let (jac, n, n_rows) = batching_action_jac();
        assert_eq!([11, 3].map(|left| jac.action_lanes.lanes(left)), [8, 4]);
        let y = batching_probe_states(n);
        let directions: Vec<Vec<f64>> = (0..11)
            .map(|j| {
                (0..n)
                    .map(|i| ((i * 7 + j * 3) % 11) as f64 - 5.0)
                    .collect()
            })
            .collect();

        let mut scratch = JacobianScratch::new(&jac);
        let mut batched = vec![vec![f64::NAN; n_rows]; directions.len()];
        jac.action_at(&mut scratch, 0.5, &y, &[], &[]).apply_lanes(
            directions.iter().map(Vec::as_slice),
            batched.iter_mut().map(Vec::as_mut_slice),
        );

        for (direction, got) in directions.iter().zip(&batched) {
            let mut want = vec![0.0; n_rows];
            jac.action_into(&mut scratch, 0.5, &y, &[], &[], direction, &mut want);
            assert_eq!(got, &want);
        }
    }

    /// `exp(y) + sin(left) y + cos(right) y` over `n - 2` rows: long element-wise vectors.
    fn long_band(n: usize) -> (Arena, NodeId) {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: n });
        let [left, centre, right] = [0, 1, 2].map(|start| {
            arena.alloc(Node::Index {
                child: y,
                start,
                end: start + n - 2,
            })
        });
        let exp = arena.alloc(Node::Exp(centre));
        let sin = arena.alloc(Node::Sin(left));
        let cos = arena.alloc(Node::Cos(right));
        let [sin, cos] = [sin, cos].map(|term| arena.alloc(Node::Mul(term, centre)));
        let partial = arena.alloc(Node::Add(exp, sin));
        let root = arena.alloc(Node::Add(partial, cos));
        (arena, root)
    }

    /// `D (exp(y) * (G y)) + sin(y)` with bidiagonal `D` and `G`: a finite-volume flux.
    fn diffusion(n: usize) -> (Arena, NodeId) {
        let bidiagonal = |arena: &mut Arena, rows: usize, cols: usize, shift: usize| {
            let (mut indptr, mut indices) = (vec![0], Vec::new());
            for row in 0..rows {
                indices.extend((row + shift..row + shift + 2).filter(|&c| (1..=cols).contains(&c)));
                indptr.push(indices.len());
            }
            let data = indices
                .iter()
                .map(|&c| if c % 2 == 0 { 1.0 } else { -1.0 })
                .collect();
            let indices = indices.iter().map(|c| c - 1).collect();
            let csr = crate::node::CsrData::try_new(
                indptr,
                indices,
                data,
                crate::node::Shape::matrix(rows, cols),
            );
            arena.alloc(Node::SparseMatrix(Box::new(csr.expect("valid bidiagonal"))))
        };
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: n });
        let gradient = bidiagonal(&mut arena, n - 1, n, 1);
        let difference = arena.alloc(Node::MatMul(gradient, y));
        let head = arena.alloc(Node::Index {
            child: y,
            start: 0,
            end: n - 1,
        });
        let diffusivity = arena.alloc(Node::Exp(head));
        let flux = arena.alloc(Node::Mul(diffusivity, difference));
        let divergence = bidiagonal(&mut arena, n, n - 1, 0);
        let net = arena.alloc(Node::MatMul(divergence, flux));
        let source = arena.alloc(Node::Sin(y));
        let root = arena.alloc(Node::Add(net, source));
        (arena, root)
    }

    #[test]
    fn lane_plans_follow_the_tapes_shape() {
        let plans = |arena: &Arena, root: NodeId, n: usize| {
            let n_rows = CompiledExpr::new(DagRef::new(arena, root)).output_len();
            let jac = JacobianData::new_wrt_states(arena, root, n_rows, n);
            (jac.n_colors(), jac.lane_width(), jac.action_lanes.widest())
        };
        // Short scalar rows are dispatch-bound: three colours take one part-empty 4-lane walk.
        let (arena, root) = banded_nonlinear(64, 1);
        assert_eq!(plans(&arena, root, 64), (3, 4, MAX_LANES));
        // Long element-wise vectors repeat their work per lane, so they walk scalar...
        let (arena, root) = long_band(2000);
        assert_eq!(plans(&arena, root, 2000), (3, 1, 1));
        // ...unless sparse products, whose rows sum side by side across lanes, carry the walk.
        let (arena, root) = diffusion(2000);
        assert_eq!(plans(&arena, root, 2000), (3, 4, MAX_LANES));
    }

    /// The bench fixtures' tapes batch, so real models take the lane-batched sweep.
    #[cfg(feature = "serialize")]
    #[test]
    fn real_model_tapes_batch() {
        use crate::snapshot::DagSnapshot;

        let fixtures: [&[u8]; 2] = [
            include_bytes!("../benches/fixtures/spme.bin"),
            include_bytes!("../benches/fixtures/dfn.bin"),
        ];
        for (bytes, colours_per_walk) in fixtures.into_iter().zip([4, MAX_LANES]) {
            let snapshot = DagSnapshot::from_bytes(bytes).expect("fixture decodes");
            let (arena, root) = (snapshot.dag().arena(), snapshot.dag().root());
            let n_rows = CompiledExpr::new(DagRef::new(arena, root)).output_len();
            let jac = JacobianData::new_wrt_states(arena, root, n_rows, snapshot.n_states);
            assert_eq!(jac.lane_width(), colours_per_walk);
            assert_eq!(jac.action_lanes.widest(), MAX_LANES);
        }
    }

    #[test]
    #[should_panic(expected = "one output per direction")]
    fn batched_actions_reject_fewer_outputs_than_directions() {
        // Without the count check, running out of outputs hung the batching loop.
        let (jac, n, _) = batching_action_jac();
        let y = batching_probe_states(n);
        let directions = vec![vec![1.0; n]; 8];
        let mut scratch = JacobianScratch::new(&jac);
        jac.action_at(&mut scratch, 0.5, &y, &[], &[])
            .apply_lanes(directions.iter().map(Vec::as_slice), std::iter::empty());
    }

    /// An iterator that reports `claimed` items however many it yields.
    struct OverReported<I> {
        inner: I,
        claimed: usize,
    }

    impl<I: Iterator> Iterator for OverReported<I> {
        type Item = I::Item;
        fn next(&mut self) -> Option<I::Item> {
            self.inner.next()
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            (self.claimed, Some(self.claimed))
        }
    }

    impl<I: Iterator> ExactSizeIterator for OverReported<I> {}

    #[test]
    #[allow(clippy::float_cmp)]
    fn batched_actions_stop_when_the_directions_run_out_early() {
        // Claiming 8 but yielding 3 used to hang the batching loop.
        let (jac, n, n_rows) = batching_action_jac();
        let y = batching_probe_states(n);
        let directions: Vec<Vec<f64>> = (1..=3).map(|value| vec![f64::from(value); n]).collect();
        let mut batched = vec![vec![f64::NAN; n_rows]; directions.len()];
        let mut scratch = JacobianScratch::new(&jac);
        jac.action_at(&mut scratch, 0.5, &y, &[], &[]).apply_lanes(
            OverReported {
                inner: directions.iter().map(Vec::as_slice),
                claimed: 8,
            },
            OverReported {
                inner: batched.iter_mut().map(Vec::as_mut_slice),
                claimed: 8,
            },
        );

        for (direction, got) in directions.iter().zip(&batched) {
            let mut want = vec![0.0; n_rows];
            jac.action_into(&mut scratch, 0.5, &y, &[], &[], direction, &mut want);
            assert_eq!(got, &want);
        }
    }

    fn act_into_short_outputs(n_directions: usize) {
        let (jac, n, n_rows) = batching_action_jac();
        let y = batching_probe_states(n);
        let directions = vec![vec![1.0; n]; n_directions];
        let mut outs = vec![vec![0.0; n_rows - 1]; n_directions];
        let mut scratch = JacobianScratch::new(&jac);
        jac.action_at(&mut scratch, 0.5, &y, &[], &[]).apply_lanes(
            directions.iter().map(Vec::as_slice),
            outs.iter_mut().map(Vec::as_mut_slice),
        );
    }

    #[test]
    #[should_panic(expected = "jacobian output buffer too small")]
    fn a_scalar_action_rejects_a_short_output() {
        act_into_short_outputs(1);
    }

    #[test]
    #[should_panic(expected = "jacobian output buffer too small")]
    fn a_four_wide_action_rejects_a_short_output() {
        act_into_short_outputs(4);
    }

    #[test]
    #[should_panic(expected = "jacobian output buffer too small")]
    fn an_eight_wide_action_rejects_a_short_output() {
        act_into_short_outputs(8);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn a_merged_layout_only_renumbers_the_slots() {
        let (n, w) = (32, 2);
        let (arena, root) = banded_nonlinear(n, w);
        let n_rows = CompiledExpr::new(DagRef::new(&arena, root)).output_len();
        let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n);

        // Make every merged row wider, so no slot number survives by luck.
        let extra = SparsityPattern::dense(n_rows, n);
        let mut merged = jac.sparsity().clone();
        merged.merge_with(&extra);
        let merged_csc = CscPattern::from_csr(&merged);
        let merged_layout = jac.layout_in(&merged_csc);

        let y: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.041, 0.23)).collect();
        let own = assemble_with(&jac, &mut JacobianScratch::new(&jac), &y);
        let mut wide = vec![f64::NAN; merged_layout.n_slots()];
        jac.assemble_into(
            &mut JacobianScratch::new(&jac),
            &merged_layout,
            0.5,
            &y,
            &[],
            &[],
            &mut wide,
        );

        let mut own_by_entry = std::collections::HashMap::new();
        for (slot, (row, col)) in jac.csc().entries().enumerate() {
            own_by_entry.insert((row, col), own[slot]);
        }
        let mut covered = 0;
        for (slot, (row, col)) in merged_csc.entries().enumerate() {
            match own_by_entry.get(&(row, col)) {
                Some(&value) => {
                    assert_eq!(wide[slot], value, "entry ({row}, {col}) moved value");
                    covered += 1;
                },
                None => assert_eq!(wide[slot], 0.0, "slot ({row}, {col}) is not ours to write"),
            }
        }
        assert_eq!(covered, jac.nnz(), "every derivative entry must be placed");
    }

    #[test]
    fn a_reverse_filled_row_writes_only_its_own_entries() {
        // At y20 = 0, `sqrt(sign(y20))` leaves `grad[20] = 0 * inf`, off the row's pattern.
        let n = 21;
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: n });
        let head = arena.alloc(Node::Index {
            child: y,
            start: 0,
            end: n - 1,
        });
        let squares = arena.alloc(Node::Mul(head, head));
        let ones = ones_row(&mut arena, n - 1);
        let sum = arena.alloc(Node::MatMul(ones, squares));
        let last = arena.alloc(Node::Index {
            child: y,
            start: n - 1,
            end: n,
        });
        let sign = arena.alloc(Node::Sign(last));
        let root_of_sign = arena.alloc(Node::Sqrt(sign));
        let dense = arena.alloc(Node::Add(sum, root_of_sign));
        let diagonal = arena.alloc(Node::Sin(y));
        let root = arena.alloc(Node::Concat(vec![dense, diagonal]));
        let jac = JacobianData::new_wrt_states(&arena, root, n + 1, n);
        assert_eq!(jac.n_dense_rows(), 1);

        let mut merged = jac.sparsity().clone();
        merged.merge_with(&SparsityPattern::dense(n + 1, n));
        let merged_csc = CscPattern::from_csr(&merged);
        let layout = jac.layout_in(&merged_csc);
        let mut y_values = vec![0.5; n];
        y_values[n - 1] = 0.0;
        let mut out = vec![f64::NAN; layout.n_slots()];
        let mut scratch = JacobianScratch::new(&jac);
        jac.assemble_into(&mut scratch, &layout, 0.5, &y_values, &[], &[], &mut out);
        let slot = merged_csc
            .entries()
            .position(|entry| entry == (0, n - 1))
            .expect("the merged buffer holds the entry");
        assert_eq!(out[slot].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    #[should_panic(expected = "layout was built for a different jacobian")]
    fn a_layout_from_another_jacobian_is_refused() {
        // Same rows, more colours: a release build once scattered this silently.
        let (n, w) = (32, 2);
        let (banded, banded_root) = banded_nonlinear(n, w);
        let wide = JacobianData::new_wrt_states(&banded, banded_root, n, n);
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: n });
        let root = arena.alloc(Node::Sin(y));
        let diagonal = JacobianData::new_wrt_states(&arena, root, n, n);
        assert!(wide.n_colors() > diagonal.n_colors());

        let mut out = vec![0.0; wide.layout().n_slots()];
        diagonal.assemble_into(
            &mut JacobianScratch::new(&diagonal),
            wide.layout(),
            0.0,
            &batching_probe_states(n),
            &[],
            &[],
            &mut out,
        );
    }

    /// [`tangent_dag`] renumbers nodes; every tape must still match a whole-arena copy's.
    #[cfg(feature = "serialize")]
    #[test]
    fn a_reachable_copy_compiles_what_a_whole_arena_copy_does() {
        use crate::snapshot::DagSnapshot;

        let bytes = include_bytes!("../benches/fixtures/dfn.bin");
        let snapshot = DagSnapshot::from_bytes(bytes).expect("fixture decodes");
        let arena = snapshot.dag().arena();
        let Node::Concat(rows) = arena.get(snapshot.dag().root()) else {
            panic!("the fixture's rhs is a concatenation");
        };
        // Interior blocks, so the whole arena holds nodes a reachable copy drops.
        let compile = |dag: Dag| {
            format!(
                "{:?}",
                SplitCompiledExpr::compile(simplify_pipeline(dag).as_ref())
            )
        };
        for &root in &rows[1..rows.len() - 1] {
            for differentiate in [tangent_wrt_states, tangent_wrt_params] {
                let mut whole = Dag::new(arena.clone(), root);
                whole.rewrite_in_place(differentiate);
                assert_eq!(
                    compile(tangent_dag(arena, root, differentiate)),
                    compile(whole),
                );
            }
        }
    }

    #[test]
    fn a_csc_pattern_transposes_its_csr() {
        // Entries at (0,0), (1,1), (2,1), (2,2).
        let csc = CscPattern::from_csr(&pattern_from_rows(3, &[vec![0], vec![1], vec![1, 2]]));
        assert_eq!((csc.nrows(), csc.ncols(), csc.nnz()), (3, 3, 4));
        assert_eq!(csc.colptr(), [0, 1, 3, 4]);
        assert_eq!(csc.rowind(), [0, 1, 2, 2]);

        let diagonal = CscPattern::from_csr(&pattern_from_rows(3, &[vec![0], vec![1], vec![2]]));
        assert_eq!(diagonal.colptr(), [0, 1, 2, 3]);
        assert_eq!(diagonal.rowind(), [0, 1, 2]);
    }

    #[test]
    fn csc_entries_skip_empty_columns() {
        let csc = CscPattern::from_csr(&pattern_from_rows(4, &[vec![1, 3], vec![], vec![1, 2]]));
        let entries = csc.entries();
        assert_eq!(entries.len(), 4);
        assert_eq!(
            entries.collect::<Vec<_>>(),
            [(0, 1), (2, 1), (2, 2), (0, 3)]
        );
    }

    #[test]
    fn slots_of_finds_a_sub_pattern_in_a_wider_csc() {
        let buffer = CscPattern::from_csr(&SparsityPattern::dense(2, 3));
        let sub = pattern_from_rows(3, &[vec![1], vec![0, 2]]);
        assert_eq!(buffer.slots_of(&sub), vec![2, 1, 5]);
    }

    #[test]
    #[should_panic(expected = "missing CSC slot for CSR entry (0, 1)")]
    fn slots_of_rejects_an_entry_the_csc_lacks() {
        let diagonal = pattern_from_rows(2, &[vec![0], vec![1]]);
        let sub = pattern_from_rows(2, &[vec![1], vec![]]);
        let _ = CscPattern::from_csr(&diagonal).slots_of(&sub);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn a_subset_build_batches_onto_the_same_values() {
        let (n, w) = (64, 3);
        let (arena, root) = banded_nonlinear(n, w);
        let n_rows = CompiledExpr::new(DagRef::new(&arena, root)).output_len();
        let subset: Vec<usize> = (0..n).collect();
        let jac = JacobianData::new_wrt_state_subset(&arena, root, n_rows, n, &subset);
        let y = batching_probe_states(n);
        assert_batched_matches_scalar(&jac, &y);

        let subset_values = assemble_with(&jac, &mut JacobianScratch::new(&jac), &y);
        let full = JacobianData::new_wrt_states(&arena, root, n_rows, n);
        let reference = assemble_with(&full, &mut JacobianScratch::new(&full), &y);
        assert_eq!(
            jac.csc().entries().collect::<Vec<_>>(),
            full.csc().entries().collect::<Vec<_>>()
        );
        for (k, (&got, &want)) in subset_values.iter().zip(&reference).enumerate() {
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().mul_add(1.0, 1.0),
                "slot {k}: subset {got} vs full {want}"
            );
        }
    }

    #[test]
    fn jacobian_wrt_states_matches_analytic() {
        let mut arena = Arena::new();
        let root = toy(&mut arena);
        let jac = JacobianData::new_wrt_states(&arena, root, 2, 2);
        let (y, p) = ([0.5, 2.0], [3.0, 4.0]);
        let dense = assemble_full(&jac, 0.0, &y, &p);
        // df0/dy = [y1, y0]; df1/dy = [cos(y0)*p1, 0]
        assert!((dense[0][0] - 2.0).abs() < 1e-12);
        assert!((dense[0][1] - 0.5).abs() < 1e-12);
        assert!(0.5f64.cos().mul_add(-4.0, dense[1][0]).abs() < 1e-12);
        assert!(dense[1][1].abs() < 1e-12);
    }

    #[test]
    fn jacobian_wrt_params_matches_analytic() {
        let mut arena = Arena::new();
        let root = toy(&mut arena);
        let jac = JacobianData::new_wrt_params(&arena, root, 2, 2);
        let (y, p) = ([0.5, 2.0], [3.0, 4.0]);
        let dense = assemble_full(&jac, 0.0, &y, &p);
        // df0/dp = [1, 0]; df1/dp = [0, sin(y0)]
        assert!((dense[0][0] - 1.0).abs() < 1e-12);
        assert!(dense[0][1].abs() < 1e-12);
        assert!(dense[1][0].abs() < 1e-12);
        assert!((dense[1][1] - 0.5f64.sin()).abs() < 1e-12);
    }

    #[test]
    fn jacobian_wrt_states_is_rectangular_for_partial_group() {
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let root = arena.alloc(Node::Mul(y0, y1));
        let jac = JacobianData::new_wrt_states(&arena, root, 1, 2);
        assert_eq!((jac.n_rows(), jac.n_cols()), (1, 2));
    }

    #[test]
    fn jacobian_wrt_state_subset_analytic() {
        // f(y) = y0*y2 + y1, so df/dy = [y2, 1, y0]; subset [0, 2] must drop the 1.
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let y2 = arena.alloc(Node::StateVector { start: 2, end: 3 });
        let prod = arena.alloc(Node::Mul(y0, y2));
        let root = arena.alloc(Node::Add(prod, y1));

        let jac = JacobianData::new_wrt_state_subset(&arena, root, 1, 3, &[0, 2]);

        assert_eq!((jac.n_rows(), jac.n_cols()), (1, 2));
        assert_eq!(jac.seed_dim, 3);

        let dense = assemble_full(&jac, 0.0, &[2.0, 3.0, 5.0], &[]);
        assert!(
            (dense[0][0] - 5.0).abs() < 1e-12,
            "df/dy0 wrong: {}",
            dense[0][0]
        );
        assert!(
            (dense[0][1] - 2.0).abs() < 1e-12,
            "df/dy2 wrong: {}",
            dense[0][1]
        );
        assert_eq!(
            jac.n_cols(),
            2,
            "subset Jacobian must have exactly 2 columns"
        );
    }

    #[test]
    fn jacobian_wrt_params_param_independent_row() {
        // Row 0 depends on no parameter, so nothing may read past a collapsed tangent tape.
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let p0 = arena.alloc(Node::InputParameter {
            name: "a".into(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let r0 = arena.alloc(Node::Mul(y0, y1));
        let r1 = arena.alloc(Node::Mul(y0, p0));
        let root = arena.alloc(Node::Concat(vec![r0, r1]));

        let jac = JacobianData::new_wrt_params(&arena, root, 2, 1);
        let dense = assemble_full(&jac, 0.0, &[3.0, 5.0], &[7.0]);
        assert!(
            dense[0][0].abs() < 1e-12,
            "df0/dp0 must be 0, got {}",
            dense[0][0]
        );
        assert!(
            (dense[1][0] - 3.0).abs() < 1e-12,
            "df1/dp0 must be y0=3, got {}",
            dense[1][0]
        );
    }

    #[test]
    fn jacobian_wrt_states_state_independent_leading_row() {
        // The leading constant child collapses in the tangent; y0's row must not shift.
        use crate::node::{ArrayData, Shape};
        let mut arena = Arena::new();
        let cv = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0, 2.0],
            shape: Shape::vector(2),
        })));
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let root = arena.alloc(Node::Concat(vec![cv, y0]));

        let jac = JacobianData::new_wrt_states(&arena, root, 3, 1);
        let dense = assemble_full(&jac, 0.0, &[9.0], &[]);
        assert!(
            dense[0][0].abs() < 1e-12,
            "row 0 must be 0, got {}",
            dense[0][0]
        );
        assert!(
            dense[1][0].abs() < 1e-12,
            "row 1 must be 0, got {}",
            dense[1][0]
        );
        assert!(
            (dense[2][0] - 1.0).abs() < 1e-12,
            "df2/dy0 must be 1, got {}",
            dense[2][0]
        );
    }

    #[test]
    fn restrict_columns_keeps_and_renumbers_the_subset() {
        let pattern = pattern_from_rows(3, &[vec![0, 1, 2], vec![1], vec![0, 2]]);
        let restricted = restrict_columns(&pattern, &[0, 2]);
        assert_eq!((restricted.nrows(), restricted.ncols()), (3, 2));
        assert_eq!(restricted.indptr(), [0, 2, 2, 4]);
        assert_eq!(restricted.indices(), [0, 1, 0, 1]);
    }

    #[test]
    #[should_panic(expected = "subset must be strictly ascending")]
    fn jacobian_wrt_state_subset_rejects_unsorted() {
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let root = arena.alloc(Node::Add(y0, y1));
        let _ = JacobianData::new_wrt_state_subset(&arena, root, 1, 2, &[1, 0]);
    }

    fn make_pattern_with_dense_row(ncols: usize, nrows: usize) -> SparsityPattern {
        let rows: Vec<Vec<usize>> = (0..nrows)
            .map(|row| {
                if row == 0 {
                    (0..ncols).collect()
                } else {
                    vec![(row - 1) % ncols]
                }
            })
            .collect();
        pattern_from_rows(ncols, &rows)
    }

    fn make_diagonal_pattern_local(n: usize) -> SparsityPattern {
        pattern_from_rows(n, &(0..n).map(|row| vec![row]).collect::<Vec<_>>())
    }

    fn make_pattern_with_n_dense_rows(ncols: usize, n_dense: usize) -> SparsityPattern {
        let rows: Vec<Vec<usize>> = (0..ncols)
            .map(|row| {
                if row < n_dense {
                    (0..ncols).collect()
                } else {
                    vec![row % ncols]
                }
            })
            .collect();
        pattern_from_rows(ncols, &rows)
    }

    #[test]
    fn detect_dense_rows_flags_wide_rows_only() {
        let pattern = make_pattern_with_dense_row(20, 20);
        assert_eq!(detect_dense_rows(&pattern.row_widths()), vec![0]);
        let diag = make_diagonal_pattern_local(20);
        assert_eq!(detect_dense_rows(&diag.row_widths()), Vec::<usize>::new());
    }

    #[test]
    fn detect_dense_rows_splits_a_whole_plateau() {
        for n_dense in [5, 8, 40] {
            let pattern = make_pattern_with_n_dense_rows(64, n_dense);
            let widths = pattern.row_widths();
            let split = detect_dense_rows(&widths);
            assert_eq!(
                split,
                (0..n_dense).collect::<Vec<_>>(),
                "the whole plateau of {n_dense} dense rows must be split"
            );
            let unsplit = widths.iter().copied().max().unwrap();
            let remaining = (n_dense..pattern.nrows())
                .map(|row| widths[row])
                .max()
                .unwrap_or(0);
            assert!(
                split.len() + remaining < unsplit,
                "{n_dense} reverse passes + {remaining} colours must beat {unsplit}"
            );
        }
    }

    #[test]
    fn no_full_colouring_beats_its_widest_swept_row() {
        // `decide_coloring` skips the full colouring once the split beats this bound.
        let pattern = make_pattern_with_n_dense_rows(48, 3);
        let swept: Vec<bool> = (0..pattern.nnz()).map(|entry| entry % 5 != 0).collect();
        let widest = (0..pattern.nrows())
            .map(|row| swept[pattern.row_span(row)].iter().filter(|&&s| s).count())
            .max()
            .unwrap();
        let full = color_columns_masked(&pattern, EntryMask::new(&pattern, &swept));
        assert!(full.n_colors() >= widest, "{} < {widest}", full.n_colors());
    }

    #[test]
    fn detect_dense_rows_declines_when_no_split_pays() {
        let all_dense = make_pattern_with_n_dense_rows(20, 20);
        assert_eq!(
            detect_dense_rows(&all_dense.row_widths()),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn detect_dense_rows_respects_the_tape_cap() {
        let ncols = MAX_DENSE_ROWS + 64;
        let pattern = make_pattern_with_n_dense_rows(ncols, MAX_DENSE_ROWS + 32);
        assert!(detect_dense_rows(&pattern.row_widths()).len() <= MAX_DENSE_ROWS);
    }

    #[test]
    fn detect_dense_rows_splits_outlier_among_many_wide_rows() {
        let ncols = 64;
        let nrows = 40;
        let rows: Vec<Vec<usize>> = (0..nrows)
            .map(|row| {
                let width = if row == 7 { ncols } else { DENSE_ROW_MIN_NNZ };
                let mut columns: Vec<usize> = (0..width).map(|c| (row + c) % ncols).collect();
                columns.sort_unstable();
                columns
            })
            .collect();
        let pattern = pattern_from_rows(ncols, &rows);

        assert_eq!(detect_dense_rows(&pattern.row_widths()), vec![7]);
        let full = color_columns(&pattern);
        let mut swept = vec![true; pattern.nnz()];
        swept[pattern.row_span(7)].fill(false);
        let reduced = color_columns_masked(&pattern, EntryMask::new(&pattern, &swept));
        assert!(
            reduced.n_colors() < full.n_colors(),
            "reduced {} should beat full {}",
            reduced.n_colors(),
            full.n_colors()
        );
    }

    #[test]
    fn mask_scalar_rows_handles_a_deep_concat_nesting() {
        const DEEP: usize = 200_000;
        let mut arena = Arena::new();
        let mut id = arena.alloc(Node::StateVector { start: 0, end: 1 });
        for _ in 0..DEEP {
            id = arena.alloc(Node::Concat(vec![id]));
        }

        let mut masked = mask_scalar_rows(&mut arena, id, &[0]).expect("a scalar row");
        let mut depth = 0;
        while let Node::Concat(children) = arena.get(masked) {
            masked = children[0];
            depth += 1;
        }
        assert_eq!(depth, DEEP);
        assert!(matches!(arena.get(masked), Node::Scalar(v) if v.to_bits() == 0));
    }

    #[test]
    fn mask_scalar_rows_preserves_nested_output_shape() {
        let mut arena = Arena::new();
        let a = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let b = arena.alloc(Node::StateVector { start: 2, end: 3 });
        let c = arena.alloc(Node::StateVector { start: 3, end: 4 });
        let inner = arena.alloc(Node::Concat(vec![a, b]));
        let root = arena.alloc(Node::Concat(vec![inner, c]));

        let masked = mask_scalar_rows(&mut arena, root, &[2, 3]).expect("scalar rows");
        let ir = TypedIr::from_dag(DagRef::new(&arena, masked));
        let expr = CompiledExpr::from_ir(ir);
        let mut scratch = vec![0.0; expr.scratch_len()];
        let result = expr.eval(&mut scratch, 0.0, &[1.0, 2.0, 3.0, 4.0], &[], &[]);
        assert_eq!(result, &[1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn mask_scalar_rows_rejects_vector_interior_and_unsorted_rows() {
        let mut arena = Arena::new();
        let vector = arena.alloc(Node::StateVector { start: 0, end: 3 });
        assert!(mask_scalar_rows(&mut arena, vector, &[1]).is_none());
        assert!(mask_scalar_rows(&mut arena, vector, &[2, 1]).is_none());
    }

    #[test]
    fn dense_rows_are_pruned_only_from_assembly_jvp() {
        let mut arena = Arena::new();
        let n = 20;
        let y = arena.alloc(Node::StateVector { start: 0, end: n });
        let square = arena.alloc(Node::Mul(y, y));
        let ones = ones_row(&mut arena, n);
        let dense = arena.alloc(Node::MatMul(ones, square));
        let mut rows = vec![dense];
        for i in 1..n {
            rows.push(arena.alloc(Node::Index {
                child: y,
                start: i,
                end: i + 1,
            }));
        }
        let root = arena.alloc(Node::Concat(rows));
        let jac = JacobianData::new_wrt_states(&arena, root, n, n);
        assert_eq!(jac.n_dense_rows(), 1);
        assert!(
            jac.assembly_tape().ir().instructions().len()
                < jac.action_tape().ir().instructions().len(),
            "dense-row pruning must shorten the raw assembly tape"
        );

        let values: Vec<f64> = (1..=n).map(|value| value as f64).collect();
        let seed = vec![1.0; n];
        let tangent = TangentInputs {
            dy: Some(&seed),
            dp: None,
        };
        let mut full_scratch = vec![0.0; jac.action_tape().scratch_len()];
        let full = jac
            .jvp_expr
            .eval_with_tangent(&mut full_scratch, 0.0, &values, &[], &[], &tangent)
            .to_vec();
        let mut assembly_scratch = vec![0.0; jac.assembly_tape().scratch_len()];
        let assembly = jac.assembly_tape().eval_with_tangent(
            &mut assembly_scratch,
            0.0,
            &values,
            &[],
            &[],
            &tangent,
        );

        let expected_dense = 2.0 * values.iter().sum::<f64>();
        assert_eq!(full[0].to_bits(), expected_dense.to_bits());
        assert_eq!(assembly[0].to_bits(), 0.0f64.to_bits());
        assert!(
            assembly[1..]
                .iter()
                .zip(&full[1..])
                .all(|(&left, &right)| left.to_bits() == right.to_bits())
        );
    }

    #[test]
    fn a_reverse_filled_modulo_row_matches_its_colour_sweep() {
        // Reverse mode truncated the rounded `fl(1 / 0.1) = 10`, not the exact 9.99...
        let n = 20;
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: n });
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let diagonal = arena.alloc(Node::Sin(y));
        let remainders = arena.alloc(Node::Modulo(y0, y));
        let ones = ones_row(&mut arena, n);
        let row = arena.alloc(Node::MatMul(ones, remainders));
        let root = arena.alloc(Node::Concat(vec![diagonal, row]));
        let reverse = JacobianData::new_wrt_states(&arena, root, n + 1, n);
        let all: Vec<usize> = (0..n).collect();
        let swept = JacobianData::new_wrt_state_subset(&arena, root, n + 1, n, &all);
        assert_eq!((reverse.n_dense_rows(), swept.n_dense_rows()), (1, 0));

        let mut y = vec![0.1; n];
        y[0] = 1.0;
        let bits = |jac: &JacobianData| -> Vec<u64> {
            assemble_full(jac, 0.0, &y, &[])[n]
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        assert_eq!(bits(&reverse), bits(&swept));
        assert_eq!(bits(&swept)[1], (-9.0_f64).to_bits());
    }

    #[test]
    fn an_unmasked_build_shares_its_two_tapes() {
        let mut arena = Arena::new();
        let root = toy(&mut arena);
        let jac = JacobianData::new_wrt_states(&arena, root, 2, 2);
        assert!(Arc::ptr_eq(jac.action_tape(), jac.assembly_tape()));
    }
}
