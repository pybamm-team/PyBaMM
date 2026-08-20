//! Jacobian construction: symbolic derivative, sparsity, coloring, assembly.
//!
//! [`JacobianData`] is built once per model and holds everything an evaluation
//! needs: the derivative DAG lowered to split-eval tapes, the CSR sparsity, a
//! column coloring, and the scatter tables that place each color's results at
//! their CSC positions. Assembly then costs a *single* primal pass, whose cached
//! result every color reuses, plus one tangent sweep per color. That sharing is
//! the whole point of lowering to split-eval tapes.
//!
//! Rows too wide to color cheaply are split out and filled by reverse mode
//! instead, so a single dense row cannot force one color per column. Each such
//! row adds one backward pass over that same primal scratch.
//!
//! Entries a compile pass proves constant are lifted out of the sweep the same
//! way: they are written from a table and, being nobody's sweep result, stop
//! constraining the coloring of the columns that share their rows.
//!
//! [`JacobianData::assemble_into`] is the *only* implementation of that sweep.
//! A consumer supplies a [`JacobianLayout`] saying where this artifact's entries
//! land in its own value buffer -- the artifact's CSC, or the wider merged CSC a
//! model shares with its mass matrix -- and a [`JacobianScratch`] carrying the
//! buffers and the lane width. Batching, the constant table, the seed-lane
//! hygiene and the dense-row passes therefore exist once, whatever the consumer.

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use crate::adjoint::AdjointTape;
use crate::arena::{Arena, NodeId, NodeMap};
use crate::coloring::{ColumnColoring, color_columns_masked};
use crate::const_entries::classify_constant_entries;
use crate::eval::{CompiledExpr, TangentInputs};
use crate::ir::TypedIr;
use crate::node::Node;
use crate::row_extract::extract_scalar_rows;
use crate::simplify::{node_len, simplify_pipeline};
use crate::sparsity::{SparsityPattern, detect_sparsity_per_output};
use crate::tangent::{tangent_wrt_params, tangent_wrt_states, tangent_wrt_subset};
use crate::tangent_batch;

/// Scratch ceiling for the batched tangent region, per assembly. Wider lanes
/// stop paying once the region leaves cache, and an unbounded width would let a
/// large model allocate a lane buffer far bigger than its own tape.
const LANE_SCRATCH_BUDGET: usize = 32 << 20;

/// Colours one batched walk of `ir` should carry, or 1 for the scalar path.
///
/// The widest monomorphised width whose lane region fits [`LANE_SCRATCH_BUDGET`];
/// tapes with too few colours or unbatchable instructions stay scalar.
fn decide_lane_width(ir: &TypedIr, n_colors: usize) -> usize {
    if n_colors < 2 || !tangent_batch::is_batchable(ir) {
        return 1;
    }
    tangent_batch::SUPPORTED_LANES
        .into_iter()
        .find(|&lanes| {
            lanes <= n_colors
                && tangent_batch::tangent_scratch_len(ir, lanes) * size_of::<f64>()
                    <= LANE_SCRATCH_BUDGET
        })
        .unwrap_or(1)
}

/// Which inputs a Jacobian differentiates against; the same two cases the
/// tangent transform seeds, so the enum is shared rather than duplicated.
pub use crate::tangent::DiffTarget;

/// CSC (Compressed Sparse Column) sparsity pattern for KLU compatibility.
///
/// KLU and other direct sparse solvers expect CSC format, while internal
/// representations use CSR. This struct stores the CSC pattern for zero-allocation
/// Jacobian assembly.
#[derive(Debug, Clone)]
pub struct CscPattern {
    /// Number of rows
    pub nrows: usize,
    /// Number of columns
    pub ncols: usize,
    /// Column pointers (length ncols + 1)
    pub colptr: Vec<usize>,
    /// Row indices for each non-zero (length nnz)
    pub rowind: Vec<usize>,
    /// Mapping from CSC index to (row, col) for assembly.
    pub csc_to_csr_map: Vec<(usize, usize)>,
}

impl CscPattern {
    /// Convert CSR sparsity pattern to CSC.
    pub fn from_csr(csr: &SparsityPattern) -> Self {
        let nrows = csr.nrows;
        let ncols = csr.ncols;
        let nnz = csr.nnz();

        // Count entries per column.
        let mut col_counts = vec![0usize; ncols];
        for &col in &csr.indices {
            col_counts[col] += 1;
        }

        // Build column pointers.
        let mut colptr = vec![0usize; ncols + 1];
        for (i, &count) in col_counts.iter().enumerate() {
            colptr[i + 1] = colptr[i] + count;
        }

        // Build row indices and CSC-to-CSR mapping.
        let mut rowind = vec![0usize; nnz];
        let mut csc_to_csr_map = vec![(0usize, 0usize); nnz];
        let mut col_pos = colptr.clone();

        for row in 0..nrows {
            let row_start = csr.indptr[row];
            let row_end = csr.indptr[row + 1];
            for csr_idx in row_start..row_end {
                let col = csr.indices[csr_idx];
                let csc_idx = col_pos[col];
                rowind[csc_idx] = row;
                csc_to_csr_map[csc_idx] = (row, col);
                col_pos[col] += 1;
            }
        }

        Self {
            nrows,
            ncols,
            colptr,
            rowind,
            csc_to_csr_map,
        }
    }

    /// Number of non-zeros.
    #[inline]
    pub const fn nnz(&self) -> usize {
        self.rowind.len()
    }
}

/// Where one entry of a color's JVP result belongs in the CSC value buffer.
///
/// Precomputing the destination is what makes assembly a copy: after a sweep, each
/// entry reads `row` of the result vector and writes `csc_idx`, with no pattern
/// search per call.
#[derive(Debug, Clone, Copy)]
pub struct ColorScatterEntry {
    /// Position in the CSC value buffer to write.
    pub csc_idx: usize,
    /// Row of the JVP result to read.
    pub row: usize,
}

/// The split-out dense rows, filled by one reverse (VJP) pass each instead of
/// one forward JVP sweep per column.
///
/// A dense row (nnz ≥ `DENSE_ROW_MIN_NNZ`) would otherwise force the whole
/// matrix coloring toward one color per touched column. Splitting the rows out
/// lets the sparse remainder colour cheaply. See [`crate::row_extract`] for why
/// the rows share a tape rather than getting one each.
#[derive(Debug, Clone)]
pub struct AdjointDenseRows {
    /// Parent output rows these gradients reconstruct, in tape-element order.
    pub rows: Vec<usize>,
    /// Reverse-AD tape whose element `i` is row `rows[i]`.
    pub tape: AdjointTape,
    /// `(column, csc_idx)` scatter targets in the parent CSC, per row.
    pub entries: Vec<Vec<(usize, usize)>>,
}

impl AdjointDenseRows {
    /// Group `rows` onto `tape`, whose element `i` must be row `rows[i]`.
    ///
    /// # Panics
    /// Panics unless the tape recovers exactly one element per row, which is
    /// the correspondence every other method here relies on.
    pub fn new(rows: Vec<usize>, tape: AdjointTape, entries: Vec<Vec<(usize, usize)>>) -> Self {
        assert_eq!(
            tape.n_rows(),
            rows.len(),
            "the tape must hold one element per split row"
        );
        assert_eq!(entries.len(), rows.len(), "one scatter list per split row");
        Self {
            rows,
            tape,
            entries,
        }
    }

    /// Rows filled by reverse mode, so reverse passes per assembly.
    #[inline]
    pub const fn n_rows(&self) -> usize {
        self.rows.len()
    }

    /// Jacobian entries these rows account for.
    pub fn n_entries(&self) -> usize {
        self.entries.iter().map(Vec::len).sum()
    }

    /// Fill every row's `scatter` targets in `out` from one shared forward pass.
    ///
    /// `scatter` is indexed by tape element, so it is [`Self::entries`] or the
    /// same lists remapped to another CSC. `grad` doubles as the gradient
    /// buffer and must span the tape's state dimension.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble_into(
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
            for &(col, csc_idx) in targets {
                out[csc_idx] = grad[col];
            }
        }
    }
}

/// Where one [`JacobianData`]'s entries land in a consumer's value buffer.
///
/// Two consumers assemble the same artifact into differently indexed buffers: a
/// standalone one into the artifact's own CSC ([`JacobianData::own_layout`]), and
/// one whose buffer merges this pattern with another -- `CompiledModel`, which
/// carries `df/dy` and the mass matrix in a single array -- into the merged slots
/// ([`JacobianData::layout_in`]). Both are the same three scatter tables against
/// different slot indices, so the sweep takes a layout rather than each consumer
/// re-deriving the tables from the artifact's public parts.
#[derive(Debug, Clone)]
pub struct JacobianLayout {
    /// Per colour, the `(slot, row)` pairs that colour's sweep recovers.
    color_to_slots: Vec<Vec<ColorScatterEntry>>,
    /// Per dense-row group, per row of that group, `(column, slot)` targets.
    dense_row_slots: Vec<Vec<Vec<(usize, usize)>>>,
    /// The starting buffer for one assembly: constants in place, zero everywhere
    /// else. One `copy_from_slice` replaces a memset plus a scattered rewrite of
    /// most of what it just wrote.
    template: Vec<f64>,
    /// `(slot, value)` for every entry a compile pass proved constant, for
    /// consumers that report the table as well as writing it.
    constant_slots: Vec<(usize, f64)>,
}

impl JacobianLayout {
    /// Slots the target buffer must hold, which is its consumer's `nnz`.
    #[inline]
    pub const fn n_slots(&self) -> usize {
        self.template.len()
    }

    /// `(slot, value)` for every entry a compile pass proved constant. Split
    /// dense rows are absent: their own tape recomputes them.
    #[inline]
    pub fn constant_slots(&self) -> &[(usize, f64)] {
        &self.constant_slots
    }
}

/// Per-solve mutable buffers for one [`JacobianData`]'s assembly and actions.
///
/// Sized from the artifact, and carrying the lane width its tangent tape
/// supports, so a caller can neither mis-size a buffer nor drive the batched
/// sweep with a scalar one. Create one per solve; two concurrent assemblies of
/// the same artifact need two.
#[derive(Debug, Clone)]
pub struct JacobianScratch {
    /// The parent tangent tape, then in sequence every dense-row sub-tape, which
    /// reuse this buffer after the parent sweep.
    tape: Vec<f64>,
    /// Scalar colour seed, spanning [`JacobianData::seed_len`].
    seed: Vec<f64>,
    /// Adjoint `bar` buffer, parallel to the widest dense-row value tape.
    bar: Vec<f64>,
    /// Colours per batched tangent walk, or 1 for the scalar per-colour path.
    lanes: usize,
    /// Lane-minor tangent region; empty when `lanes == 1`.
    tangent_lanes: Vec<f64>,
    /// Lane-minor colour seeds; empty when `lanes == 1`.
    seed_lanes: Vec<f64>,
}

impl JacobianScratch {
    /// Buffers for `jac` at the widest lane width its tangent tape supports.
    pub fn new(jac: &JacobianData) -> Self {
        Self::with_lanes(jac, jac.lane_width())
    }

    /// Buffers pinned to the scalar one-colour-per-walk path: the reference the
    /// batched sweep is compared against, and what a tape that cannot batch gets
    /// from [`Self::new`] anyway.
    pub fn scalar(jac: &JacobianData) -> Self {
        Self::with_lanes(jac, 1)
    }

    fn with_lanes(jac: &JacobianData, lanes: usize) -> Self {
        let batched = lanes > 1;
        Self {
            tape: vec![0.0; jac.max_scratch_len()],
            seed: vec![0.0; jac.seed_len().max(1)],
            bar: vec![0.0; jac.max_adjoint_tape_len().max(1)],
            lanes,
            tangent_lanes: vec![
                0.0;
                if batched {
                    tangent_batch::tangent_scratch_len(jac.assembly_tape().ir(), lanes)
                } else {
                    0
                }
            ],
            seed_lanes: vec![0.0; if batched { jac.seed_dim * lanes } else { 0 }],
        }
    }

    /// Colours this scratch sweeps per walk of the tangent tape.
    #[inline]
    pub const fn lane_width(&self) -> usize {
        self.lanes
    }
}

/// Everything `finish_with_options` switches on, so the three constructors
/// differ by data rather than by a run of positional flags.
#[derive(Debug, Clone, Copy)]
struct BuildOptions {
    /// Fill wide scalar rows by reverse mode rather than coloring against
    /// them. Only a full-state build can, since it owns every column.
    split_dense_rows: bool,
    /// Sweep only the entries a compile pass could not prove constant, writing
    /// the rest from a table. Off is the reference path the exactness tests
    /// compare against.
    split_constants: bool,
}

/// What [`JacobianData::decide_coloring`] settled on: the coloring driving the
/// sweep, the entry mask it was chosen under, and the rows it handed to reverse
/// mode instead.
struct ColoringDecision {
    coloring: ColumnColoring,
    swept: Vec<bool>,
    dense_rows: Vec<AdjointDenseRows>,
    /// Rows `detect_dense_rows` nominated, adopted or not.
    n_candidate_rows: usize,
}

/// Prepared derivative artifact for one (expression, wrt) pair.
///
/// Holds the tangent-transformed expression, sparsity pattern, column coloring
/// and dense-row split. Immutable after build, so it is shared via `Arc` and read
/// by any number of concurrent assemblies; each supplies its own
/// [`JacobianScratch`] and a [`JacobianLayout`] naming its output slots.
///
/// The fields are private on purpose: every one of them is an input to
/// [`Self::assemble_into`], and the whole point of this type is that the sweep
/// they describe has one implementation. Read them through the accessors, add a
/// [`JacobianLayout`] for a new output ordering, and never rebuild the tables.
#[derive(Debug)]
pub struct JacobianData {
    /// Complete tangent-transformed expression with split-eval partitioning.
    /// Matrix-free Jacobian actions use this tape, including split dense rows.
    jvp_expr: Arc<CompiledExpr>,
    /// Parent tape used by colored assembly. When dense rows are split out,
    /// their outputs are zeroed so only the adjoint tapes evaluate them.
    assembly_jvp_expr: Arc<CompiledExpr>,
    /// CSR sparsity pattern of the derivative (no mass, no `cj`, pure `df/d·`).
    sparsity: SparsityPattern,
    /// CSC sparsity pattern (KLU/scipy ordering) of the same entries.
    csc: CscPattern,
    /// Column coloring that drives the per-call JVP sweep count.
    coloring: ColumnColoring,
    /// Rows of the derivative, matching the primal expression's output length.
    n_rows: usize,
    /// Columns of the derivative: states, or parameters, or the subset size.
    n_cols: usize,
    /// What this Jacobian differentiates with respect to.
    wrt: DiffTarget,
    /// For `States` with a column subset (algebraic blocks): global state
    /// index per local column. Empty means identity (column `i` = state `i`).
    col_to_global: Vec<usize>,
    /// Length of the seed buffer callers must supply. Full-state/params:
    /// `n_cols`. Subset: the full state dimension, tangent nodes index
    /// global state positions, so the `dy` slice must span all of them.
    seed_dim: usize,
    /// Dense rows split out of the column coloring, all on one shared tape.
    /// Empty unless a `new_wrt_states` build detected splittable dense rows AND
    /// the split strictly lowered the colour count.
    dense_rows: Vec<AdjointDenseRows>,
    /// Rows `detect_dense_rows` nominated, whether or not they were adopted.
    /// `n_candidate_rows` above `n_dense_rows()` is a declined split, which is
    /// correct but slower, and would otherwise be invisible.
    n_candidate_rows: usize,
    /// Per CSR entry of `sparsity`: whether a sweep must recover it. Always
    /// one flag per entry, all true when the build split nothing.
    swept_entries: Vec<bool>,
    /// `(csr_idx, value)` for every entry a compile pass proved constant,
    /// excluding split dense rows, whose own tape fills them. Kept in CSR order,
    /// since a layout is what maps them to one consumer's slots.
    constant_csr_entries: Vec<(usize, f64)>,
    /// Colours a batched walk of the tangent tape carries, or 1 for the scalar
    /// path. Decided once here rather than re-scanning the tape per scratch.
    lane_width: usize,
    /// This artifact's own-CSC layout, built on first ask and shared thereafter.
    /// Identical by construction for every standalone consumer, so minting one
    /// each would be the multi-copy-of-derived-tables shape this module exists to
    /// remove; lazy because a consumer that only assembles into a merged buffer
    /// never needs it.
    own_layout: OnceLock<JacobianLayout>,
    /// The COO triplet for that layout, on the same terms.
    own_coo: OnceLock<(Vec<usize>, Vec<usize>)>,
}

impl JacobianData {
    /// `df/dy` over the full state vector.
    ///
    /// Enables dense-row splitting: wide scalar rows are extracted into 1×n
    /// sub-Jacobians so the sparse remainder colours cheaply.
    pub fn new_wrt_states(arena: &Arena, root: NodeId, n_rows: usize, n_states: usize) -> Self {
        Self::new_wrt_states_inner(arena, root, n_rows, n_states, true)
    }

    /// As [`Self::new_wrt_states`], sweeping every entry rather than lifting the
    /// constant ones out. The reference path the exactness tests compare against.
    pub fn new_wrt_states_unsplit(
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
        let mut diff_arena = arena.clone();
        let tangent_root = tangent_wrt_states(&mut diff_arena, root);
        let sparsity = detect_sparsity_per_output(arena, root, n_rows, n_states);
        Self::finish_with_options(
            arena,
            root,
            diff_arena,
            tangent_root,
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

    /// `df/dp` over the full parameter vector. Sparsity is taken as dense
    /// (parameters are scalars; coloring degenerates to one sweep per
    /// column, matching unit-seed cost). Parameter sparsity detection is
    /// a later optimisation with no API impact.
    pub fn new_wrt_params(arena: &Arena, root: NodeId, n_rows: usize, n_params: usize) -> Self {
        let mut diff_arena = arena.clone();
        let tangent_root = tangent_wrt_params(&mut diff_arena, root);
        let sparsity = SparsityPattern::dense(n_rows, n_params);
        Self::finish_with_options(
            arena,
            root,
            diff_arena,
            tangent_root,
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

    /// `dg/dy_subset` for an algebraic block: differentiate w.r.t. the given
    /// global state indices only; columns are local (`0..subset.len()`).
    ///
    /// `subset` must be strictly ascending (sorted, no duplicates). The
    /// global-to-local column remap and `build_csr_to_csc_map` both rely on
    /// ascending CSR column order; an unsorted subset silently mis-maps entries.
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
        let active: HashSet<usize> = subset.iter().copied().collect();
        let mut diff_arena = arena.clone();
        let tangent_root = tangent_wrt_subset(&mut diff_arena, root, &active);
        let full = detect_sparsity_per_output(arena, root, n_rows, n_states);
        let filtered = filter_sparsity_columns(&full, subset);
        // Remap global columns to local positions so the pattern is (n_rows, subset.len()).
        let mut global_to_local = vec![usize::MAX; n_states];
        for (local, &g) in subset.iter().enumerate() {
            global_to_local[g] = local;
        }
        let mut local = SparsityPattern::new(n_rows, subset.len());
        local.indptr.clone_from(&filtered.indptr);
        local.indices = filtered
            .indices
            .iter()
            .map(|&c| global_to_local[c])
            .collect();
        Self::finish_with_options(
            arena,
            root,
            diff_arena,
            tangent_root,
            local,
            DiffTarget::States,
            subset.to_vec(),
            n_states,
            BuildOptions {
                // A dense row's adjoint tape reconstructs gradients over every
                // column, which a subset build does not own.
                split_dense_rows: false,
                // `classify_constant_entries` keys each coefficient by seed
                // index, i.e. global state position, while this pattern's columns
                // are local -- so lifting constants here needs the pattern
                // restated in seed space first. Not a reference-path choice.
                split_constants: false,
            },
        )
    }

    /// Assemble the artifact. `arena`/`root` are the ORIGINAL (pre-tangent)
    /// primal graph, dense-row splitting resolves and re-differentiates rows
    /// over them, so they must be threaded through even though the parent tape
    /// evaluates `tangent_root` in `diff_arena`.
    #[allow(clippy::too_many_arguments)]
    fn finish_with_options(
        arena: &Arena,
        root: NodeId,
        diff_arena: Arena,
        tangent_root: NodeId,
        sparsity: SparsityPattern,
        wrt: DiffTarget,
        col_to_global: Vec<usize>,
        seed_dim: usize,
        options: BuildOptions,
    ) -> Self {
        let (mut diff_arena, tangent_root) = simplify_pipeline(diff_arena, tangent_root);

        let csc = CscPattern::from_csr(&sparsity);
        let row_to_csc = build_row_to_csc_entries(&csc);

        // Classify on the simplified tangent tape, so a folded value follows
        // the same operator order the sweep would have executed.
        let (swept, mut constant_csr_entries) = if options.split_constants {
            classify_constant_entries(&diff_arena, tangent_root, &sparsity)
        } else {
            (vec![true; sparsity.nnz()], Vec::new())
        };

        // Adopt the reduced coloring only if every dense row extracts AND it strictly
        // beats the full coloring; else fall back. The sparsity is never altered.
        let ColoringDecision {
            coloring,
            swept: swept_entries,
            dense_rows,
            n_candidate_rows,
        } = Self::decide_coloring(
            arena,
            root,
            &sparsity,
            &row_to_csc,
            swept,
            options.split_dense_rows,
        );
        let skip_rows: Vec<usize> = dense_rows
            .iter()
            .flat_map(|split| split.rows.iter().copied())
            .collect();
        retain_outside_rows(&sparsity, &skip_rows, &mut constant_csr_entries);

        let jvp_ir = TypedIr::from_arena_split_eval(&diff_arena, tangent_root);
        let jvp_expr = Arc::new(CompiledExpr::from_ir(jvp_ir));
        // Masking is pruning, not correctness: nothing reads a split row's sweep
        // output, so an unmaskable row only leaves a value the assembly discards.
        let assembly_jvp_expr = match mask_scalar_rows(&mut diff_arena, tangent_root, &skip_rows) {
            Some(root) if root != tangent_root => {
                let masked_ir = TypedIr::from_arena_split_eval(&diff_arena, root);
                Arc::new(CompiledExpr::from_ir(masked_ir))
            },
            _ => Arc::clone(&jvp_expr),
        };
        let (n_rows, n_cols) = (sparsity.nrows, sparsity.ncols);
        let lane_width = decide_lane_width(assembly_jvp_expr.ir(), coloring.n_colors);
        Self {
            jvp_expr,
            assembly_jvp_expr,
            sparsity,
            csc,
            coloring,
            n_rows,
            n_cols,
            wrt,
            col_to_global,
            seed_dim,
            dense_rows,
            n_candidate_rows,
            swept_entries,
            constant_csr_entries,
            lane_width,
            own_layout: OnceLock::new(),
            own_coo: OnceLock::new(),
        }
    }

    /// Choose the column coloring, and with it which rows leave the sweep.
    ///
    /// `swept` is the mask both decisions read; an adopted split returns it
    /// narrowed, since a split row is nobody's sweep result and clearing its
    /// entries is what takes it out of the coloring.
    fn decide_coloring(
        arena: &Arena,
        root: NodeId,
        sparsity: &SparsityPattern,
        row_to_csc: &[Vec<(usize, usize)>],
        swept: Vec<bool>,
        split_dense_rows: bool,
    ) -> ColoringDecision {
        let no_candidates = |swept: Vec<bool>| ColoringDecision {
            coloring: color_columns_masked(sparsity, &swept),
            swept,
            dense_rows: Vec::new(),
            n_candidate_rows: 0,
        };
        if !split_dense_rows {
            return no_candidates(swept);
        }
        // Measure candidates by swept width: a row of known entries costs no
        // colours, so filling it by reverse mode would waste a pass.
        let widths = (0..sparsity.nrows)
            .map(|row| {
                swept[sparsity.indptr[row]..sparsity.indptr[row + 1]]
                    .iter()
                    .filter(|&&s| s)
                    .count()
            })
            .collect::<Vec<_>>();
        let candidates = detect_dense_rows(&widths);
        if candidates.is_empty() {
            return no_candidates(swept);
        }

        let mut narrowed = swept.clone();
        for &row in &candidates {
            narrowed[sparsity.indptr[row]..sparsity.indptr[row + 1]].fill(false);
        }
        let reduced = color_columns_masked(sparsity, &narrowed);
        let full = color_columns_masked(sparsity, &swept);
        let improves = reduced.n_colors < full.n_colors;
        // Built once and moved out on either decline path, since the adopt path
        // returns `narrowed` and `reduced` instead.
        let declined = ColoringDecision {
            coloring: full,
            swept,
            dense_rows: Vec::new(),
            n_candidate_rows: candidates.len(),
        };
        if !improves {
            return declined;
        }

        // Extraction is the expensive half of the decision, so it runs only once
        // the colouring is known to improve.
        let mut dense_rows = Vec::with_capacity(candidates.len().div_ceil(ROWS_PER_TAPE));
        for group in candidates.chunks(ROWS_PER_TAPE) {
            let Some(block) = extract_scalar_rows(arena, root, group) else {
                return declined;
            };
            let entries = block
                .rows
                .iter()
                .map(|&row| row_to_csc[row].clone())
                .collect();
            let tape = AdjointTape::new(&block.arena, block.root, sparsity.ncols);
            dense_rows.push(AdjointDenseRows::new(block.rows, tape, entries));
        }
        ColoringDecision {
            coloring: reduced,
            swept: narrowed,
            n_candidate_rows: candidates.len(),
            dense_rows,
        }
    }

    /// This artifact's entries laid out on its own CSC, which is what a standalone
    /// consumer (a `scipy` matrix, a Newton sub-block) assembles into.
    ///
    /// Built on first ask and shared thereafter: it is the same tables for every
    /// such consumer, so handing each one its own copy would be the duplication
    /// this module exists to remove.
    pub fn layout(&self) -> &JacobianLayout {
        self.own_layout.get_or_init(|| self.layout_in(&self.csc))
    }

    /// This artifact's entries laid out on `buffer`, a pattern whose slots it
    /// shares with another -- the merged `df/dy` + mass CSC a model assembles
    /// into. Slots this artifact has no entry for keep the zero the template
    /// writes, so the consumer's own pass can fold into them afterwards.
    ///
    /// # Panics
    /// Panics unless `buffer` is a superset of [`Self::sparsity`], since every
    /// entry of this artifact must have a slot to land in.
    #[must_use]
    pub fn layout_in(&self, buffer: &CscPattern) -> JacobianLayout {
        let row_to_slots = &build_row_to_csc_entries(buffer);
        let n_slots = buffer.nnz();
        let csr_to_slot = build_csr_to_csc_map(&self.sparsity, row_to_slots);
        let color_to_slots = build_color_scatter_entries(
            &self.sparsity,
            &csr_to_slot,
            &self.coloring,
            &self.swept_entries,
        );
        // Each dense row scatters grad[col] -> slot; a merged buffer numbers its
        // slots differently from this artifact's own CSC, so take its row map.
        let dense_row_slots = self
            .dense_rows
            .iter()
            .map(|split| {
                split
                    .rows
                    .iter()
                    .map(|&row| row_to_slots[row].clone())
                    .collect()
            })
            .collect();
        let constant_slots = map_entries_to_csc(&self.constant_csr_entries, &csr_to_slot);
        let mut template = vec![0.0; n_slots];
        for &(slot, value) in &constant_slots {
            template[slot] = value;
        }
        // A split row is nobody's sweep result: several same-coloured columns land
        // on it, so bucketing one would alias their sums. `decide_coloring` clears
        // those entries from `swept_entries`, and this is that holding.
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
            constant_slots,
        }
    }

    /// COO `(rows, cols)` in the order [`Self::assemble_into`] writes a
    /// [`layout`](Self::layout) buffer, with each local column reported as the
    /// global state index it stands for.
    ///
    /// The pair a consumer needs when it wants the triplet form rather than this
    /// artifact's CSC. Built on first ask and shared thereafter.
    pub fn coo_global_indices(&self) -> (&[usize], &[usize]) {
        let (rows, cols) = self.own_coo.get_or_init(|| {
            self.csc
                .csc_to_csr_map
                .iter()
                .map(|&(row, col)| (row, self.global_column(col)))
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

    /// Column coloring that decides the per-call sweep count. With a dense-row
    /// split adopted it is the reduced coloring, so it does not cover every row.
    #[inline]
    pub const fn coloring(&self) -> &ColumnColoring {
        &self.coloring
    }

    /// Sweeps of the tangent tape one assembly costs, before batching.
    #[inline]
    pub const fn n_colors(&self) -> usize {
        self.coloring.n_colors
    }

    /// What this Jacobian differentiates with respect to.
    #[inline]
    pub const fn wrt(&self) -> DiffTarget {
        self.wrt
    }

    /// Rows of the derivative, matching the primal expression's output length.
    #[inline]
    pub const fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Columns of the derivative: states, parameters, or the subset size.
    #[inline]
    pub const fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// `(csr_idx, value)` for every entry a compile pass proved constant, in this
    /// artifact's CSR order. Split dense rows are absent.
    #[inline]
    pub fn constant_csr_entries(&self) -> &[(usize, f64)] {
        &self.constant_csr_entries
    }

    /// The rows filled by reverse mode, grouped by shared tape.
    #[inline]
    pub fn dense_rows(&self) -> &[AdjointDenseRows] {
        &self.dense_rows
    }

    /// Rows `detect_dense_rows` nominated, adopted or not.
    #[inline]
    pub const fn n_candidate_rows(&self) -> usize {
        self.n_candidate_rows
    }

    /// The unpruned tangent tape, which every matrix-free action walks.
    #[inline]
    pub const fn action_tape(&self) -> &Arc<CompiledExpr> {
        &self.jvp_expr
    }

    /// The tape colored assembly walks: [`action_tape`](Self::action_tape) with
    /// the split dense rows' outputs masked off, or the same tape when no row
    /// could be masked.
    #[inline]
    pub const fn assembly_tape(&self) -> &Arc<CompiledExpr> {
        &self.assembly_jvp_expr
    }

    /// Colours a batched walk of the tangent tape carries, or 1 for the scalar
    /// per-colour path. Decided once at build time.
    #[inline]
    pub const fn lane_width(&self) -> usize {
        self.lane_width
    }

    /// Length of the seed buffer [`JacobianScratch`] allocates: `n_cols` for a
    /// full build, the whole state dimension for a subset, whose tangent nodes
    /// index global state positions.
    const fn seed_len(&self) -> usize {
        self.seed_dim
    }

    /// Scratch length covering the parent tape and every dense-row value tape.
    /// The tapes run in sequence after the parent sweep and reuse the same
    /// buffer, so [`JacobianScratch`] sizes to this maximum.
    fn max_scratch_len(&self) -> usize {
        self.jvp_expr
            .scratch_len()
            .max(self.assembly_jvp_expr.scratch_len())
            .max(self.max_adjoint_tape_len())
    }

    /// Length of the adjoint `bar` buffer: the largest dense-row value tape,
    /// 0 when no rows were split.
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
        self.dense_rows
            .iter()
            .map(AdjointDenseRows::n_entries)
            .sum()
    }

    /// Instructions across the split rows' tapes: what the split costs in
    /// compiled memory, against the sweeps it removes.
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
    /// The constant template, then the colour sweep at `scratch`'s lane width,
    /// then one reverse pass per split dense row. Every slot the layout names is
    /// written -- from the template, one colour's scatter, or a dense row -- so
    /// callers need not zero `out`; nothing outside those slots is touched, which
    /// is what lets a merged buffer carry another pattern's entries through the
    /// call and fold its own term in afterwards.
    ///
    /// No mass term and no `cj`: those are the solver's, and keeping them out is
    /// why the three consumers can share this one driver.
    ///
    /// # Panics
    /// Panics if `out` is shorter than [`JacobianLayout::n_slots`].
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
        // A layout built from another artifact would scatter this one's rows into
        // the wrong slots, which the sweep itself cannot notice.
        debug_assert_eq!(
            layout.color_to_slots.len(),
            self.coloring.n_colors,
            "layout was built for a different jacobian"
        );
        debug_assert_eq!(
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

        // Runs after the parent sweep, so `tape` is free as the value tape. Lands
        // before any consumer postpass: a mass term subtracts on diagonal slots,
        // so it must follow ALL derivative fills.
        for (split, slots) in self.dense_rows.iter().zip(&layout.dense_row_slots) {
            let JacobianScratch {
                tape, bar, seed, ..
            } = scratch;
            split.assemble_into(tape, bar, seed, t, y, y_dot, inputs, slots, out);
        }
    }

    /// One tangent-tape walk per colour. Nothing reads the primal section when no
    /// colour sweeps, which is the shape of a Jacobian that folded entirely.
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
        if self.coloring.n_colors == 0 {
            return;
        }
        self.assembly_jvp_expr
            .run_primal_section(&mut scratch.tape, t, y, y_dot, inputs);
        for color in 0..self.coloring.n_colors {
            self.sweep_one_color_scalar(scratch, layout, color, out);
        }
    }

    /// Sweep colours in blocks, one tangent-tape walk per block. Lane `l` carries
    /// colour `block_start + l`.
    ///
    /// Block widths come from [`Self::block_width`], so a tail shorter than the
    /// scratch's lane count costs a narrower walk rather than a padded full-width
    /// one. Per-lane arithmetic does not depend on the width, so which widths a
    /// sweep happens to use cannot change the assembled values.
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
        let n_colors = self.coloring.n_colors;
        if n_colors == 0 {
            return;
        }
        self.assembly_jvp_expr
            .run_primal_section(&mut scratch.tape, t, y, y_dot, inputs);

        // Undoing a block's seeds beats memsetting the lane region: each column
        // is seeded once per assembly, so this is O(n_states), not O(n * blocks).
        let mut block = 0;
        while block < n_colors {
            let stride = Self::block_width(scratch.lanes, n_colors - block);
            if stride == 1 {
                // One colour left: a scalar walk beats a vector one carrying it.
                self.sweep_one_color_scalar(scratch, layout, block, out);
                block += 1;
                continue;
            }
            // A tail block may carry fewer colours than its stride; the surplus
            // lanes are never seeded, so they stay zero and scatter nothing.
            let carried = stride.min(n_colors - block);
            self.write_seed_lanes(scratch, block, carried, stride, 1.0);

            let result = self.run_tangent_lanes(scratch, stride);

            for lane in 0..carried {
                for entry in &layout.color_to_slots[block + lane] {
                    out[entry.csc_idx] = result[entry.row * stride + lane];
                }
            }
            self.write_seed_lanes(scratch, block, carried, stride, 0.0);
            block += carried;
        }
    }

    /// Colours a tail must carry before a part-empty vector block beats that many
    /// scalar walks.
    ///
    /// A `K`-lane walk costs `ratio` scalar walks, where `ratio < K` because the
    /// operator gather is paid once and the lanes go through SIMD. Padding `r`
    /// colours into one block therefore wins exactly when `ratio < r`. Measured
    /// over deliberately different tangent tapes -- sparse-matmul (FV-like),
    /// shallow and deep elementwise, and the SPMe/DFN fixtures -- `ratio` at
    /// `K = 4` spans 1.53 to 2.33, so `r >= 3` wins everywhere while `r == 2` is
    /// a loss on the gather-light half. Tapes that would amortise worse still
    /// cannot reach here: `is_batchable` keeps them on the scalar path entirely.
    const MIN_PADDED_TAIL: usize = 3;

    /// Width of the next block: the widest monomorphised width the remaining
    /// colours fill outright, else the narrowest vector width once the tail is
    /// long enough to pay for the empty lanes, else a scalar walk. Never exceeds
    /// `lanes`, which is what the scratch buffers were sized for.
    fn block_width(lanes: usize, remaining: usize) -> usize {
        let usable = || {
            tangent_batch::SUPPORTED_LANES
                .into_iter()
                .filter(|&w| w <= lanes)
        };
        if let Some(width) = usable().find(|&width| width <= remaining) {
            return width;
        }
        match usable().min() {
            Some(narrowest) if remaining >= Self::MIN_PADDED_TAIL => narrowest,
            _ => 1,
        }
    }

    /// One colour through the scalar tangent section, reusing the primal region
    /// the batched sweep already filled. Only ever the tail of the block loop, so
    /// no later block reads the tangent slots this overwrites.
    fn sweep_one_color_scalar(
        &self,
        scratch: &mut JacobianScratch,
        layout: &JacobianLayout,
        color: usize,
        out: &mut [f64],
    ) {
        let JacobianScratch { tape, seed, .. } = scratch;
        seed.fill(0.0);
        for &col in self.coloring.columns_with_color(color) {
            seed[self.global_column(col)] = 1.0;
        }
        let result = self
            .assembly_jvp_expr
            .run_tangent_section(tape, &self.seed_as_tangent(seed));
        for entry in &layout.color_to_slots[color] {
            out[entry.csc_idx] = result[entry.row];
        }
    }

    /// Write `value` into the seed positions of the `carried` colours starting at
    /// `block`, at that block's lane `stride`, leaving every other slot untouched.
    /// Called with 1.0 before the sweep and 0.0 after, which is what keeps
    /// `seed_lanes` zero between blocks without a memset.
    fn write_seed_lanes(
        &self,
        scratch: &mut JacobianScratch,
        block: usize,
        carried: usize,
        stride: usize,
        value: f64,
    ) {
        for lane in 0..carried {
            for &col in self.coloring.columns_with_color(block + lane) {
                scratch.seed_lanes[self.global_column(col) * stride + lane] = value;
            }
        }
    }

    /// Dispatch the batched tangent sweep to its monomorphised lane width.
    fn run_tangent_lanes<'s>(&self, scratch: &'s mut JacobianScratch, lanes: usize) -> &'s [f64] {
        let ir = self.assembly_jvp_expr.ir();
        match lanes {
            8 => tangent_batch::run_tangent_batch::<8>(
                ir,
                &scratch.tape,
                &mut scratch.tangent_lanes,
                &scratch.seed_lanes,
            ),
            4 => tangent_batch::run_tangent_batch::<4>(
                ir,
                &scratch.tape,
                &mut scratch.tangent_lanes,
                &scratch.seed_lanes,
            ),
            other => unreachable!("unsupported jacobian lane width {other}"),
        }
    }

    /// Point the seed buffer at whichever tangent input this artifact seeds, the
    /// one place [`DiffTarget`] decides anything at assembly time.
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
    /// `v` is one value per column of this Jacobian. A subset build's columns are
    /// local while its tangent nodes index global state positions, so the seed is
    /// scattered through `col_to_global` first; an artifact owning every column
    /// seeds from `v` directly, with no copy.
    ///
    /// # Panics
    /// Panics if `v` is shorter than [`Self::n_cols`].
    #[allow(clippy::too_many_arguments)]
    pub fn action_into(
        &self,
        scratch: &mut JacobianScratch,
        t: f64,
        y: &[f64],
        y_dot: &[f64],
        inputs: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) {
        assert!(
            v.len() >= self.n_cols,
            "jacobian tangent buffer too small: need {}, got {}",
            self.n_cols,
            v.len()
        );
        if !self.col_to_global.is_empty() {
            scratch.seed.fill(0.0);
            for (col, &seed) in v.iter().enumerate().take(self.n_cols) {
                scratch.seed[self.global_column(col)] = seed;
            }
        }
        let JacobianScratch { tape, seed, .. } = scratch;
        let seed = if self.col_to_global.is_empty() {
            v
        } else {
            seed
        };
        let result =
            self.jvp_expr
                .eval_with_tangent(tape, t, y, y_dot, inputs, &self.seed_as_tangent(seed));
        out[..result.len()].copy_from_slice(result);
    }
}

/// Build a row→[(col, `csc_idx`)] lookup table from a `CscPattern`.
pub fn build_row_to_csc_entries(csc: &CscPattern) -> Vec<Vec<(usize, usize)>> {
    let mut row_to_csc_entries = vec![Vec::new(); csc.nrows];
    for (csc_idx, &(row, col)) in csc.csc_to_csr_map.iter().enumerate() {
        row_to_csc_entries[row].push((col, csc_idx));
    }
    row_to_csc_entries
}

/// Map each CSR non-zero to its CSC slot index.
pub fn build_csr_to_csc_map(
    csr: &SparsityPattern,
    row_to_csc_entries: &[Vec<(usize, usize)>],
) -> Vec<usize> {
    let mut map = vec![0usize; csr.nnz()];
    for (row, csc_entries) in row_to_csc_entries.iter().enumerate().take(csr.nrows) {
        let row_start = csr.indptr[row];
        let row_end = csr.indptr[row + 1];
        let mut csc_pos = 0usize;
        for (csr_idx, map_entry) in map.iter_mut().enumerate().take(row_end).skip(row_start) {
            let col = csr.indices[csr_idx];
            while csc_entries[csc_pos].0 < col {
                csc_pos += 1;
            }
            assert_eq!(
                csc_entries[csc_pos].0, col,
                "missing CSC slot for CSR entry ({row}, {col})"
            );
            *map_entry = csc_entries[csc_pos].1;
        }
    }
    map
}

/// Build per-color scatter entries from a sparsity pattern, CSR→CSC map, and coloring.
///
/// Only entries `swept` marks are bucketed. An entry a sweep does not recover
/// MUST be left out: for a split dense row several same-colour columns land on
/// it and bucketing would alias their sums, and for a constant entry the sweep
/// value is polluted by design.
fn build_color_scatter_entries(
    jac_y_sparsity: &SparsityPattern,
    jac_y_csr_to_csc_map: &[usize],
    coloring: &ColumnColoring,
    swept: &[bool],
) -> Vec<Vec<ColorScatterEntry>> {
    let mut color_to_csc_entries = vec![Vec::new(); coloring.n_colors];
    for (csr_idx, row) in jac_y_sparsity.entry_rows().into_iter().enumerate() {
        if !swept[csr_idx] {
            continue;
        }
        let color = coloring.colors[jac_y_sparsity.indices[csr_idx]];
        color_to_csc_entries[color].push(ColorScatterEntry {
            csc_idx: jac_y_csr_to_csc_map[csr_idx],
            row,
        });
    }
    color_to_csc_entries
}

/// Remap `(csr_idx, value)` entries into a CSC ordering.
fn map_entries_to_csc(entries: &[(usize, f64)], csr_to_csc: &[usize]) -> Vec<(usize, f64)> {
    entries
        .iter()
        .map(|&(csr_idx, value)| (csr_to_csc[csr_idx], value))
        .collect()
}

/// Keep only the entries outside `rows`. A split dense row is filled wholesale
/// by its own tape, so a table write there would be overwritten anyway.
fn retain_outside_rows(pattern: &SparsityPattern, rows: &[usize], entries: &mut Vec<(usize, f64)>) {
    if rows.is_empty() || entries.is_empty() {
        return;
    }
    let mut dropped = vec![false; pattern.nnz()];
    for &row in rows {
        dropped[pattern.indptr[row]..pattern.indptr[row + 1]].fill(true);
    }
    entries.retain(|&(csr_idx, _)| !dropped[csr_idx]);
}

/// Filter a sparsity pattern to retain only the columns in `allowed_columns`.
fn filter_sparsity_columns(
    pattern: &SparsityPattern,
    allowed_columns: &[usize],
) -> SparsityPattern {
    let mut filtered = SparsityPattern::new(pattern.nrows, pattern.ncols);
    let mut allowed = vec![false; pattern.ncols];
    for &col in allowed_columns {
        allowed[col] = true;
    }

    for row in 0..pattern.nrows {
        let row_start = pattern.indptr[row];
        let row_end = pattern.indptr[row + 1];
        filtered.indptr[row] = filtered.indices.len();
        for &col in &pattern.indices[row_start..row_end] {
            if allowed[col] {
                filtered.indices.push(col);
            }
        }
    }
    filtered.indptr[pattern.nrows] = filtered.indices.len();
    filtered
}

/// Rows this wide force column coloring toward one color per touched column
/// (max-row-nnz is the coloring lower bound), so they are split out.
pub(crate) const DENSE_ROW_MIN_NNZ: usize = 16;
/// Safety cap on reverse passes, so a pathological pattern cannot make the build
/// allocate an unbounded number of adjoint tapes.
pub(crate) const MAX_DENSE_ROWS: usize = 1024;
/// Rows per shared adjoint tape.
///
/// Every row of a group walks its group's whole tape backwards, so a group
/// trades assembly time for compiled memory. Measured on the 12x12 pouch cell
/// (one assembly / total tape): 1 row 66 ms / 7.8M instrs, 4 rows 58 ms / 2.4M,
/// 16 rows 85 ms / 1.0M, all 144 rows 215 ms / 283k. Four is the knee.
pub(crate) const ROWS_PER_TAPE: usize = 4;

/// The `k` widest rows, with `k` chosen to minimise what the Jacobian costs.
///
/// Splitting the `k` widest leaves a colouring bound of `nnz(wide[k])` and buys it
/// for `k` reverse passes, so the sweep count is `k + nnz(wide[k])`, against
/// `nnz(wide[0])` unsplit. Taking the minimum covers both a lone outlier row and a
/// plateau of equally wide rows, where splitting part of it lowers nothing at all
/// and only splitting the whole plateau pays -- a 2D current collector puts one
/// dense row per collector node, so the plateau is the common case, not the
/// pathological one. Past `nnz(wide[0])` passes a split costs more than colouring
/// everything, so the objective caps itself and `MAX_DENSE_ROWS` only bounds how
/// many tapes a build will construct.
///
/// `row_widths` counts only the entries a sweep must recover, so a row of
/// known entries costs no colours and is never worth a reverse pass.
pub(crate) fn detect_dense_rows(row_widths: &[usize]) -> Vec<usize> {
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
fn mask_scalar_rows(arena: &mut Arena, root: NodeId, rows: &[usize]) -> Option<NodeId> {
    if rows.is_empty() {
        return Some(root);
    }
    if !rows.windows(2).all(|pair| pair[0] < pair[1]) {
        return None;
    }
    let mut lens = NodeMap::new(arena.len());
    mask_scalar_rows_inner(arena, root, rows, &mut lens)
}

fn mask_scalar_rows_inner(
    arena: &mut Arena,
    id: NodeId,
    rows: &[usize],
    lens: &mut NodeMap<usize>,
) -> Option<NodeId> {
    if rows.is_empty() {
        return Some(id);
    }
    match arena.get(id).clone() {
        Node::Concat(children) => {
            let mut offset = 0;
            let mut masked = Vec::with_capacity(children.len());
            for child in children {
                let len = node_len(arena, child, lens);
                let start = rows.partition_point(|&row| row < offset);
                let end = rows.partition_point(|&row| row < offset + len);
                let local_rows: Vec<usize> =
                    rows[start..end].iter().map(|&row| row - offset).collect();
                masked.push(mask_scalar_rows_inner(arena, child, &local_rows, lens)?);
                offset += len;
            }
            if rows.iter().any(|&row| row >= offset) {
                return None;
            }
            Some(arena.alloc(Node::Concat(masked)))
        },
        _ if rows == [0] && node_len(arena, id, lens) == 1 => Some(arena.alloc(Node::Scalar(0.0))),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::arena::Arena;
    use crate::coloring::color_columns;
    use crate::node::Node;

    /// f(y, p) = [y0*y1 + p0, sin(y0) * p1], mixed state and parameter dependencies.
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
        // Expand CSC data into a dense matrix for assertion.
        let mut dense = vec![vec![0.0; jac.n_cols()]; jac.n_rows()];
        for (col, span) in jac.csc().colptr.windows(2).enumerate() {
            for k in span[0]..span[1] {
                dense[jac.csc().rowind[k]][col] = data[k];
            }
        }
        dense
    }

    /// Nonlinear banded expression over `n` states with half-width `w`: enough
    /// colours to batch, and a tangent tape of nothing but batchable
    /// instructions. Shared with `model`'s tests, whose batching fixtures must be
    /// the same shape as these to be comparable.
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

    fn assemble_with(jac: &JacobianData, scratch: &mut JacobianScratch, y: &[f64]) -> Vec<f64> {
        let layout = jac.layout();
        let mut out = vec![f64::NAN; layout.n_slots()];
        jac.assemble_into(scratch, layout, 0.5, y, &[], &[], &mut out);
        out
    }

    /// The batched sweep is the scalar one's arithmetic re-laned, so it must agree
    /// bit for bit. With one driver this pins every consumer at once, not just the
    /// one a test picks.
    #[allow(clippy::float_cmp)] // exact equality is the point
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
        let n_rows = CompiledExpr::new(&arena, root).output_len();
        let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n);
        assert_batched_matches_scalar(&jac, &batching_probe_states(n));
    }

    /// A layout onto a wider buffer is a remap, not a second derivation: the
    /// derivative values land unchanged, and the slots this pattern has no entry
    /// for are left for their own owner.
    #[test]
    #[allow(clippy::float_cmp)] // exact equality is the point
    fn a_merged_layout_only_renumbers_the_slots() {
        let (n, w) = (32, 2);
        let (arena, root) = banded_nonlinear(n, w);
        let n_rows = CompiledExpr::new(&arena, root).output_len();
        let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n);

        // A dense row per state, as a mass matrix that couples everything would
        // give: every merged row is wider than the derivative's, so no slot
        // number survives by luck.
        let mut extra = SparsityPattern::new(n_rows, n);
        extra.indptr = (0..=n_rows).map(|row| row * n).collect();
        extra.indices = (0..n_rows).flat_map(|_| 0..n).collect();
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

        // Every derivative entry, matched (row, col) to (row, col) across the two
        // slot numberings.
        let mut own_by_entry = std::collections::HashMap::new();
        for (slot, &(row, col)) in jac.csc().csc_to_csr_map.iter().enumerate() {
            own_by_entry.insert((row, col), own[slot]);
        }
        let mut covered = 0;
        for (slot, &(row, col)) in merged_csc.csc_to_csr_map.iter().enumerate() {
            match own_by_entry.get(&(row, col)) {
                Some(&value) => {
                    assert_eq!(wide[slot], value, "entry ({row}, {col}) moved value");
                    covered += 1;
                },
                // A slot only the other pattern owns: zeroed, never written.
                None => assert_eq!(wide[slot], 0.0, "slot ({row}, {col}) is not ours to write"),
            }
        }
        assert_eq!(covered, jac.nnz(), "every derivative entry must be placed");
    }

    /// A subset build seeds global state positions through `col_to_global`, so its
    /// batched lane seeds must land there too -- the one place the two sweeps
    /// could disagree about which column a lane carries.
    #[test]
    #[allow(clippy::float_cmp)] // exact equality is the point
    fn a_subset_build_batches_onto_the_same_values() {
        let (n, w) = (64, 3);
        let (arena, root) = banded_nonlinear(n, w);
        let n_rows = CompiledExpr::new(&arena, root).output_len();
        let subset: Vec<usize> = (0..n).collect();
        let jac = JacobianData::new_wrt_state_subset(&arena, root, n_rows, n, &subset);
        let y = batching_probe_states(n);
        assert_batched_matches_scalar(&jac, &y);

        // And the same derivative the full-state build produces.
        let subset_values = assemble_with(&jac, &mut JacobianScratch::new(&jac), &y);
        let full = JacobianData::new_wrt_states(&arena, root, n_rows, n);
        let reference = assemble_with(&full, &mut JacobianScratch::new(&full), &y);
        assert_eq!(jac.csc().csc_to_csr_map, full.csc().csc_to_csr_map);
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
        // Single output row over 2 states: shape (1, 2)
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let root = arena.alloc(Node::Mul(y0, y1));
        let jac = JacobianData::new_wrt_states(&arena, root, 1, 2);
        assert_eq!((jac.n_rows(), jac.n_cols()), (1, 2));
    }

    #[test]
    fn jacobian_wrt_state_subset_analytic() {
        // f(y) = [y0*y2 + y1], so df/dy = [y2, 1, y0]. subset = [0, 2] maps local
        // cols 0,1 to global 0,2; at y=[2,3,5] df/dy1 = 1 must not appear.
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let y2 = arena.alloc(Node::StateVector { start: 2, end: 3 });
        let prod = arena.alloc(Node::Mul(y0, y2));
        let root = arena.alloc(Node::Add(prod, y1));

        let jac = JacobianData::new_wrt_state_subset(&arena, root, 1, 3, &[0, 2]);

        // Shape checks.
        assert_eq!((jac.n_rows(), jac.n_cols()), (1, 2));
        // seed_len must equal full state dimension, not subset size.
        assert_eq!(jac.seed_len(), 3);

        let dense = assemble_full(&jac, 0.0, &[2.0, 3.0, 5.0], &[]);
        // col 0 (global 0): df/dy0 = y2 = 5
        assert!(
            (dense[0][0] - 5.0).abs() < 1e-12,
            "df/dy0 wrong: {}",
            dense[0][0]
        );
        // col 1 (global 2): df/dy2 = y0 = 2
        assert!(
            (dense[0][1] - 2.0).abs() < 1e-12,
            "df/dy2 wrong: {}",
            dense[0][1]
        );
        // df/dy1 = 1 must not appear in the 2-column result.
        assert_eq!(
            jac.n_cols(),
            2,
            "subset Jacobian must have exactly 2 columns"
        );
    }

    #[test]
    fn jacobian_wrt_params_param_independent_row() {
        // f = [y0*y1, y0*p0]: row 0 depends on no parameter, so df/dp must give
        // (0,0) == 0 and (1,0) == y0 with no read past a collapsed tangent tape.
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
        // f = concat(const_vec[1,2], y0): when the leading constant child collapses
        // in the tangent tape, y0's derivative must stay in row 2, not shift to 0.
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
    #[should_panic(expected = "subset must be strictly ascending")]
    fn jacobian_wrt_state_subset_rejects_unsorted() {
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let root = arena.alloc(Node::Add(y0, y1));
        // Reversed subset, must panic.
        let _ = JacobianData::new_wrt_state_subset(&arena, root, 1, 2, &[1, 0]);
    }

    /// `nrows` x `ncols` pattern: row 0 fully dense (`ncols` nnz), every
    /// other row carries a single diagonal-ish nonzero.
    fn make_pattern_with_dense_row(ncols: usize, nrows: usize) -> SparsityPattern {
        let mut pattern = SparsityPattern::new(nrows, ncols);
        let mut idx = 0;
        for row in 0..nrows {
            pattern.indptr[row] = idx;
            if row == 0 {
                pattern.indices.extend(0..ncols);
                idx += ncols;
            } else {
                pattern.indices.push((row - 1) % ncols);
                idx += 1;
            }
        }
        pattern.indptr[nrows] = idx;
        pattern
    }

    /// `n` x `n` identity-like pattern: one nonzero per row, on the diagonal.
    fn make_diagonal_pattern_local(n: usize) -> SparsityPattern {
        let mut pattern = SparsityPattern::new(n, n);
        for row in 0..n {
            pattern.indptr[row] = row;
            pattern.indices.push(row);
        }
        pattern.indptr[n] = n;
        pattern
    }

    /// `ncols` x `ncols` pattern with the first `n_dense` rows fully dense
    /// and the remainder carrying a single nonzero each.
    fn make_pattern_with_n_dense_rows(ncols: usize, n_dense: usize) -> SparsityPattern {
        let nrows = ncols;
        let mut pattern = SparsityPattern::new(nrows, ncols);
        let mut idx = 0;
        for row in 0..nrows {
            pattern.indptr[row] = idx;
            if row < n_dense {
                pattern.indices.extend(0..ncols);
                idx += ncols;
            } else {
                pattern.indices.push(row % ncols);
                idx += 1;
            }
        }
        pattern.indptr[nrows] = idx;
        pattern
    }

    #[test]
    fn detect_dense_rows_flags_wide_rows_only() {
        // 20 cols; row 0 dense (20 nnz), rows 1..19 diagonal-ish (1 nnz)
        let pattern = make_pattern_with_dense_row(20, 20);
        assert_eq!(detect_dense_rows(&pattern.row_widths()), vec![0]);
        // all-sparse pattern → no detection
        let diag = make_diagonal_pattern_local(20);
        assert!(detect_dense_rows(&diag.row_widths()).is_empty());
    }

    /// A 2D current collector puts one dense row per collector node, so the
    /// widest rows are a plateau of equal widths rather than a lone outlier.
    /// Splitting part of a plateau lowers nothing, so a heuristic that stops at a
    /// fixed handful splits none of it and leaves colouring at one colour per
    /// column -- the shape that took a 2818-state pouch model to 2816 colours.
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
            // and the split has to actually pay for itself
            let unsplit = widths.iter().copied().max().unwrap();
            let remaining = (n_dense..pattern.nrows)
                .map(|row| widths[row])
                .max()
                .unwrap_or(0);
            assert!(
                split.len() + remaining < unsplit,
                "{n_dense} reverse passes + {remaining} colours must beat {unsplit}"
            );
        }
    }

    /// Splitting cannot pay when every row is equally dense: `k` reverse passes
    /// leave the bound untouched, so plain colouring stays the cheaper option.
    #[test]
    fn detect_dense_rows_declines_when_no_split_pays() {
        let all_dense = make_pattern_with_n_dense_rows(20, 20);
        assert!(detect_dense_rows(&all_dense.row_widths()).is_empty());
    }

    /// The number of reverse passes stays bounded however pathological the
    /// pattern, so a build cannot be made to allocate unbounded adjoint tapes.
    #[test]
    fn detect_dense_rows_respects_the_tape_cap() {
        let ncols = MAX_DENSE_ROWS + 64;
        let pattern = make_pattern_with_n_dense_rows(ncols, MAX_DENSE_ROWS + 32);
        assert!(detect_dense_rows(&pattern.row_widths()).len() <= MAX_DENSE_ROWS);
    }

    #[test]
    fn detect_dense_rows_splits_outlier_among_many_wide_rows() {
        // The outlier alone sets the coloring bound, so it must still be split
        // when the count of merely-wide rows runs far past MAX_DENSE_ROWS.
        let ncols = 64;
        let nrows = 40;
        let mut pattern = SparsityPattern::new(nrows, ncols);
        let mut idx = 0;
        for row in 0..nrows {
            pattern.indptr[row] = idx;
            let width = if row == 7 { ncols } else { DENSE_ROW_MIN_NNZ };
            pattern
                .indices
                .extend((0..width).map(|c| (row + c) % ncols));
            idx += width;
        }
        pattern.indptr[nrows] = idx;

        assert_eq!(detect_dense_rows(&pattern.row_widths()), vec![7]);
        // and dropping it genuinely lowers the coloring bound
        let full = color_columns(&pattern);
        let mut swept = vec![true; pattern.nnz()];
        swept[pattern.indptr[7]..pattern.indptr[8]].fill(false);
        let reduced = color_columns_masked(&pattern, &swept);
        assert!(
            reduced.n_colors < full.n_colors,
            "reduced {} should beat full {}",
            reduced.n_colors,
            full.n_colors
        );
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
        let ir = TypedIr::from_arena(&arena, masked);
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
        let ones = arena.alloc(Node::SparseMatrix(Box::new(
            crate::node::CsrData::try_new(
                vec![0, n],
                (0..n).collect(),
                vec![1.0; n],
                crate::node::Shape::matrix(1, n),
            )
            .expect("valid row matrix"),
        )));
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
        // This fixture has no Conditional, so raw and common tape lengths
        // coincide; the assertion is about dense-row pruning, not branches.
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
    fn unsplit_jacobian_shares_full_and_assembly_jvp() {
        let mut arena = Arena::new();
        let root = toy(&mut arena);
        let jac = JacobianData::new_wrt_states(&arena, root, 2, 2);
        assert!(Arc::ptr_eq(jac.action_tape(), jac.assembly_tape()));
    }
}
