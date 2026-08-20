//! `CompiledModel` for DAE system evaluation.
//!
//! `PyBaMM` models are DAE systems of the form `M * y' = f(t, y)` where `M` is
//! a constant mass matrix (often singular), `f(t, y)` is the right-hand side,
//! and `y` is the state vector. Newton iteration computes `J = df/dy - cj * M`.
//!
//! `CompiledModel` holds the primal evaluator, symbolic `df/dy`, mass matrix,
//! and sparsity/coloring information for efficient Jacobian assembly. It is
//! immutable, so evaluation needs a `Workspace` alongside it; `ModelEvaluator`
//! pairs the two into one `&mut self` handle for callers that want that.

use std::sync::Arc;

use crate::arena::{Arena, NodeId};
use crate::coloring::ColumnColoring;
use crate::eval::{CompiledExpr, TangentInputs};
use crate::ir::TypedIr;
use crate::jacobian::{
    CscPattern, JacobianData, JacobianLayout, JacobianScratch, build_csr_to_csc_map,
    build_row_to_csc_entries,
};
use crate::node::CsrData;
use crate::observable::{ObservableKind, ObservableScratch, ObservableSet, seed_param_tangent};
use crate::simplify::simplify_pipeline;
use crate::sparsity::SparsityPattern;
use crate::tangent::tangent_wrt_params;

/// Classification of the mass matrix for dispatch in hot-path operations.
///
/// Detected once at `CompiledModel::new()` time. Identity and `DiagonalSelector`
/// paths replace the general sparse matvec with O(n) operations.
#[derive(Clone, Debug)]
enum MassKind {
    /// M = I (pure ODE, all states differential).
    Identity,
    /// M = diag(mask): 1 for differential states, 0 for algebraic.
    DiagonalSelector(Vec<bool>),
    /// Arbitrary sparse M, fallback to CSR matvec.
    General,
}

/// Classify a CSR mass matrix into the fastest applicable [`MassKind`].
///
/// Returns `Identity` when `M = I`, `DiagonalSelector` when every row has at
/// most one non-zero on the diagonal with value 0 or 1, and `General`
/// otherwise.
// Exact 0.0/1.0 compares classify structural mass entries; an epsilon
// tolerance would misclassify a near-identity matrix as `Identity`.
#[allow(clippy::float_cmp)]
fn classify_mass_matrix(mass: &CsrData) -> MassKind {
    let n = mass.shape.rows;
    if n != mass.shape.cols {
        return MassKind::General;
    }

    for row in 0..n {
        let start = mass.indptr[row];
        let end = mass.indptr[row + 1];
        let row_nnz = end - start;

        if row_nnz > 1 {
            return MassKind::General;
        }
        if row_nnz == 1 {
            if mass.indices[start] != row {
                return MassKind::General;
            }
            let val = mass.data[start];
            if val != 1.0 && val != 0.0 {
                return MassKind::General;
            }
        }
    }

    let nnz = mass.indptr[n];
    if nnz == n && mass.data.iter().all(|&v| v == 1.0) {
        return MassKind::Identity;
    }

    let mut mask = vec![false; n];
    for (row, m) in mask.iter_mut().enumerate() {
        let start = mass.indptr[row];
        let end = mass.indptr[row + 1];
        if start < end && mass.data[start] == 1.0 {
            *m = true;
        }
    }
    MassKind::DiagonalSelector(mask)
}

/// Jacobian assembly policy used by [`CompiledModel`].
///
/// One variant today, and every model compiles to it. The enum exists so a
/// second policy (a dense or reverse-only assembly) can be added without
/// changing the reporting surface; until one lands there is nothing to select
/// between, so nothing reports a request separately from an outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JacobianStrategy {
    /// Assemble through colored JVP passes.
    Coloring,
}

impl JacobianStrategy {
    /// Stable name for stats and Python-side reporting.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Coloring => "coloring",
        }
    }
}

/// Lightweight Jacobian-assembly stats exposed to benchmarks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JacobianStats {
    /// Strategy this model compiled to. `JacobianStrategy` has a single variant,
    /// so there is no request/selection to report separately.
    pub strategy: JacobianStrategy,
    /// Colors in the adopted coloring, so tangent sweeps per assembly.
    pub n_colors: usize,
    /// Non-zeros in the assembled Jacobian.
    pub nnz: usize,
    /// Rows taken out of the coloring and filled by reverse mode instead, so
    /// also the reverse passes one assembly adds.
    pub n_dense_rows: usize,
    /// Rows `detect_dense_rows` nominated. Above `n_dense_rows` means the
    /// split was declined — correct, but the colouring stayed wide.
    pub n_dense_row_candidates: usize,
    /// Entries a compile pass proved constant, written from a table rather
    /// than swept.
    pub n_constant_entries: usize,
    /// Columns a sweep still has to produce, so those the coloring covers.
    pub n_swept_columns: usize,
    /// Jacobian entries those dense rows account for.
    pub dense_row_entries: usize,
    /// Instructions across those tapes: the split's compiled-memory cost.
    pub dense_row_tape_instructions: usize,
    /// Instructions in the shared primal section of the split-eval tape, `None`
    /// when the Jacobian tape is not split.
    pub split_eval_primal_instructions: Option<usize>,
    /// Common tape + dispatch: what actually runs regardless of branch.
    pub split_eval_total_instructions: Option<usize>,
    /// Raw tape length, branch blocks included.
    pub split_eval_raw_instructions: Option<usize>,
    /// How many dispatches `split_eval_total_instructions` includes, one per
    /// short-circuited conditional, in each of the primal and tangent halves.
    pub split_eval_dispatch_count: usize,
    /// Per-branch block lengths, in tape order.
    pub branch_block_lens: Vec<u32>,
    /// Colours per batched tangent sweep, or 1 when the tape runs the scalar
    /// per-colour path.
    pub jac_lane_width: usize,
}

/// The compiled algebraic sub-block: residual `g(t, y)` and its Jacobian
/// `dg/dy_alg`, for the Newton solver that consistently initialises the
/// algebraic states.
///
/// One of three [`JacobianData`] adapters, so the block gets the colour sweep,
/// its lane batching and its dense-row handling from the assembly module rather
/// than restating them; what is left here is the block's own descriptor.
#[derive(Debug, Clone)]
struct AlgebraicBlock {
    /// `g(t, y)`, compiled through the same simplify pipeline as the residual.
    residual: Arc<CompiledExpr>,
    /// `dg/dy_alg` over the algebraic columns only. Its layout, its COO triplet
    /// and the block's width are all facts about this artifact, so they are read
    /// from it rather than cached here where they could drift out of step.
    jac: Arc<JacobianData>,
}

/// The algebraic sub-block to compile alongside the right-hand side, for the
/// Newton solver that consistently initialises algebraic states.
#[derive(Debug, Clone, Copy)]
pub struct CompiledModelAlgebraicBlock<'a> {
    rhs: NodeId,
    var_indices: &'a [usize],
}

impl<'a> CompiledModelAlgebraicBlock<'a> {
    /// `rhs` is the algebraic residual `g(t, y)`, and `var_indices` names the
    /// algebraic states by global state index.
    #[inline]
    pub const fn new(rhs: NodeId, var_indices: &'a [usize]) -> Self {
        Self { rhs, var_indices }
    }
}

/// Optional artifacts to compile with a model, so the many-argument constructors
/// do not multiply with every combination.
#[derive(Debug, Clone, Copy, Default)]
#[must_use]
pub struct CompiledModelOptions<'a> {
    sens_param_indices: &'a [usize],
    algebraic_block: Option<CompiledModelAlgebraicBlock<'a>>,
}

impl<'a> CompiledModelOptions<'a> {
    /// No sensitivities and no algebraic block, matching a plain
    /// [`CompiledModel::new`].
    #[inline]
    pub const fn new() -> Self {
        Self {
            sens_param_indices: &[],
            algebraic_block: None,
        }
    }

    /// Compile `df/dp` for these parameters, named by global parameter index. The
    /// order given becomes the sensitivity column order for the whole solve.
    #[inline]
    pub const fn with_sensitivities(mut self, sens_param_indices: &'a [usize]) -> Self {
        self.sens_param_indices = sens_param_indices;
        self
    }

    /// Compile the algebraic residual and its Jacobian as well as the RHS.
    #[inline]
    pub const fn with_algebraic(
        mut self,
        algebraic_block: CompiledModelAlgebraicBlock<'a>,
    ) -> Self {
        self.algebraic_block = Some(algebraic_block);
        self
    }

    #[inline]
    const fn sens_param_indices(self) -> &'a [usize] {
        self.sens_param_indices
    }

    #[inline]
    const fn algebraic_block(self) -> Option<CompiledModelAlgebraicBlock<'a>> {
        self.algebraic_block
    }
}

/// Per-solve mutable scratch for evaluating a [`CompiledModel`].
///
/// One evaluation buffer per `CompiledExpr` plus auxiliary buffers, created
/// fresh per solve so repeated solves never share mutable state.
#[derive(Debug, Clone)]
pub struct Workspace {
    pub(crate) primal_scratch: Vec<f64>,
    /// Every buffer the `df/dy` assembly and its matrix-free actions need,
    /// sized and lane-widthed by the assembly module.
    pub(crate) jac_scratch: JacobianScratch,
    pub(crate) sens_scratch: Option<Vec<f64>>,
    /// Buffers for the two observable families, each sized by its own set.
    pub(crate) output_scratch: ObservableScratch,
    pub(crate) event_scratch: ObservableScratch,
    pub(crate) algebraic_scratch: Option<Vec<f64>>,
    /// The algebraic block's own assembly buffers; `None` without a block.
    pub(crate) algebraic_jac_scratch: Option<JacobianScratch>,
    pub(crate) cj: f64,
    pub(crate) mv_buffer: Vec<f64>,
    pub(crate) sens_dp_buffer: Vec<f64>,
    /// Counts `sens_primal_pass` calls so tests can pin the batching mechanism.
    #[cfg(test)]
    pub(crate) sens_primal_passes: usize,
}

impl Workspace {
    /// Set the cj coefficient used by Jacobian/residual assembly.
    #[inline]
    pub const fn set_cj(&mut self, cj: f64) {
        self.cj = cj;
    }

    /// Current cj coefficient.
    #[inline]
    pub const fn cj(&self) -> f64 {
        self.cj
    }

    /// The buffers paired with the observables of `kind`, alongside the
    /// parameter-tangent buffer their sens action seeds.
    ///
    /// Handed out together because the borrow checker cannot see through two
    /// separate accessors that both take `&mut self`.
    #[inline]
    fn observable_tangent(&mut self, kind: ObservableKind) -> (&mut ObservableScratch, &mut [f64]) {
        let scratch = match kind {
            ObservableKind::Outputs => &mut self.output_scratch,
            ObservableKind::Events => &mut self.event_scratch,
        };
        (scratch, &mut self.sens_dp_buffer)
    }

    /// The buffers paired with the observables of `kind`.
    #[inline]
    fn observable_scratch(&mut self, kind: ObservableKind) -> &mut ObservableScratch {
        self.observable_tangent(kind).0
    }
}

/// A compiled DAE system: primal expression, symbolic Jacobian, mass matrix.
///
/// Immutable after construction, so it is shared via `Arc` and read by any
/// number of concurrent solves. Mutable scratch lives in a [`Workspace`] created
/// via [`Self::create_workspace`]; [`ModelEvaluator`] bundles the two for
/// callers that would rather hold one `&mut self` handle.
#[derive(Clone)]
pub struct CompiledModel {
    primal_expr: Arc<CompiledExpr>,
    jac: Arc<JacobianData>,
    /// Mass matrix M in CSR format.
    mass_matrix: CsrData,
    mass_kind: MassKind,
    sparsity: SparsityPattern,
    csc_sparsity: CscPattern,
    n_states: usize,
    n_params: usize,
    /// `jac`'s entries on the MERGED CSC this model assembles into, which carries
    /// the mass pattern's slots too. The remap is all this model adds to the
    /// assembly module's tables.
    jac_layout: JacobianLayout,
    /// Mass-matrix CSR entry to CSC slot mapping.
    mass_to_csc_map: Vec<usize>,
    algebraic_ids: Vec<bool>,
    sens_expr: Option<Arc<CompiledExpr>>,
    /// Global parameter index for each sensitivity (range `0..n_params`).
    sens_param_indices: Vec<usize>,
    /// Output variables H(t, y; p) and their tangent tapes.
    outputs: ObservableSet,
    /// Event functions g(t, y; p) and their tangent tapes. Fused into one tape
    /// once built, since both hot loops evaluate every event each step.
    events: ObservableSet,
    algebraic: Option<AlgebraicBlock>,
}

impl CompiledModel {
    /// Create a new `CompiledModel` from a primal expression and mass matrix.
    ///
    /// Builds the symbolic derivative df/dy, detects sparsity, and computes
    /// coloring for efficient Jacobian assembly.
    pub fn new(
        arena: &Arena,
        rhs: NodeId,
        mass_matrix: CsrData,
        n_states: usize,
        n_params: usize,
    ) -> Self {
        // The rhs is compiled raw. Any value-preserving-but-ULP-shifting fold
        // stalls IDA's Newton; `simplify`'s int-pow lowering alone moves the
        // DFN residual 4096 ULP. `residual_is_compiled_bit_exactly` pins this.
        let primal_ir = TypedIr::from_arena(arena, rhs);
        let n_outputs = primal_ir.output_len();

        let jac = Arc::new(JacobianData::new_wrt_states(
            arena, rhs, n_outputs, n_states,
        ));

        let mass_sparsity = SparsityPattern::from_csr_data(&mass_matrix);
        let mut sparsity = jac.sparsity().clone();
        sparsity.merge_with(&mass_sparsity);

        // Convert CSR sparsity to CSC for KLU compatibility
        let csc_sparsity = CscPattern::from_csr(&sparsity);

        let row_to_csc_entries = build_row_to_csc_entries(&csc_sparsity);
        let mass_to_csc_map = build_csr_to_csc_map(&mass_sparsity, &row_to_csc_entries);
        // The one thing this model adds to the assembly module: its own slot
        // numbering, since a merged buffer interleaves the mass pattern's entries.
        let jac_layout = jac.layout_in(&csc_sparsity);

        // Compile primal expression
        let primal_expr = Arc::new(CompiledExpr::from_ir(primal_ir));

        // Detect algebraic states: a state i is algebraic iff row i of the
        // mass matrix has no diagonal entry. Used by IDA's `id` vector.
        let algebraic_ids: Vec<bool> = (0..n_states)
            .map(|i| {
                let row_start = mass_matrix.indptr[i];
                let row_end = mass_matrix.indptr[i + 1];
                !mass_matrix.indices[row_start..row_end].contains(&i)
            })
            .collect();

        // Classify mass matrix once for fast-path dispatch in hot-path methods.
        let mass_kind = classify_mass_matrix(&mass_matrix);

        Self {
            primal_expr,
            jac,
            mass_matrix,
            mass_kind,
            sparsity,
            csc_sparsity,
            n_states,
            n_params,
            jac_layout,
            mass_to_csc_map,
            algebraic_ids,
            sens_expr: None,
            sens_param_indices: Vec::new(),
            outputs: ObservableSet::new(),
            events: ObservableSet::new(),
            algebraic: None,
        }
    }

    /// Create a `CompiledModel` with forward-sensitivity expressions for
    /// the given parameter indices (each must be `< n_params`).
    ///
    /// All sensitivities reuse a single compiled tangent expression.
    /// Panics if any entry in `sens_param_indices` is out of range.
    pub fn new_with_sens(
        arena: &Arena,
        rhs: NodeId,
        mass_matrix: CsrData,
        n_states: usize,
        n_params: usize,
        sens_param_indices: &[usize],
    ) -> Self {
        Self::new_with_options(
            arena,
            rhs,
            mass_matrix,
            n_states,
            n_params,
            CompiledModelOptions::new().with_sensitivities(sens_param_indices),
        )
    }

    /// Build a `CompiledModel` with optional algebraic sub-block expressions.
    ///
    /// `algebraic_rhs` is the algebraic residual expression g(t, y).
    /// `algebraic_variable_indices` lists which state indices are algebraic.
    ///
    /// When `algebraic_rhs` is `None`, this is equivalent to calling [`Self::new`].
    pub fn new_with_algebraic(
        arena: &Arena,
        rhs: NodeId,
        mass_matrix: CsrData,
        n_states: usize,
        n_params: usize,
        algebraic_rhs: Option<NodeId>,
        algebraic_variable_indices: &[usize],
    ) -> Self {
        let options = algebraic_rhs.map_or_else(CompiledModelOptions::new, |algebraic_rhs| {
            CompiledModelOptions::new().with_algebraic(CompiledModelAlgebraicBlock::new(
                algebraic_rhs,
                algebraic_variable_indices,
            ))
        });

        Self::new_with_options(arena, rhs, mass_matrix, n_states, n_params, options)
    }

    /// Build a `CompiledModel` with any combination of sensitivities and an
    /// algebraic sub-block.
    pub fn new_with_options(
        arena: &Arena,
        rhs: NodeId,
        mass_matrix: CsrData,
        n_states: usize,
        n_params: usize,
        options: CompiledModelOptions<'_>,
    ) -> Self {
        let mut model = Self::new(arena, rhs, mass_matrix, n_states, n_params);
        model.apply_options(arena, rhs, options);
        model
    }

    fn apply_options(&mut self, arena: &Arena, rhs: NodeId, options: CompiledModelOptions<'_>) {
        self.compile_sensitivities(arena, rhs, options.sens_param_indices());
        if let Some(algebraic_block) = options.algebraic_block() {
            self.compile_algebraic_block(arena, algebraic_block.rhs, algebraic_block.var_indices);
        }
    }

    fn compile_sensitivities(&mut self, arena: &Arena, rhs: NodeId, sens_param_indices: &[usize]) {
        if sens_param_indices.is_empty() {
            return;
        }
        let mut seen = vec![false; self.n_params];
        for &idx in sens_param_indices {
            assert!(
                idx < self.n_params,
                "sens_param_indices[{idx}] is out of range for n_params={}",
                self.n_params,
            );
            assert!(
                !std::mem::replace(&mut seen[idx], true),
                "sens_param_indices contains a repeated index: {idx}",
            );
        }

        let mut diff_arena = arena.clone();
        let df_dp = tangent_wrt_params(&mut diff_arena, rhs);
        let (final_arena, df_dp) = simplify_pipeline(diff_arena, df_dp);
        let ir = TypedIr::from_arena_split_eval(&final_arena, df_dp);
        self.sens_expr = Some(Arc::new(CompiledExpr::from_ir(ir)));
        self.sens_param_indices = sens_param_indices.to_vec();
    }

    /// Compile the algebraic residual and its Jacobian, the third adapter onto
    /// the assembly module.
    ///
    /// The Jacobian is a plain [`JacobianData`] over the algebraic columns, so
    /// this block gets the colour sweep, its lane batching and the seed-lane
    /// hygiene from there; what is compiled here is the residual and the block's
    /// COO descriptor.
    fn compile_algebraic_block(
        &mut self,
        arena: &Arena,
        algebraic: NodeId,
        algebraic_variable_indices: &[usize],
    ) {
        let (algebraic_arena, algebraic_simplified) = simplify_pipeline(arena.clone(), algebraic);
        let algebraic_ir = TypedIr::from_arena(&algebraic_arena, algebraic_simplified);
        let n_rows = algebraic_ir.output_len();
        let residual = Arc::new(CompiledExpr::from_ir(algebraic_ir));

        let jac = Arc::new(JacobianData::new_wrt_state_subset(
            arena,
            algebraic,
            n_rows,
            self.n_states,
            algebraic_variable_indices,
        ));
        self.algebraic = Some(AlgebraicBlock { residual, jac });
    }

    /// Scratch length required by the primal `f(t, y)` expression.
    #[cfg(test)]
    pub fn primal_scratch_len(&self) -> usize {
        self.primal_expr.scratch_len()
    }

    /// Allocate a fresh [`Workspace`] sized for this model's expressions.
    pub fn create_workspace(&self) -> Workspace {
        Workspace {
            primal_scratch: vec![0.0; self.primal_expr.scratch_len()],
            jac_scratch: JacobianScratch::new(&self.jac),
            sens_scratch: self.sens_expr.as_ref().map(|e| vec![0.0; e.scratch_len()]),
            output_scratch: self.outputs.create_scratch(),
            event_scratch: self.events.create_scratch(),
            algebraic_scratch: self
                .algebraic
                .as_ref()
                .map(|block| vec![0.0; block.residual.scratch_len()]),
            algebraic_jac_scratch: self
                .algebraic
                .as_ref()
                .map(|block| JacobianScratch::new(&block.jac)),
            cj: 0.0,
            mv_buffer: vec![0.0; self.n_states],
            sens_dp_buffer: vec![0.0; self.n_params.max(1)],
            #[cfg(test)]
            sens_primal_passes: 0,
        }
    }

    /// Shared handle to the primal `f(t, y)` expression (no recompilation).
    #[inline]
    pub fn primal_expr_arc(&self) -> Arc<CompiledExpr> {
        Arc::clone(&self.primal_expr)
    }

    /// Shared handle to the prepared df/dy artifact (no recompilation).
    #[inline]
    pub fn jacobian_data(&self) -> Arc<JacobianData> {
        Arc::clone(&self.jac)
    }

    /// The observables of `kind`: their tapes, layout and scratch sizing.
    #[inline]
    pub const fn observables(&self, kind: ObservableKind) -> &ObservableSet {
        match kind {
            ObservableKind::Outputs => &self.outputs,
            ObservableKind::Events => &self.events,
        }
    }

    /// Shared handle to the algebraic residual expression, if present.
    #[inline]
    pub fn algebraic_expr_arc(&self) -> Option<Arc<CompiledExpr>> {
        self.algebraic
            .as_ref()
            .map(|block| Arc::clone(&block.residual))
    }

    /// Shared handle to the prepared `dg/dy_alg` artifact, if present. The
    /// binding's standalone algebraic Jacobian is a view onto this, never a
    /// second tangent transform of the same expression.
    #[inline]
    pub fn algebraic_jacobian_data(&self) -> Option<Arc<JacobianData>> {
        self.algebraic.as_ref().map(|block| Arc::clone(&block.jac))
    }

    /// Length of the state vector this model was compiled against.
    #[inline]
    pub const fn n_states(&self) -> usize {
        self.n_states
    }

    /// Number of input parameters declared at compile time.
    #[inline]
    pub const fn n_params(&self) -> usize {
        self.n_params
    }

    /// Length of `f(t, y)`, which is `n_states` for a well-formed model.
    #[inline]
    #[allow(clippy::missing_const_for_fn)] // Calls non-const method
    pub fn output_len(&self) -> usize {
        self.primal_expr.output_len()
    }

    /// Evaluate f(t, y) into `output`.
    #[inline]
    pub fn eval_rhs(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        output: &mut [f64],
    ) {
        let result = self
            .primal_expr
            .eval(&mut ws.primal_scratch, t, y, &[], inputs);
        output[..result.len()].copy_from_slice(result);
    }

    /// Compute (df/dy - cj*M) @ v via forward-mode AD and sparse matvec.
    pub fn jac_mul(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        v: &[f64],
        output: &mut [f64],
    ) {
        self.jac_mul_primal(ws, t, y, inputs, v, output);

        // Subtract cj * (M @ v), dispatching on mass kind for fast paths.
        if ws.cj != 0.0 {
            let n = self.n_states.min(output.len());
            match &self.mass_kind {
                MassKind::Identity => {
                    for i in 0..n {
                        output[i] -= ws.cj * v[i];
                    }
                },
                MassKind::DiagonalSelector(mask) => {
                    for (i, &is_differential) in mask.iter().take(n).enumerate() {
                        if is_differential {
                            output[i] -= ws.cj * v[i];
                        }
                    }
                },
                MassKind::General => {
                    csr_matvec(&self.mass_matrix, v, &mut ws.mv_buffer);
                    let cj = ws.cj;
                    for (out, &mv) in output.iter_mut().zip(&ws.mv_buffer).take(n) {
                        *out -= cj * mv;
                    }
                },
            }
        }
    }

    /// Compute DAE residual r = M*y' - f(t, y).
    ///
    /// Used by IDAKLU and other DAE solvers.
    pub fn eval_residual(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        yp: &[f64],
        inputs: &[f64],
        r: &mut [f64],
    ) {
        // Compute f(t, y) first so we can fuse M*y' - f in a single pass.
        let f = self
            .primal_expr
            .eval(&mut ws.primal_scratch, t, y, &[], inputs);
        let n = self.n_states.min(r.len());
        match &self.mass_kind {
            MassKind::Identity => {
                for i in 0..n {
                    r[i] = yp[i] - f[i];
                }
            },
            MassKind::DiagonalSelector(mask) => {
                for (i, &is_differential) in mask.iter().take(n).enumerate() {
                    r[i] = if is_differential { yp[i] } else { 0.0 } - f[i];
                }
            },
            MassKind::General => {
                csr_matvec(&self.mass_matrix, yp, &mut ws.mv_buffer);
                for i in 0..n {
                    r[i] = ws.mv_buffer[i] - f[i];
                }
            },
        }
    }

    /// Compute pure Jacobian-vector product: df/dy @ v (no mass term)
    ///
    /// This is for pybammsolvers ABI which subtracts cj*M@v separately.
    pub fn jac_action(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        v: &[f64],
        jv: &mut [f64],
    ) {
        self.jac_mul_primal(ws, t, y, inputs, v, jv);
    }

    /// Compute mass matrix action: M @ v (no beta term).
    ///
    /// Dispatches to an O(n) fast path for `Identity` and `DiagonalSelector`
    /// mass matrices, falling back to a general CSR matvec for `General`.
    pub fn mass_action(&self, v: &[f64], mv: &mut [f64]) {
        self.mass_action_into(v, mv);
    }

    /// Helper: write M @ x into `mv` (no beta). Dispatches on mass kind.
    fn mass_action_into(&self, x: &[f64], mv: &mut [f64]) {
        match &self.mass_kind {
            MassKind::Identity => {
                mv[..self.n_states].copy_from_slice(&x[..self.n_states]);
            },
            MassKind::DiagonalSelector(mask) => {
                for i in 0..self.n_states {
                    mv[i] = if mask[i] { x[i] } else { 0.0 };
                }
            },
            MassKind::General => {
                csr_matvec(&self.mass_matrix, x, mv);
            },
        }
    }

    /// CSR pattern of the assembled Jacobian `df/dy - cj*M`, which is the union
    /// of the `df/dy` and mass patterns.
    #[inline]
    pub const fn sparsity(&self) -> &SparsityPattern {
        &self.sparsity
    }

    /// CSR pattern of `df/dy` alone, without the mass entries.
    #[inline]
    pub fn jac_y_sparsity(&self) -> &SparsityPattern {
        self.jac.sparsity()
    }

    /// Coloring that drives the JVP sweep count, decided once at compile time.
    /// With a dense-row split adopted it is the reduced coloring, so it does not
    /// cover every row of [`jac_y_sparsity`](Self::jac_y_sparsity).
    #[inline]
    pub fn coloring(&self) -> &ColumnColoring {
        self.jac.coloring()
    }

    /// `(csc_idx, value)` in the assembled CSC for every df/dy entry a compile
    /// pass proved constant. Split dense rows are absent: their own tape
    /// recomputes them.
    #[inline]
    pub fn constant_jacobian_entries(&self) -> &[(usize, f64)] {
        self.jac_layout.constant_slots()
    }

    /// The mass matrix `M` in CSR, as Python supplied it.
    #[inline]
    pub const fn mass_matrix(&self) -> &CsrData {
        &self.mass_matrix
    }

    /// Get the algebraic-state mask: `true` = algebraic, `false` = differential.
    ///
    /// A state `i` is classified algebraic when row `i` of the mass matrix has
    /// no diagonal entry, which matches `PyBaMM`'s convention.
    #[inline]
    pub fn algebraic_ids(&self) -> &[bool] {
        &self.algebraic_ids
    }

    /// The same mask in IDA's numeric convention: 1.0 differential, 0.0
    /// algebraic, the inverse polarity of [`algebraic_ids`](Self::algebraic_ids).
    pub fn algebraic_ids_f64(&self, output: &mut [f64]) {
        for (i, &is_alg) in self.algebraic_ids.iter().enumerate() {
            output[i] = if is_alg { 0.0 } else { 1.0 };
        }
    }

    /// Number of algebraic states, or 0 without an algebraic sub-block.
    #[inline]
    pub fn n_algebraic(&self) -> usize {
        self.algebraic
            .as_ref()
            .map_or(0, |block| block.jac.n_cols())
    }

    /// Whether an algebraic sub-block was compiled for the Newton solver.
    #[inline]
    pub const fn has_algebraic(&self) -> bool {
        self.algebraic.is_some()
    }

    /// Non-zeros in the assembled algebraic Jacobian `dg/dy_alg`.
    pub fn algebraic_jacobian_nnz(&self) -> usize {
        self.algebraic.as_ref().map_or(0, |block| block.jac.nnz())
    }

    /// COO row indices of the algebraic Jacobian, in the order
    /// [`assemble_algebraic_jacobian_into`](Self::assemble_algebraic_jacobian_into) writes.
    #[inline]
    pub fn algebraic_jacobian_row_indices(&self) -> &[usize] {
        self.algebraic
            .as_ref()
            .map_or(&[], |block| block.jac.coo_global_indices().0)
    }

    /// COO column indices matching
    /// [`algebraic_jacobian_row_indices`](Self::algebraic_jacobian_row_indices),
    /// as global state indices.
    #[inline]
    pub fn algebraic_jacobian_col_indices(&self) -> &[usize] {
        self.algebraic
            .as_ref()
            .map_or(&[], |block| block.jac.coo_global_indices().1)
    }

    /// Evaluate the algebraic residual g(t, y) into `output`.
    ///
    /// No-op when no algebraic sub-block was compiled.
    pub fn eval_algebraic_residual(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        output: &mut [f64],
    ) {
        if let (Some(block), Some(scratch)) =
            (self.algebraic.as_ref(), ws.algebraic_scratch.as_mut())
        {
            let result = block.residual.eval(scratch, t, y, &[], inputs);
            output[..result.len()].copy_from_slice(result);
        }
    }

    /// Compute (`dg/dy_alg`) @ v for the algebraic Jacobian-vector product.
    ///
    /// No-op when no algebraic sub-block was compiled.
    pub fn eval_algebraic_jacobian_action(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        v: &[f64],
        output: &mut [f64],
    ) {
        if let (Some(block), Some(scratch)) =
            (self.algebraic.as_ref(), ws.algebraic_jac_scratch.as_mut())
        {
            let n_algebraic = block.jac.n_cols();
            assert!(
                output.len() >= n_algebraic,
                "algebraic output buffer too small: need {n_algebraic}, got {}",
                output.len()
            );
            block.jac.action_into(scratch, t, y, &[], inputs, v, output);
        }
    }

    /// Assemble the algebraic Jacobian `dg/dy_alg` into `jac_data`.
    ///
    /// The output order matches [`Self::algebraic_jacobian_row_indices`] and
    /// [`Self::algebraic_jacobian_col_indices`].
    pub fn assemble_algebraic_jacobian_into(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        jac_data: &mut [f64],
    ) {
        let Some(block) = self.algebraic.as_ref() else {
            return;
        };
        let scratch = ws
            .algebraic_jac_scratch
            .as_mut()
            .expect("an algebraic block implies its workspace scratch");
        block
            .jac
            .assemble_into(scratch, block.jac.layout(), t, y, &[], inputs, jac_data);
    }

    /// Number of parameters sensitivities were requested for, which may be a
    /// subset of [`n_params`](Self::n_params). Zero without sensitivities.
    #[inline]
    pub const fn n_sens_params(&self) -> usize {
        self.sens_param_indices.len()
    }

    /// Evaluate ∂f/∂p for the sensitivity indexed by `sens_idx` into `output`.
    ///
    /// `sens_idx` indexes into `sens_param_indices` (range `0..n_sens_params()`).
    /// Panics if `sens_idx >= n_sens_params()` or if no sensitivities were
    /// configured at construction time.
    pub fn eval_sens(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        sens_idx: usize,
        output: &mut [f64],
    ) {
        let expr = self
            .sens_expr
            .as_ref()
            .expect("eval_sens called on a model without sensitivities");
        let scratch = ws
            .sens_scratch
            .as_mut()
            .expect("sens_scratch missing from workspace");
        let global_idx = self.sens_param_indices[sens_idx];

        // Build the unit dp vector in the pre-allocated buffer
        debug_assert!(global_idx < ws.sens_dp_buffer.len());
        ws.sens_dp_buffer.fill(0.0);
        ws.sens_dp_buffer[global_idx] = 1.0;
        let tangent = TangentInputs {
            dy: None,
            dp: Some(&ws.sens_dp_buffer),
        };

        let result = expr.eval_with_tangent(scratch, t, y, &[], inputs, &tangent);
        output[..result.len()].copy_from_slice(result);
    }

    /// Run the primal section of `sens_expr` once for (t, y, inputs). The primal
    /// slots in `sens_scratch` then serve every subsequent tangent column.
    /// Only callable on models compiled with sensitivities; panics otherwise.
    pub fn sens_primal_pass(&self, ws: &mut Workspace, t: f64, y: &[f64], inputs: &[f64]) {
        #[cfg(test)]
        {
            ws.sens_primal_passes += 1;
        }
        let expr = self.sens_expr.as_ref().expect("no sensitivity expression");
        let scratch = ws
            .sens_scratch
            .as_mut()
            .expect("sens_scratch missing from workspace");
        expr.run_primal_section(scratch, t, y, &[], inputs);
    }

    /// Tangent-only sweep for the column of parameter `param_idx`. Requires a prior
    /// `sens_primal_pass` at the same (t, y, inputs) on this workspace.
    /// Only callable on models compiled with sensitivities; panics otherwise.
    pub fn sens_tangent_column(&self, ws: &mut Workspace, param_idx: usize, output: &mut [f64]) {
        let expr = self.sens_expr.as_ref().expect("no sensitivity expression");
        debug_assert!(param_idx < ws.sens_dp_buffer.len());
        ws.sens_dp_buffer.fill(0.0);
        ws.sens_dp_buffer[param_idx] = 1.0;
        let scratch = ws
            .sens_scratch
            .as_mut()
            .expect("sens_scratch missing from workspace");
        let tangent = TangentInputs {
            dy: None,
            dp: Some(&ws.sens_dp_buffer),
        };
        let result = expr.run_tangent_section(scratch, &tangent);
        output[..result.len()].copy_from_slice(result);
    }

    /// Evaluate ∂f/∂p for all configured sensitivity parameters into `output`.
    ///
    /// Layout: `output[i*n_states + j] = ∂f_j/∂p_i`. The buffer must have
    /// length at least `n_sens_params() * n_states`. Runs the shared primal
    /// section once, then one tangent-only sweep per parameter.
    pub fn eval_sens_all(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        output: &mut [f64],
    ) {
        // No configured sensitivities: keep the historical silent no-op.
        if self.n_sens_params() == 0 {
            return;
        }
        let n_states = self.n_states;
        self.sens_primal_pass(ws, t, y, inputs);
        for (i, &param_idx) in self.sens_param_indices.iter().enumerate() {
            let start = i * n_states;
            self.sens_tangent_column(ws, param_idx, &mut output[start..start + n_states]);
        }
    }

    /// Compile and append an output-variable expression to this model.
    ///
    /// `node` must already exist in `arena`. Its length is captured at compile
    /// time, along with the dH/dp and dH/dy tangent graphs.
    pub fn add_output(&mut self, arena: &Arena, node: NodeId) {
        self.outputs.push(arena, node);
    }

    /// Compile and append an event expression to this model.
    ///
    /// Events are used for root-finding during integration. When any event
    /// expression crosses zero, the solver can terminate or take action.
    ///
    /// `node` must already exist in `arena`.
    pub fn add_event(&mut self, arena: &Arena, node: NodeId) {
        self.events.push(arena, node);
    }

    /// Fuse the events into one tape (see [`ObservableSet::fuse`]).
    ///
    /// Call once, after every event is added. `event_roots` must be the same
    /// nodes, in the same order, passed to [`Self::add_event`], from `arena`.
    pub fn fuse_events(&mut self, arena: &mut Arena, event_roots: &[NodeId]) {
        self.events.fuse(arena, event_roots);
    }

    /// Number of compiled output-variable expressions.
    #[inline]
    pub const fn n_outputs(&self) -> usize {
        self.outputs.count()
    }

    /// Length of output variable `var_idx`.
    ///
    /// Panics if `var_idx >= n_outputs()`.
    #[inline]
    pub fn output_len_at(&self, var_idx: usize) -> usize {
        self.outputs.len_at(var_idx)
    }

    /// Length of all output variables concatenated, the buffer size
    /// [`eval_observables`](Self::eval_observables) needs for
    /// [`ObservableKind::Outputs`].
    #[inline]
    pub const fn total_output_len(&self) -> usize {
        self.outputs.total_len()
    }

    /// Number of compiled event expressions.
    #[inline]
    pub const fn n_events(&self) -> usize {
        self.events.count()
    }

    /// Length of all events concatenated, the buffer size
    /// [`eval_observables`](Self::eval_observables) needs for
    /// [`ObservableKind::Events`].
    #[inline]
    pub const fn total_event_len(&self) -> usize {
        self.events.total_len()
    }

    /// Evaluate output variable `var_idx` into `output`, returning the count written.
    ///
    /// Panics if `var_idx >= n_outputs()` or `output.len() < output_len_at(var_idx)`.
    pub fn eval_output(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        var_idx: usize,
        output: &mut [f64],
    ) -> usize {
        self.outputs.eval_at(
            ws.observable_scratch(ObservableKind::Outputs),
            t,
            y,
            inputs,
            var_idx,
            output,
        )
    }

    /// Evaluate event `event_idx` into `output`, returning the count written.
    ///
    /// Panics if `event_idx >= n_events()` or `output.len() < the event's length`.
    pub fn eval_event(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        event_idx: usize,
        output: &mut [f64],
    ) -> usize {
        self.events.eval_at(
            ws.observable_scratch(ObservableKind::Events),
            t,
            y,
            inputs,
            event_idx,
            output,
        )
    }

    /// Evaluate every observable of `kind` into `output`, concatenated.
    ///
    /// Panics if `output.len()` is under the family's total length.
    pub fn eval_observables(
        &self,
        ws: &mut Workspace,
        kind: ObservableKind,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        output: &mut [f64],
    ) {
        self.observables(kind)
            .eval_all(ws.observable_scratch(kind), t, y, inputs, output);
    }

    /// Assemble `J = df/dy - cj * M` in COO (row, col, value) format, `cj`
    /// taken from the workspace.
    ///
    /// Allocates on every call and has no binding, so it is a convenience for
    /// tests and callers holding a `CompiledModel` directly; solver hot paths
    /// want [`Self::assemble_jacobian_csc_into`]. Expanded from the CSC
    /// assembly rather than re-deriving the colour loop, so the two cannot
    /// drift over which route fills a given slot.
    pub fn assemble_jacobian(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let nnz = self.sparsity.nnz();
        let mut values = vec![0.0; nnz];
        self.assemble_jacobian_csc_into_coloring(ws, t, y, inputs, &mut values);

        let (row_indices, col_indices) = self.csc_sparsity.csc_to_csr_map.iter().copied().unzip();
        (row_indices, col_indices, values)
    }

    /// CSC pattern of the assembled Jacobian `df/dy - cj*M`, the layout KLU and
    /// diffsol expect; not `df/dy` alone, which is
    /// [`jac_y_sparsity`](Self::jac_y_sparsity).
    #[inline]
    pub const fn csc_sparsity(&self) -> &CscPattern {
        &self.csc_sparsity
    }

    /// Non-zeros in the assembled Jacobian, the buffer size every
    /// `assemble_jacobian_csc_*` call needs.
    #[inline]
    pub const fn nnz(&self) -> usize {
        self.sparsity.nnz()
    }

    /// Jacobian assembly strategy this model compiled to.
    #[inline]
    pub const fn jacobian_strategy(&self) -> JacobianStrategy {
        JacobianStrategy::Coloring
    }

    /// Compile-time Jacobian metrics: colors, non-zeros and dense-row counts,
    /// for benchmark attribution.
    pub fn jacobian_stats(&self) -> JacobianStats {
        let ir = self.jac.assembly_tape().ir();
        let split_info = ir.split_eval_info();
        JacobianStats {
            strategy: JacobianStrategy::Coloring,
            n_colors: self.jac.n_colors(),
            nnz: self.nnz(),
            n_dense_rows: self.jac.n_dense_rows(),
            n_dense_row_candidates: self.jac.n_candidate_rows(),
            n_constant_entries: self.jac.constant_csr_entries().len(),
            n_swept_columns: self.jac.coloring().n_seeded_columns(),
            dense_row_entries: self.jac.dense_row_entries(),
            dense_row_tape_instructions: self.jac.dense_row_tape_instructions(),
            split_eval_primal_instructions: split_info.map(|s| s.primal_end),
            split_eval_total_instructions: Some(ir.common_instruction_count()),
            split_eval_raw_instructions: Some(ir.instructions().len()),
            split_eval_dispatch_count: ir.dispatch_count(),
            branch_block_lens: ir.branch_block_lens(),
            jac_lane_width: self.jac.lane_width(),
        }
    }

    /// Assemble the Jacobian into a pre-allocated CSC data buffer.
    ///
    /// Zero-allocation version for FFI/IDAKLU integration. The Jacobian
    /// is `J = df/dy - cj * M` where `cj` lives in `ws`.
    /// Panics if `jac_data.len() < nnz()`.
    pub fn assemble_jacobian_csc_into(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        jac_data: &mut [f64],
    ) {
        self.assemble_jacobian_csc_into_coloring(ws, t, y, inputs, jac_data);
    }

    /// Assemble the Jacobian into a pre-allocated CSC data buffer using the
    /// coloring-based JVP approach.
    ///
    /// Canonical Jacobian assembly method: one primal evaluation, then
    /// `n_colors` tangent sweeps with precomputed scatter.
    /// Panics if `jac_data.len() < nnz()`.
    pub fn assemble_jacobian_csc_into_coloring(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        jac_data: &mut [f64],
    ) {
        let nnz = self.sparsity.nnz();
        assert!(
            jac_data.len() >= nnz,
            "jac_data buffer too small: need {nnz}, got {}",
            jac_data.len()
        );

        self.assemble_dfdy_into(ws, t, y, inputs, jac_data);
        self.apply_mass_postpass(ws, jac_data);
    }

    /// `df/dy` into the merged CSC, by handing the assembly module this model's
    /// slot layout. Shared by both assembly entry points, which differ only in
    /// the mass term, so a future stage cannot reach one and miss the other.
    fn assemble_dfdy_into(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        jac_data: &mut [f64],
    ) {
        self.jac.assemble_into(
            &mut ws.jac_scratch,
            &self.jac_layout,
            t,
            y,
            &[],
            inputs,
            jac_data,
        );
    }

    fn jac_mul_primal(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        v: &[f64],
        output: &mut [f64],
    ) {
        self.jac
            .action_into(&mut ws.jac_scratch, t, y, &[], inputs, v, output);
    }

    /// Fold `-cj*M` into the merged CSC, after every derivative fill: the
    /// diagonal slots it touches are ones a colour sweep or a dense row has
    /// already written.
    fn apply_mass_postpass(&self, ws: &Workspace, jac_data: &mut [f64]) {
        if ws.cj == 0.0 {
            return;
        }
        for (mass_idx, &csc_idx) in self.mass_to_csc_map.iter().enumerate() {
            jac_data[csc_idx] -= ws.cj * self.mass_matrix.data[mass_idx];
        }
    }

    /// Assemble df/dy in CSC format using graph coloring, without applying
    /// the mass matrix post-pass (diffsol handles mass separately).
    ///
    /// Panics if `jac_data.len() < nnz`.
    pub fn assemble_jacobian_csc_no_mass(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        jac_data: &mut [f64],
    ) {
        let nnz = self.csc_sparsity.rowind.len();
        assert!(
            jac_data.len() >= nnz,
            "jac_data buffer too small: need {nnz}, got {}",
            jac_data.len()
        );
        self.assemble_dfdy_into(ws, t, y, inputs, jac_data);
        // No apply_mass_postpass, diffsol handles mass matrix separately
    }

    /// Batch-evaluate every output tape over `k` trajectory points; see
    /// [`ObservableSet::eval_batch`] for the buffer layouts.
    pub fn eval_outputs_batch(
        &self,
        ws: &mut Workspace,
        k: usize,
        ts: &[f64],
        y_cols: &[f64],
        inputs: &[f64],
        out: &mut [f64],
    ) {
        self.outputs.eval_batch(
            ws.observable_scratch(ObservableKind::Outputs),
            k,
            ts,
            y_cols,
            self.n_states,
            inputs,
            out,
        );
    }

    /// Compute the sensitivity action df/dp @ v.
    ///
    /// `v` is parameter-space; `sens_params[i]` is the global index of `v[i]`.
    /// Panics if no sensitivity expression was compiled.
    // Evaluation point, tangent, mapping and output are all distinct arguments.
    #[allow(clippy::too_many_arguments)]
    pub fn sens_action(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        sens_params: &[usize],
        v: &[f64],
        output: &mut [f64],
    ) {
        let sens_expr = self.sens_expr.as_ref().expect("no sensitivity expression");
        seed_param_tangent(&mut ws.sens_dp_buffer, sens_params, v);
        let scratch = ws.sens_scratch.as_mut().expect("no sensitivity scratch");
        let tangent = TangentInputs {
            dy: None,
            dp: Some(&ws.sens_dp_buffer),
        };
        let result = sens_expr.eval_with_tangent(scratch, t, y, &[], inputs, &tangent);
        output[..result.len()].copy_from_slice(result);
    }

    /// Whether `df/dp` was compiled, which requires sensitivities at build time.
    pub const fn has_sens(&self) -> bool {
        self.sens_expr.is_some()
    }

    /// Global parameter index for each configured sensitivity parameter.
    #[inline]
    pub fn sens_param_indices(&self) -> &[usize] {
        &self.sens_param_indices
    }

    /// Compute dH/dp · v over every observable of `kind`, into `output`.
    ///
    /// `v` is parameter-space; `sens_params[i]` is the global index of `v[i]`.
    // Evaluation point, family, tangent, mapping and output are all distinct.
    #[allow(clippy::too_many_arguments)]
    pub fn observable_sens_action(
        &self,
        ws: &mut Workspace,
        kind: ObservableKind,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        sens_params: &[usize],
        v: &[f64],
        output: &mut [f64],
    ) {
        let (scratch, dp) = ws.observable_tangent(kind);
        seed_param_tangent(dp, sens_params, v);
        self.observables(kind)
            .sens_action(scratch, t, y, inputs, dp, output);
    }

    /// Compute dH/dy · v over every observable of `kind`, into `output`.
    ///
    /// `v` is state-space, of length `n_states`.
    // Evaluation point, family, tangent and output are all distinct arguments.
    #[allow(clippy::too_many_arguments)]
    pub fn observable_jac_action(
        &self,
        ws: &mut Workspace,
        kind: ObservableKind,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        v: &[f64],
        output: &mut [f64],
    ) {
        self.observables(kind)
            .jac_action(ws.observable_scratch(kind), t, y, inputs, v, output);
    }

    /// Project state sensitivities onto output-variable sensitivities; see
    /// [`ObservableSet::sens_project`] for the buffer layouts.
    pub fn output_sens_project(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        y_sens: &[f64],
        out: &mut [f64],
    ) {
        let (scratch, dp) = ws.observable_tangent(ObservableKind::Outputs);
        self.outputs.sens_project(
            scratch,
            dp,
            t,
            y,
            inputs,
            &self.sens_param_indices,
            y_sens,
            self.n_states,
            out,
        );
    }
}

impl std::fmt::Debug for CompiledModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledModel")
            .field("n_states", &self.n_states)
            .field("n_params", &self.n_params)
            .field("output_len", &self.primal_expr.output_len())
            .field("sparsity_nnz", &self.sparsity.nnz())
            .field("n_colors", &self.jac.n_colors())
            .field("jacobian_strategy", &JacobianStrategy::Coloring.as_str())
            .finish_non_exhaustive()
    }
}

/// One [`CompiledModel`] paired with an owned mutable [`Workspace`], so a
/// caller can evaluate through `&mut self` without managing scratch by hand.
///
/// Owning the workspace is what makes this per-solve rather than shareable: the
/// FFI/pybammsolvers ABI path, the Python `CompiledModel` binding, and one-shot
/// eval/tests each hold one. Solves that share a model concurrently pass
/// `Arc<CompiledModel>` and their own [`Workspace`] instead.
///
/// Cloning shares the immutable [`CompiledModel`] via `Arc` and gives the clone
/// its own fresh [`Workspace`], so it does not deep-copy the mass matrix,
/// sparsity, scatter tables, or workspace buffers.
///
/// Its interface is only what the workspace buys; everything else is reached
/// through [`Deref`](std::ops::Deref).
pub struct ModelEvaluator {
    compiled: Arc<CompiledModel>,
    workspace: Workspace,
}

impl Clone for ModelEvaluator {
    fn clone(&self) -> Self {
        let mut workspace = self.compiled.create_workspace();
        workspace.set_cj(self.workspace.cj());
        Self {
            compiled: Arc::clone(&self.compiled),
            workspace,
        }
    }
}

impl std::fmt::Debug for ModelEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelEvaluator")
            .field("n_states", &self.compiled.n_states)
            .field("n_params", &self.compiled.n_params)
            .field("output_len", &self.compiled.primal_expr.output_len())
            .field("cj", &self.workspace.cj)
            .field("sparsity_nnz", &self.compiled.sparsity.nnz())
            .field("n_colors", &self.compiled.jac.n_colors())
            .field("jacobian_strategy", &JacobianStrategy::Coloring.as_str())
            .finish_non_exhaustive()
    }
}

/// Lends the wrapped [`CompiledModel`]'s read-only interface rather than
/// restating it.
///
/// No `DerefMut`: the artifact is shared behind an `Arc`, and only
/// `add_output`/`add_event`/`fuse_events` may mutate it.
impl std::ops::Deref for ModelEvaluator {
    type Target = CompiledModel;

    #[inline]
    fn deref(&self) -> &CompiledModel {
        &self.compiled
    }
}

/// Drops the workspace, so an `impl Into<Arc<CompiledModel>>` parameter takes
/// an evaluator as readily as an `Arc`.
impl From<ModelEvaluator> for Arc<CompiledModel> {
    fn from(evaluator: ModelEvaluator) -> Self {
        evaluator.into_compiled()
    }
}

impl ModelEvaluator {
    /// Create a new `ModelEvaluator` wrapper from a primal expression and mass matrix.
    pub fn new(
        arena: &Arena,
        rhs: NodeId,
        mass_matrix: CsrData,
        n_states: usize,
        n_params: usize,
    ) -> Self {
        let compiled = CompiledModel::new(arena, rhs, mass_matrix, n_states, n_params);
        let workspace = compiled.create_workspace();
        Self {
            compiled: Arc::new(compiled),
            workspace,
        }
    }

    /// Create a `ModelEvaluator` with forward-sensitivity expressions.
    pub fn new_with_sens(
        arena: &Arena,
        rhs: NodeId,
        mass_matrix: CsrData,
        n_states: usize,
        n_params: usize,
        sens_param_indices: &[usize],
    ) -> Self {
        let compiled = CompiledModel::new_with_sens(
            arena,
            rhs,
            mass_matrix,
            n_states,
            n_params,
            sens_param_indices,
        );
        let workspace = compiled.create_workspace();
        Self {
            compiled: Arc::new(compiled),
            workspace,
        }
    }

    /// Create a `ModelEvaluator` with optional algebraic sub-block expressions.
    pub fn new_with_algebraic(
        arena: &Arena,
        rhs: NodeId,
        mass_matrix: CsrData,
        n_states: usize,
        n_params: usize,
        algebraic_rhs: Option<NodeId>,
        algebraic_variable_indices: &[usize],
    ) -> Self {
        let compiled = CompiledModel::new_with_algebraic(
            arena,
            rhs,
            mass_matrix,
            n_states,
            n_params,
            algebraic_rhs,
            algebraic_variable_indices,
        );
        let workspace = compiled.create_workspace();
        Self {
            compiled: Arc::new(compiled),
            workspace,
        }
    }

    /// Create a `ModelEvaluator` with any combination of sensitivities and an algebraic sub-block.
    pub fn new_with_options(
        arena: &Arena,
        rhs: NodeId,
        mass_matrix: CsrData,
        n_states: usize,
        n_params: usize,
        options: CompiledModelOptions<'_>,
    ) -> Self {
        let compiled =
            CompiledModel::new_with_options(arena, rhs, mass_matrix, n_states, n_params, options);
        let workspace = compiled.create_workspace();
        Self {
            compiled: Arc::new(compiled),
            workspace,
        }
    }

    /// Consume the wrapper, returning the shared immutable compiled model
    /// (drops the workspace). The `Arc` is moved out without deep-copying.
    pub fn into_compiled(self) -> Arc<CompiledModel> {
        self.compiled
    }

    /// Borrow the immutable compiled model.
    pub fn compiled(&self) -> &CompiledModel {
        &self.compiled
    }

    /// Bind a fresh [`Workspace`] to an already-compiled model, the inverse of
    /// [`into_compiled`](Self::into_compiled). Nothing is recompiled, so N
    /// evaluators cost N workspaces and no lowering.
    pub fn from_compiled(compiled: Arc<CompiledModel>) -> Self {
        let workspace = compiled.create_workspace();
        Self {
            compiled,
            workspace,
        }
    }

    /// Set the cj coefficient for Jacobian computation.
    #[inline]
    pub const fn set_cj(&mut self, cj: f64) {
        self.workspace.set_cj(cj);
    }

    /// Get the current cj coefficient.
    #[inline]
    pub const fn cj(&self) -> f64 {
        self.workspace.cj()
    }

    /// Evaluate `f(t, y; p)` into `out`, using this model's own workspace.
    #[inline]
    pub fn eval_rhs(&mut self, t: f64, y: &[f64], inputs: &[f64], out: &mut [f64]) {
        self.compiled
            .eval_rhs(&mut self.workspace, t, y, inputs, out);
    }

    /// `(df/dy - cj*M) @ v`, the Newton-iteration matrix action, with `cj` taken
    /// from [`set_cj`](Self::set_cj).
    pub fn jac_mul(&mut self, t: f64, y: &[f64], inputs: &[f64], v: &[f64], out: &mut [f64]) {
        self.compiled
            .jac_mul(&mut self.workspace, t, y, inputs, v, out);
    }

    /// `df/dy @ v` with no mass term, for callers that subtract `cj*M @ v`
    /// themselves.
    pub fn jac_action(&mut self, t: f64, y: &[f64], inputs: &[f64], v: &[f64], jv: &mut [f64]) {
        self.compiled
            .jac_action(&mut self.workspace, t, y, inputs, v, jv);
    }

    /// DAE residual `r = M*y' - f(t, y; p)`, the form IDAKLU solves.
    pub fn eval_residual(&mut self, t: f64, y: &[f64], yp: &[f64], inputs: &[f64], r: &mut [f64]) {
        self.compiled
            .eval_residual(&mut self.workspace, t, y, yp, inputs, r);
    }

    /// Algebraic residual `g(t, y; p)`, or nothing when the model has no
    /// algebraic sub-block.
    pub fn eval_algebraic_residual(
        &mut self,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        output: &mut [f64],
    ) {
        self.compiled
            .eval_algebraic_residual(&mut self.workspace, t, y, inputs, output);
    }

    /// `dg/dy_alg @ v` over the algebraic states only, again a no-op without an
    /// algebraic sub-block.
    pub fn eval_algebraic_jacobian_action(
        &mut self,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        v: &[f64],
        output: &mut [f64],
    ) {
        self.compiled
            .eval_algebraic_jacobian_action(&mut self.workspace, t, y, inputs, v, output);
    }

    /// Assemble `dg/dy_alg` into `jac_data`, ordered to match
    /// [`algebraic_jacobian_row_indices`](CompiledModel::algebraic_jacobian_row_indices).
    pub fn assemble_algebraic_jacobian_into(
        &mut self,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        jac_data: &mut [f64],
    ) {
        self.compiled
            .assemble_algebraic_jacobian_into(&mut self.workspace, t, y, inputs, jac_data);
    }

    /// `df/dp` for one sensitivity parameter, where `sens_idx` is a position in
    /// [`sens_param_indices`](CompiledModel::sens_param_indices), not a global parameter
    /// index.
    pub fn eval_sens(
        &mut self,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        sens_idx: usize,
        output: &mut [f64],
    ) {
        self.compiled
            .eval_sens(&mut self.workspace, t, y, inputs, sens_idx, output);
    }

    /// `df/dp` for every configured sensitivity parameter, laid out as
    /// `output[i*n_states + j] = df_j/dp_i`, sharing one primal pass across all
    /// of them.
    pub fn eval_sens_all(&mut self, t: f64, y: &[f64], inputs: &[f64], output: &mut [f64]) {
        self.compiled
            .eval_sens_all(&mut self.workspace, t, y, inputs, output);
    }

    /// Project state sensitivities onto output-variable sensitivities. See
    /// `CompiledModel::output_sens_project`.
    pub fn output_sens_project(
        &mut self,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        y_sens: &[f64],
        out: &mut [f64],
    ) {
        self.compiled
            .output_sens_project(&mut self.workspace, t, y, inputs, y_sens, out);
    }

    /// Evaluate output variable `var_idx`, returning how many elements it wrote.
    pub fn eval_output(
        &mut self,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        var_idx: usize,
        output: &mut [f64],
    ) -> usize {
        self.compiled
            .eval_output(&mut self.workspace, t, y, inputs, var_idx, output)
    }

    /// Evaluate event `event_idx`, returning how many elements it wrote.
    pub fn eval_event(
        &mut self,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        event_idx: usize,
        output: &mut [f64],
    ) -> usize {
        self.compiled
            .eval_event(&mut self.workspace, t, y, inputs, event_idx, output)
    }

    /// Evaluate every observable of `kind` into one concatenated buffer; see
    /// [`CompiledModel::eval_observables`].
    pub fn eval_observables(
        &mut self,
        kind: ObservableKind,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        output: &mut [f64],
    ) {
        self.compiled
            .eval_observables(&mut self.workspace, kind, t, y, inputs, output);
    }

    /// Batch-evaluate every output variable over `k` trajectory points; see
    /// [`CompiledModel::eval_outputs_batch`] for the buffer layouts.
    pub fn eval_outputs_batch(
        &mut self,
        k: usize,
        ts: &[f64],
        y_cols: &[f64],
        inputs: &[f64],
        out: &mut [f64],
    ) {
        self.compiled
            .eval_outputs_batch(&mut self.workspace, k, ts, y_cols, inputs, out);
    }

    /// Assemble `df/dy - cj*M` as fresh COO triples.
    ///
    /// Allocates on every call, so it suits the Python bindings rather than a
    /// solver loop; the `_csc_into` forms below write into a caller's buffer.
    pub fn assemble_jacobian(
        &mut self,
        t: f64,
        y: &[f64],
        inputs: &[f64],
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        self.compiled
            .assemble_jacobian(&mut self.workspace, t, y, inputs)
    }

    /// Assemble `df/dy - cj*M` into a caller-owned CSC value buffer of
    /// [`nnz`](CompiledModel::nnz) elements.
    pub fn assemble_jacobian_csc_into(
        &mut self,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        jac_data: &mut [f64],
    ) {
        self.compiled
            .assemble_jacobian_csc_into(&mut self.workspace, t, y, inputs, jac_data);
    }

    /// The same assembly driven by the column coloring: one primal pass then one
    /// tangent sweep per color, which is the fast path solvers use.
    pub fn assemble_jacobian_csc_into_coloring(
        &mut self,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        jac_data: &mut [f64],
    ) {
        self.compiled.assemble_jacobian_csc_into_coloring(
            &mut self.workspace,
            t,
            y,
            inputs,
            jac_data,
        );
    }

    /// Assemble `df/dy` without folding in the mass matrix, for diffsol, which
    /// applies `M` itself.
    pub fn assemble_jacobian_csc_no_mass(
        &mut self,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        jac_data: &mut [f64],
    ) {
        self.compiled
            .assemble_jacobian_csc_no_mass(&mut self.workspace, t, y, inputs, jac_data);
    }

    /// Compile and append an output-variable expression to this model.
    /// Rebuilds the workspace after mutation to include a scratch buffer for the new expr.
    pub fn add_output(&mut self, arena: &Arena, node: NodeId) {
        // Construction-time mutation before the model is shared, so `make_mut`
        // takes the unique `Arc` in place (copy-on-write only if ever shared).
        Arc::make_mut(&mut self.compiled).add_output(arena, node);
        self.workspace = self.compiled.create_workspace();
    }

    /// Compile and append an event expression to this model.
    /// Rebuilds the workspace after mutation to include a scratch buffer for the new expr.
    pub fn add_event(&mut self, arena: &Arena, node: NodeId) {
        Arc::make_mut(&mut self.compiled).add_event(arena, node);
        self.workspace = self.compiled.create_workspace();
    }

    /// Fuse the events (see [`CompiledModel::fuse_events`]) and resize the
    /// workspace scratch to match. Call once, after all events are added.
    pub fn fuse_events(&mut self, arena: &mut Arena, event_roots: &[NodeId]) {
        Arc::make_mut(&mut self.compiled).fuse_events(arena, event_roots);
        self.workspace = self.compiled.create_workspace();
    }
}

/// CSR matrix-vector product: y = A @ x
///
/// Computes the sparse matrix-vector product for the mass matrix.
#[inline]
fn csr_matvec(csr: &CsrData, x: &[f64], y: &mut [f64]) {
    let rows = csr.shape.rows;
    for (row, y_elem) in y.iter_mut().enumerate().take(rows) {
        let start = csr.indptr[row];
        let end = csr.indptr[row + 1];
        let mut sum = 0.0;
        for idx in start..end {
            sum += csr.data[idx] * x[csr.indices[idx]];
        }
        *y_elem = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jacobian::tests::banded_nonlinear;
    use crate::node::{Node, Shape};

    #[test]
    fn data_workspace_roundtrip_eval_rhs() {
        // dy0/dt = -y0, dy1/dt = -2*y1
        let mut arena = Arena::new();
        let sv0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sv1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let neg_one = arena.alloc(Node::Scalar(-1.0));
        let neg_two = arena.alloc(Node::Scalar(-2.0));
        let r0 = arena.alloc(Node::Mul(neg_one, sv0));
        let r1 = arena.alloc(Node::Mul(neg_two, sv1));
        let rhs = arena.alloc(Node::Concat(vec![r0, r1]));
        let mass = CsrData {
            indptr: vec![0, 1, 2],
            indices: vec![0, 1],
            data: vec![1.0, 1.0],
            shape: Shape { rows: 2, cols: 2 },
        };
        let compiled = ModelEvaluator::new(&arena, rhs, mass, 2, 0).into_compiled();

        let mut ws = compiled.create_workspace();
        let mut out = vec![0.0; 2];
        compiled.eval_rhs(&mut ws, 0.0, &[3.0, 5.0], &[], &mut out);
        assert!((out[0] - (-3.0)).abs() < 1e-12);
        assert!((out[1] - (-10.0)).abs() < 1e-12);

        // Workspace scratch lengths match the compiled model's expressions.
        assert_eq!(ws.primal_scratch.len(), compiled.primal_scratch_len());
    }

    #[test]
    fn clone_shares_data_and_is_workspace_independent() {
        let mut arena = Arena::new();
        let sv = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let neg = arena.alloc(Node::Scalar(-1.0));
        let rhs = arena.alloc(Node::Mul(neg, sv));
        let model = ModelEvaluator::new(&arena, rhs, identity_mass_matrix(1), 1, 0);

        let mut clone = model.clone();
        // Immutable compiled data is shared via Arc, not deep-copied.
        assert!(Arc::ptr_eq(&model.compiled, &clone.compiled));
        // Workspaces are independent: mutating the clone leaves the original.
        clone.workspace.set_cj(7.0);
        assert!((clone.workspace.cj() - 7.0).abs() < 1e-12);
        assert!(model.workspace.cj().abs() < 1e-12);
    }

    #[test]
    fn deref_lends_the_artifact_rather_than_copying_it() {
        let mut arena = Arena::new();
        let sv = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let neg = arena.alloc(Node::Scalar(-1.0));
        let rhs = arena.alloc(Node::Mul(neg, sv));
        let model = ModelEvaluator::new(&arena, rhs, identity_mass_matrix(1), 1, 0);

        // The lent reference IS the shared artifact, so an accessor reached
        // through `Deref` cannot drift from one reached through `compiled()`.
        assert!(std::ptr::eq(&raw const *model, model.compiled()));
        assert!(std::ptr::eq(
            &raw const *model,
            Arc::as_ptr(&model.compiled)
        ));
        assert_eq!(model.n_states(), model.compiled().n_states());
    }

    #[test]
    fn inherent_workspace_methods_win_over_the_lent_ones() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Mul(two, y));
        let mut model = ModelEvaluator::new(&arena, rhs, identity_mass_matrix(2), 2, 0);

        // Same name on both types: the evaluator's own `&mut self` form is the
        // one that resolves, and it agrees with driving the artifact by hand.
        let mut through_wrapper = [0.0; 2];
        model.eval_rhs(0.0, &[1.0, 2.0], &[], &mut through_wrapper);

        let compiled = Arc::clone(&model.compiled);
        let mut ws = compiled.create_workspace();
        let mut through_artifact = [0.0; 2];
        compiled.eval_rhs(&mut ws, 0.0, &[1.0, 2.0], &[], &mut through_artifact);

        // Bitwise: the two paths must be the same evaluation, not merely close.
        assert_eq!(
            through_wrapper.map(f64::to_bits),
            through_artifact.map(f64::to_bits)
        );
    }

    #[test]
    fn mutators_repoint_the_lend() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Mul(two, y));
        let mut model = ModelEvaluator::new(&arena, rhs, identity_mass_matrix(2), 2, 0);
        assert_eq!(model.n_outputs(), 0);

        model.add_output(&arena, y);

        // Read through the lend, so `add_output`'s `Arc::make_mut` swap has to
        // be visible there and not just through a stale forwarded copy.
        assert_eq!(model.n_outputs(), 1);
        assert_eq!(model.total_output_len(), 2);
    }

    /// Create an identity mass matrix of size n.
    fn identity_mass_matrix(n: usize) -> CsrData {
        CsrData {
            shape: Shape::matrix(n, n),
            indptr: (0..=n).collect(),
            indices: (0..n).collect(),
            data: vec![1.0; n],
        }
    }

    /// Assemble `jac` directly, through its own canonical layout: what a caller
    /// holding the shared artifact rather than the model would do.
    fn assemble_through(
        jac: &JacobianData,
        mut scratch: JacobianScratch,
        t: f64,
        y: &[f64],
    ) -> Vec<f64> {
        let mut out = vec![f64::NAN; jac.layout().n_slots()];
        jac.assemble_into(&mut scratch, jac.layout(), t, y, &[], &[], &mut out);
        out
    }

    /// A mass matrix with no diagonal at all: every state is algebraic, so the
    /// algebraic sub-block spans the whole system.
    fn all_algebraic_mass_matrix(n: usize) -> CsrData {
        CsrData {
            shape: Shape::matrix(n, n),
            indptr: vec![0; n + 1],
            indices: Vec::new(),
            data: Vec::new(),
        }
    }

    /// A fully algebraic banded model, whose `dg/dy_alg` is wide enough to batch.
    fn build_batched_algebraic_model(n: usize, half_width: usize) -> ModelEvaluator {
        let (arena, root) = banded_nonlinear(n, half_width);
        let indices: Vec<usize> = (0..n).collect();
        ModelEvaluator::new_with_algebraic(
            &arena,
            root,
            all_algebraic_mass_matrix(n),
            n,
            0,
            Some(root),
            &indices,
        )
    }

    /// The algebraic block is a third adapter onto the assembly module, so it gets
    /// the lane batching the `df/dy` path has without restating it -- and the
    /// values must not depend on which width ran.
    #[test]
    #[allow(clippy::float_cmp)] // exact equality is the point
    fn the_algebraic_block_inherits_lane_batching() {
        let (n, half_width) = (64usize, 3usize);
        let mut model = build_batched_algebraic_model(n, half_width);
        let jac = model
            .algebraic_jacobian_data()
            .expect("the fixture compiles an algebraic block");
        assert!(
            JacobianScratch::new(&jac).lane_width() > 1,
            "the algebraic fixture must batch"
        );

        let y: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.037, 0.41)).collect();
        let mut batched = vec![f64::NAN; model.algebraic_jacobian_nnz()];
        model.assemble_algebraic_jacobian_into(0.5, &y, &[], &mut batched);

        // The same artifact swept one colour per walk: the reference the batched
        // sweep has to reproduce.
        let scalar = assemble_through(&jac, JacobianScratch::scalar(&jac), 0.5, &y);
        assert_eq!(batched, scalar);
    }

    /// The standalone `dg/dy_alg` handle is the artifact the model compiled, not a
    /// second tangent transform of the same expression, so the two cannot drift
    /// in pattern, colouring or value.
    #[test]
    #[allow(clippy::float_cmp)] // exact equality is the point
    fn the_standalone_algebraic_jacobian_is_the_compiled_artifact() {
        let n = 24usize;
        let mut model = build_batched_algebraic_model(n, 2);
        let first = model.algebraic_jacobian_data().expect("algebraic block");
        let second = model.algebraic_jacobian_data().expect("algebraic block");
        assert!(
            Arc::ptr_eq(&first, &second),
            "the handle must lend the compiled artifact, never rebuild it"
        );

        let (rows, cols) = first.coo_global_indices();
        assert_eq!(rows, model.algebraic_jacobian_row_indices());
        assert_eq!(cols, model.algebraic_jacobian_col_indices());

        let y: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.13, 0.7)).collect();
        let mut through_model = vec![f64::NAN; model.algebraic_jacobian_nnz()];
        model.assemble_algebraic_jacobian_into(0.25, &y, &[], &mut through_model);

        let through_handle = assemble_through(&first, JacobianScratch::new(&first), 0.25, &y);
        assert_eq!(through_model, through_handle);
    }

    #[test]
    fn test_compiled_model_new() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Mul(two, y)); // f(y) = 2*y

        let mass = identity_mass_matrix(2);

        let model = ModelEvaluator::new(&arena, rhs, mass, 2, 0);

        assert_eq!(model.n_states(), 2);
        assert_eq!(model.n_params(), 0);
        assert_eq!(model.output_len(), 2);
        assert!(model.cj().abs() < f64::EPSILON);
    }

    #[test]
    fn test_compiled_model_eval_rhs() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Mul(two, y)); // f(y) = 2*y

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 2, 0);

        let y_vals = [1.0, 2.0];
        let mut output = [0.0, 0.0];
        model.eval_rhs(0.0, &y_vals, &[], &mut output);

        // 2 * [1, 2] = [2, 4]
        assert!((output[0] - 2.0).abs() < 1e-14);
        assert!((output[1] - 4.0).abs() < 1e-14);
    }

    #[test]
    fn test_compiled_model_jac_mul_no_mass() {
        // f(y) = 2*y, so df/dy = 2*I
        // With cj=0: (df/dy - 0*M) @ v = 2*I @ v = 2*v
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Mul(two, y));

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 2, 0);

        // cj = 0, so mass matrix doesn't contribute
        model.set_cj(0.0);

        let y_vals = [1.0, 2.0];
        let v = [1.0, 0.0];
        let mut output = [0.0, 0.0];
        model.jac_mul(0.0, &y_vals, &[], &v, &mut output);

        // df/dy @ [1, 0] = 2 * [1, 0] = [2, 0]
        assert!(
            (output[0] - 2.0).abs() < 1e-14,
            "Expected 2.0, got {}",
            output[0]
        );
        assert!(output[1].abs() < 1e-14, "Expected 0.0, got {}", output[1]);
    }

    #[test]
    fn test_compiled_model_jac_mul_with_mass() {
        // f(y) = 2*y, so df/dy = 2*I
        // With cj=0.5 and M=I: (df/dy - cj*M) = 2*I - 0.5*I = 1.5*I
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Mul(two, y));

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 2, 0);

        model.set_cj(0.5);

        let y_vals = [1.0, 2.0];
        let v = [1.0, 0.0];
        let mut output = [0.0, 0.0];
        model.jac_mul(0.0, &y_vals, &[], &v, &mut output);

        // (2 - 0.5) * [1, 0] = [1.5, 0]
        assert!(
            (output[0] - 1.5).abs() < 1e-14,
            "Expected 1.5, got {}",
            output[0]
        );
        assert!(output[1].abs() < 1e-14, "Expected 0.0, got {}", output[1]);
    }

    #[test]
    fn test_compiled_model_jac_mul_second_direction() {
        // Same model, but test v = [0, 1]
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Mul(two, y));

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 2, 0);

        model.set_cj(0.5);

        let y_vals = [1.0, 2.0];
        let v = [0.0, 1.0];
        let mut output = [0.0, 0.0];
        model.jac_mul(0.0, &y_vals, &[], &v, &mut output);

        // (2 - 0.5) * [0, 1] = [0, 1.5]
        assert!(output[0].abs() < 1e-14, "Expected 0.0, got {}", output[0]);
        assert!(
            (output[1] - 1.5).abs() < 1e-14,
            "Expected 1.5, got {}",
            output[1]
        );
    }

    #[test]
    fn test_compiled_model_sparsity() {
        // f(y) = [sin(y0), sin(y1)]: diagonal but state-dependent, so the
        // coloring is what has to make assembly cheap here.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let rhs = arena.alloc(Node::Sin(y));

        let mass = identity_mass_matrix(2);
        let model = ModelEvaluator::new(&arena, rhs, mass, 2, 0);

        // Diagonal Jacobian should only need 1 color
        assert_eq!(
            model.coloring().n_colors,
            1,
            "Diagonal should need 1 color, got {}",
            model.coloring().n_colors
        );
    }

    /// Nonlinear banded ODE whose 5-wide stencil needs enough colours to
    /// engage the batched sweep.
    fn build_batched_model(n: usize) -> ModelEvaluator {
        build_banded_model(n, 2)
    }

    /// Nonlinear ODE with a one-sided stencil, whose colour count is `reach + 1`
    /// -- so unlike the symmetric builder it can land on an even count, which is
    /// what puts a tail of exactly two colours under test.
    fn build_one_sided_model(n: usize, reach: usize) -> ModelEvaluator {
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
                for offset in 1..=reach {
                    let term = arena.alloc(Node::Sin(states[(i + offset).min(n - 1)]));
                    acc = arena.alloc(Node::Add(acc, term));
                }
                acc
            })
            .collect();
        let rhs = arena.alloc(Node::Concat(rows));
        ModelEvaluator::new(&arena, rhs, identity_mass_matrix(n), n, 0)
    }

    /// Nonlinear banded ODE of a chosen half-bandwidth, so the colour count --
    /// and with it the block/tail split -- can be dialled.
    fn build_banded_model(n: usize, half_width: usize) -> ModelEvaluator {
        let (arena, rhs) = banded_nonlinear(n, half_width);
        ModelEvaluator::new(&arena, rhs, identity_mass_matrix(n), n, 0)
    }

    /// A tail narrower than the lane width takes its own narrower walk, so the
    /// block loop must still reproduce the unbatched tape column for column.
    #[test]
    fn a_narrow_tail_block_matches_the_unbatched_tape() {
        // Symmetric half-widths give odd colour counts, one-sided reaches give
        // even ones, so between them the tails cover every shape: scalar tails
        // under lane 4 and lane 8, tails of two and three (which straddle
        // MIN_PADDED_TAIL), a narrower vector block, and a vector block followed
        // by several scalar walks.
        let fixtures = [
            (64usize, 2usize, true),
            (64, 4, true),
            (64, 6, true),
            (64, 7, true),
        ]
        .into_iter()
        .chain([
            (64, 5, false),
            (64, 9, false),
            (64, 11, false),
            (64, 13, false),
        ]);
        for (n, reach, symmetric) in fixtures {
            let half_width = reach;
            let mut model = if symmetric {
                build_banded_model(n, half_width)
            } else {
                build_one_sided_model(n, reach)
            };
            let stats = model.jacobian_stats();
            let (colors, lanes) = (stats.n_colors, stats.jac_lane_width);
            assert!(lanes > 1, "n={n} w={half_width} must batch");
            assert_ne!(
                colors % lanes,
                0,
                "n={n} w={half_width} must leave a partial tail block"
            );

            // cj = 0 keeps this a pure df/dy comparison: no mass postpass.
            model.set_cj(0.0);
            let y: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.037, 0.41)).collect();
            let mut assembled = vec![f64::NAN; model.nnz()];
            model.assemble_jacobian_csc_into(0.0, &y, &[], &mut assembled);

            let (colptr, rowind) = {
                let csc = model.csc_sparsity();
                (csc.colptr.clone(), csc.rowind.clone())
            };
            let mut column = vec![0.0; n];
            let mut seed = vec![0.0; n];
            for col in 0..n {
                seed.fill(0.0);
                seed[col] = 1.0;
                model.jac_action(0.0, &y, &[], &seed, &mut column);
                for k in colptr[col]..colptr[col + 1] {
                    let (row, got) = (rowind[k], assembled[k]);
                    let want = column[row];
                    assert!(
                        got.to_bits() == want.to_bits() || (got == 0.0 && want == 0.0),
                        "n={n} reach={reach} entry ({row}, {col}): batched {got}, tape {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn batched_seeds_are_restored_between_assemblies() {
        // The batched sweep clears only the lanes it set, so a stale 1.0 left
        // behind would silently add another column into a later block.
        let n = 24;
        let mut model = build_batched_model(n);
        assert!(model.jacobian_stats().jac_lane_width > 1, "needs batching");
        model.set_cj(0.25);

        let states: [Vec<f64>; 2] = [
            (0..n).map(|i| (i as f64).mul_add(0.11, 0.3)).collect(),
            (0..n).map(|i| (i as f64).mul_add(-0.07, 0.9)).collect(),
        ];
        let mut repeated = vec![0.0; model.nnz()];
        for _ in 0..3 {
            for y in &states {
                model.assemble_jacobian_csc_into(0.0, y, &[], &mut repeated);
            }
        }

        let mut fresh_model = build_batched_model(n);
        fresh_model.set_cj(0.25);
        let mut fresh = vec![0.0; fresh_model.nnz()];
        fresh_model.assemble_jacobian_csc_into(0.0, &states[1], &[], &mut fresh);
        assert!(
            repeated
                .iter()
                .zip(&fresh)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "a repeated assembly must match a first one bit for bit"
        );
    }

    #[test]
    fn a_linear_model_assembles_without_sweeping() {
        // f(y) = [y0, y1]: every entry folds at compile time, so the coloring
        // seeds nothing and assembly is a table write.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let mut model = ModelEvaluator::new(&arena, y, identity_mass_matrix(2), 2, 0);

        let stats = model.jacobian_stats();
        assert_eq!(stats.n_colors, 0);
        assert_eq!(stats.n_swept_columns, 0);
        assert_eq!(stats.n_constant_entries, 2);

        model.set_cj(0.25);
        let mut data = vec![f64::NAN; model.nnz()];
        model.assemble_jacobian_csc_into_coloring(0.0, &[7.0, -3.0], &[], &mut data);
        assert_eq!(data, vec![0.75, 0.75]);
    }

    #[test]
    fn test_compiled_model_nonlinear() {
        // f(y) = y^2, so df/dy = 2*y (diagonal)
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Pow(y, two)); // y^2

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 2, 0);

        model.set_cj(0.0);

        // At y = [3, 4], df/dy = diag([6, 8])
        let y_vals = [3.0, 4.0];
        let v = [1.0, 1.0];
        let mut output = [0.0, 0.0];
        model.jac_mul(0.0, &y_vals, &[], &v, &mut output);

        // df/dy @ [1, 1] = [6, 8]
        assert!(
            (output[0] - 6.0).abs() < 1e-12,
            "Expected 6.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - 8.0).abs() < 1e-12,
            "Expected 8.0, got {}",
            output[1]
        );
    }

    #[test]
    fn test_compiled_model_with_params() {
        // f(y) = k * y where k is a parameter
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let k = arena.alloc(Node::InputParameter {
            name: "k".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let rhs = arena.alloc(Node::Mul(k, y));

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 2, 1);

        assert_eq!(model.n_params(), 1);

        // Test eval_rhs with k=3
        let y_vals = [1.0, 2.0];
        let inputs = [3.0];
        let mut output = [0.0, 0.0];
        model.eval_rhs(0.0, &y_vals, &inputs, &mut output);

        // 3 * [1, 2] = [3, 6]
        assert!((output[0] - 3.0).abs() < 1e-14);
        assert!((output[1] - 6.0).abs() < 1e-14);

        // Test jac_mul: df/dy = k*I = 3*I
        model.set_cj(1.0);
        let v = [1.0, 0.0];
        model.jac_mul(0.0, &y_vals, &inputs, &v, &mut output);

        // (3 - 1) * [1, 0] = [2, 0]
        assert!(
            (output[0] - 2.0).abs() < 1e-14,
            "Expected 2.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_csr_matvec() {
        // Test sparse matrix-vector product
        // M = [[2, 0], [0, 3]]
        let csr = CsrData {
            shape: Shape::matrix(2, 2),
            indptr: vec![0, 1, 2],
            indices: vec![0, 1],
            data: vec![2.0, 3.0],
        };

        let x = [1.0, 2.0];
        let mut y = [0.0, 0.0];
        csr_matvec(&csr, &x, &mut y);

        // [2*1, 3*2] = [2, 6]
        assert!((y[0] - 2.0).abs() < 1e-14);
        assert!((y[1] - 6.0).abs() < 1e-14);
    }

    #[test]
    fn test_csr_matvec_tridiagonal() {
        // M = [[1, 2, 0], [3, 4, 5], [0, 6, 7]]
        let csr = CsrData {
            shape: Shape::matrix(3, 3),
            indptr: vec![0, 2, 5, 7],
            indices: vec![0, 1, 0, 1, 2, 1, 2],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        };

        let x = [1.0, 2.0, 3.0];
        let mut y = [0.0, 0.0, 0.0];
        csr_matvec(&csr, &x, &mut y);

        // y = [1*1 + 2*2, 3*1 + 4*2 + 5*3, 6*2 + 7*3]
        assert!((y[0] - 5.0).abs() < 1e-14);
        assert!((y[1] - 26.0).abs() < 1e-14);
        assert!((y[2] - 33.0).abs() < 1e-14);
    }

    #[test]
    fn test_compiled_model_debug() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Mul(two, y));

        let mass = identity_mass_matrix(2);
        let model = ModelEvaluator::new(&arena, rhs, mass, 2, 0);

        let debug_str = format!("{model:?}");
        assert!(debug_str.contains("ModelEvaluator"));
        assert!(debug_str.contains("n_states: 2"));
    }

    #[test]
    fn test_compiled_model_residual() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Mul(two, y)); // f(y) = 2*y

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 2, 0);

        let y_vals = [1.0, 2.0];
        let yp = [3.0, 4.0];
        let mut r = [0.0, 0.0];
        model.eval_residual(0.0, &y_vals, &yp, &[], &mut r);

        // M*y' - f(y) = I*[3, 4] - 2*[1, 2] = [3, 4] - [2, 4] = [1, 0]
        assert!((r[0] - 1.0).abs() < 1e-14, "Expected 1.0, got {}", r[0]);
        assert!(r[1].abs() < 1e-14, "Expected 0.0, got {}", r[1]);
    }

    #[test]
    fn test_compiled_model_with_zero_tangent_direction() {
        // When v = 0, zero propagation should optimize away most of the JVP computation
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Mul(two, y));

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 2, 0);
        model.set_cj(0.5);

        let y_vals = [1.0, 2.0];
        let v = [0.0, 0.0]; // Zero tangent direction
        let mut output = [0.0, 0.0];
        model.jac_mul(0.0, &y_vals, &[], &v, &mut output);

        // Result should be [0, 0]
        assert!(output[0].abs() < 1e-14);
        assert!(output[1].abs() < 1e-14);
    }

    #[test]
    fn test_compiled_model_residual_nonidentity_mass() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });

        // Non-identity mass matrix: M = diag([2, 3])
        let mass = CsrData {
            shape: Shape::matrix(2, 2),
            indptr: vec![0, 1, 2],
            indices: vec![0, 1],
            data: vec![2.0, 3.0],
        };

        let mut model = ModelEvaluator::new(&arena, y, mass, 2, 0);

        let y_vals = [1.0, 2.0];
        let yp = [3.0, 4.0];
        let mut r = [0.0, 0.0];
        model.eval_residual(0.0, &y_vals, &yp, &[], &mut r);

        // M*y' - f(y) = [2*3, 3*4] - [1, 2] = [6, 12] - [1, 2] = [5, 10]
        assert!((r[0] - 5.0).abs() < 1e-14, "Expected 5.0, got {}", r[0]);
        assert!((r[1] - 10.0).abs() < 1e-14, "Expected 10.0, got {}", r[1]);
    }

    #[test]
    fn test_algebraic_ids_inferred_from_mass() {
        // Mass matrix: row 0 has diag, row 1 missing diag, row 2 has diag.
        // Row 1 is therefore algebraic.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let mass = CsrData {
            shape: Shape::matrix(3, 3),
            indptr: vec![0, 1, 1, 2],
            indices: vec![0, 2],
            data: vec![1.0, 1.0],
        };
        let model = ModelEvaluator::new(&arena, y, mass, 3, 0);
        assert_eq!(model.algebraic_ids(), &[false, true, false]);
    }

    #[test]
    fn test_algebraic_ids_f64_for_ida() {
        // 1.0 = differential, 0.0 = algebraic (matches IDA's id vector format).
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let mass = CsrData {
            shape: Shape::matrix(3, 3),
            indptr: vec![0, 1, 1, 2],
            indices: vec![0, 2],
            data: vec![1.0, 1.0],
        };
        let model = ModelEvaluator::new(&arena, y, mass, 3, 0);
        let mut ids = vec![0.0; 3];
        model.algebraic_ids_f64(&mut ids);
        assert_eq!(ids, vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_algebraic_ids_identity_mass_all_differential() {
        // Identity mass = pure ODE: every state is differential.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 4 });
        let model = ModelEvaluator::new(&arena, y, identity_mass_matrix(4), 4, 0);
        assert_eq!(model.algebraic_ids(), &[false, false, false, false]);
    }

    /// Shared fixture: f(y) = k * y^2 with k as `InputParameter` index 0.
    fn build_sens_test_model() -> (ModelEvaluator, [f64; 2]) {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let k = arena.alloc(Node::InputParameter {
            name: "k".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let two = arena.alloc(Node::Scalar(2.0));
        let y_sq = arena.alloc(Node::Pow(y, two));
        let rhs = arena.alloc(Node::Mul(k, y_sq));

        let mass = identity_mass_matrix(2);
        let model = ModelEvaluator::new_with_sens(&arena, rhs, mass, 2, 1, &[0]);
        (model, [3.0, 4.0])
    }

    /// Shared fixture: f(y) = k1 * y; sensitivities for both k1 (idx 0) and k2 (idx 1).
    fn build_two_param_sens_model() -> (ModelEvaluator, [f64; 2]) {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let k1 = arena.alloc(Node::InputParameter {
            name: "k1".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let _k2 = arena.alloc(Node::InputParameter {
            name: "k2".to_string(),
            index: 1,
            offset: 1,
            width: 1,
        });
        let rhs = arena.alloc(Node::Mul(k1, y));

        let mass = identity_mass_matrix(2);
        let model = ModelEvaluator::new_with_sens(&arena, rhs, mass, 2, 2, &[0, 1]);
        (model, [3.0, 4.0])
    }

    #[test]
    fn test_sens_expr_compiled_with_split_eval() {
        // Same fixture as test_eval_sens_matches_analytical (f = k*y^2, sens wrt k).
        let (model, _y) = build_sens_test_model();
        let sens_expr = model.compiled.sens_expr.as_ref().expect("sens expr");
        assert!(
            sens_expr.has_split_eval(),
            "sens_expr must be split-eval compiled so the primal runs once per (t,y)"
        );
    }

    #[test]
    fn test_eval_sens_matches_analytical() {
        // f(y) = k * y^2 with k as InputParameter index 0.  ∂f/∂k = y^2.
        let (mut model, y_vals) = build_sens_test_model();
        assert_eq!(model.n_sens_params(), 1);

        // y = [3, 4], k = 7 -> ∂f/∂k = y^2 = [9, 16]
        let inputs = [7.0];
        let mut out = [0.0; 2];
        model.eval_sens(0.0, &y_vals, &inputs, 0, &mut out);
        assert!((out[0] - 9.0).abs() < 1e-10, "expected 9, got {}", out[0]);
        assert!((out[1] - 16.0).abs() < 1e-10, "expected 16, got {}", out[1]);
    }

    #[test]
    fn test_eval_sens_all_layout() {
        // f(y) = k1 * y; sensitivities for both k1 (idx 0) and k2 (idx 1).
        // ∂f/∂k1 = y, ∂f/∂k2 = 0.
        let (mut model, y_vals) = build_two_param_sens_model();

        let inputs = [5.0, 9.0];
        let mut out = vec![-1.0; 4];
        model.eval_sens_all(0.0, &y_vals, &inputs, &mut out);

        // Layout: out[i*n_states + j] = ∂f_j/∂p_i
        // i=0 (k1): [3, 4]; i=1 (k2): [0, 0]
        assert!((out[0] - 3.0).abs() < 1e-10);
        assert!((out[1] - 4.0).abs() < 1e-10);
        assert!(out[2].abs() < 1e-10);
        assert!(out[3].abs() < 1e-10);
    }

    #[test]
    fn test_eval_sens_all_matches_single_column_eval() {
        // eval_sens_all (primal-once batch) must agree with per-column eval_sens
        // (full-stream) exactly, same tape, same arithmetic.
        let (mut model, y) = build_two_param_sens_model();
        let n = model.compiled.n_states;
        let k = model.compiled.n_sens_params();
        let mut batched = vec![0.0; n * k];
        model.eval_sens_all(0.0, &y, &[1.5, 2.5], &mut batched);
        for i in 0..k {
            let mut single = vec![0.0; n];
            model.eval_sens(0.0, &y, &[1.5, 2.5], i, &mut single);
            assert_eq!(
                &batched[i * n..(i + 1) * n],
                &single[..],
                "column {i} diverged"
            );
        }
    }

    /// Shared fixture whose requested subset is not a leading prefix.
    ///
    /// `f(y) = k0 * y + k1 * y^2` and `H = k0 * y[0] + k1 * y[1]`, with sensitivities
    /// requested for k1 (index 1) alone. Every parameter derivative is distinct and
    /// non-zero, so a seed landing on the wrong parameter shows up as a wrong value
    /// rather than a zero.
    fn build_trailing_param_sens_model() -> (CompiledModel, [f64; 2]) {
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
        let k0 = arena.alloc(Node::InputParameter {
            name: "k0".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let k1 = arena.alloc(Node::InputParameter {
            name: "k1".to_string(),
            index: 1,
            offset: 1,
            width: 1,
        });
        let two = arena.alloc(Node::Scalar(2.0));
        let y_squared = arena.alloc(Node::Pow(y, two));
        let linear = arena.alloc(Node::Mul(k0, y));
        let quadratic = arena.alloc(Node::Mul(k1, y_squared));
        let rhs = arena.alloc(Node::Add(linear, quadratic));

        let out_left = arena.alloc(Node::Mul(k0, y0));
        let out_right = arena.alloc(Node::Mul(k1, y1));
        let out_node = arena.alloc(Node::Add(out_left, out_right));

        let mut data =
            CompiledModel::new_with_sens(&arena, rhs, identity_mass_matrix(2), 2, 2, &[1]);
        data.add_output(&arena, out_node);
        (data, [3.0, 4.0])
    }

    #[test]
    fn sens_action_seeds_in_parameter_space() {
        // df/dk1 = y^2. The seed is indexed by parameter, so its 1.0 sits at index 1;
        // reading it as a subset-space seed would pick up the 0.0 at index 0 instead.
        let (model, y) = build_trailing_param_sens_model();
        let mut ws = model.create_workspace();
        let mut got = [0.0; 2];
        model.sens_action(
            &mut ws,
            0.0,
            &y,
            &[5.0, 9.0],
            &[0, 1],
            &[0.0, 1.0],
            &mut got,
        );
        assert!((got[0] - 9.0).abs() < 1e-10, "expected 9, got {}", got[0]);
        assert!((got[1] - 16.0).abs() < 1e-10, "expected 16, got {}", got[1]);
    }

    #[test]
    fn sens_action_maps_a_subset_seed_to_its_global_parameter() {
        // A one-entry subset seed naming k1 must differentiate k1, not whatever
        // parameter sits at position 0 of the seed.
        let (model, y) = build_trailing_param_sens_model();
        let mut ws = model.create_workspace();
        let inputs = [5.0, 9.0];

        let mut subset = [0.0; 2];
        model.sens_action(&mut ws, 0.0, &y, &inputs, &[1], &[1.0], &mut subset);

        let mut global = [0.0; 2];
        model.sens_action(&mut ws, 0.0, &y, &inputs, &[0, 1], &[0.0, 1.0], &mut global);

        assert_eq!(subset.as_slice(), global.as_slice());
        // df/dk1 = y^2 = [9, 16]
        assert!((subset[0] - 9.0).abs() < 1e-10, "got {}", subset[0]);
        assert!((subset[1] - 16.0).abs() < 1e-10, "got {}", subset[1]);
    }

    #[test]
    fn an_output_sens_action_maps_a_subset_seed_to_its_global_parameter() {
        // H = k0*y0 + k1*y1, so dH/dk1 = y1 = 4.0. Reading the seed as a prefix
        // would give dH/dk0 = y0 = 3.0 instead.
        let (model, y) = build_trailing_param_sens_model();
        let mut ws = model.create_workspace();
        let mut got = vec![0.0; model.total_output_len()];
        model.observable_sens_action(
            &mut ws,
            ObservableKind::Outputs,
            0.0,
            &y,
            &[5.0, 9.0],
            &[1],
            &[1.0],
            &mut got,
        );
        assert!((got[0] - 4.0).abs() < 1e-10, "expected 4, got {}", got[0]);
    }

    #[test]
    fn sens_tangent_column_is_named_by_parameter_index() {
        // Column 0 is k0, whose df/dk0 = y, even though only k1 was registered: the
        // compiled tangent is generic over the parameters and the seed picks the column.
        let (model, y) = build_trailing_param_sens_model();
        let mut ws = model.create_workspace();
        model.sens_primal_pass(&mut ws, 0.0, &y, &[5.0, 9.0]);
        let mut got = [0.0; 2];
        model.sens_tangent_column(&mut ws, 0, &mut got);
        assert!((got[0] - 3.0).abs() < 1e-10, "expected 3, got {}", got[0]);
        assert!((got[1] - 4.0).abs() < 1e-10, "expected 4, got {}", got[1]);
    }

    #[test]
    fn output_sens_project_seeds_the_requested_parameter() {
        // dH/dk1 = y[1] = 4 and dH/dk0 = y[0] = 3, so projecting the one requested
        // column must give 4: the subset-to-parameter mapping belongs to the caller.
        let (model, y) = build_trailing_param_sens_model();
        let mut ws = model.create_workspace();
        let y_sens = vec![0.0; model.n_states];
        let mut got = vec![0.0; model.total_output_len()];
        model.output_sens_project(&mut ws, 0.0, &y, &[5.0, 9.0], &y_sens, &mut got);
        assert_eq!(got.len(), 1);
        assert!((got[0] - 4.0).abs() < 1e-10, "expected 4, got {}", got[0]);
    }

    #[test]
    fn test_n_sens_params_zero_when_no_sensitivities() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let model = ModelEvaluator::new(&arena, y, identity_mass_matrix(2), 2, 0);
        assert_eq!(model.n_sens_params(), 0);
    }

    #[test]
    fn test_eval_sens_all_noop_without_sensitivities() {
        // Zero configured sensitivities: eval_sens_all must return without
        // touching sens_expr (absent), matching the old 0-column loop.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let mut model = ModelEvaluator::new(&arena, y, identity_mass_matrix(2), 2, 0);
        let mut out: [f64; 0] = [];
        model.eval_sens_all(0.0, &[3.0, 4.0], &[], &mut out);
    }

    #[test]
    fn test_add_output_compiles_and_evaluates() {
        // f(y) = y is the rhs; output_var = 2 * y[0].
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = arena.alloc(Node::Index {
            child: y_full,
            start: 0,
            end: 1,
        });
        let two = arena.alloc(Node::Scalar(2.0));
        let var0 = arena.alloc(Node::Mul(two, y0));

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, y_full, mass, 2, 0);
        model.add_output(&arena, var0);

        assert_eq!(model.n_outputs(), 1);
        assert_eq!(model.output_len_at(0), 1);

        // var0(y=[3, 4]) = 2 * 3 = 6
        let mut out = [0.0; 1];
        let written = model.eval_output(0.0, &[3.0, 4.0], &[], 0, &mut out);
        assert_eq!(written, 1);
        assert!((out[0] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_n_outputs_zero_initially() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let model = ModelEvaluator::new(&arena, y, identity_mass_matrix(2), 2, 0);
        assert_eq!(model.n_outputs(), 0);
    }

    #[test]
    fn test_outputs_compose_with_sensitivities() {
        // Verify add_output works on a model that already has sensitivities.
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let k = arena.alloc(Node::InputParameter {
            name: "k".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let rhs = arena.alloc(Node::Mul(k, y_full));
        let y0 = arena.alloc(Node::Index {
            child: y_full,
            start: 0,
            end: 1,
        });

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new_with_sens(&arena, rhs, mass, 2, 1, &[0]);
        model.add_output(&arena, y0);

        assert_eq!(model.n_sens_params(), 1);
        assert_eq!(model.n_outputs(), 1);
        assert_eq!(model.output_len_at(0), 1);

        let y_vals = [3.0, 4.0];
        let inputs = [2.0];

        // ∂f/∂k = y -> [3, 4]
        let mut sens_out = [0.0; 2];
        model.eval_sens(0.0, &y_vals, &inputs, 0, &mut sens_out);
        assert!((sens_out[0] - 3.0).abs() < 1e-12);
        assert!((sens_out[1] - 4.0).abs() < 1e-12);

        // var0 = y[0] = 3
        let mut out = [0.0; 1];
        model.eval_output(0.0, &y_vals, &inputs, 0, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_new_with_options_compiles_algebraic_and_sensitivities() {
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let y1 = arena.alloc(Node::Index {
            child: y_full,
            start: 1,
            end: 2,
        });
        let y2 = arena.alloc(Node::Index {
            child: y_full,
            start: 2,
            end: 3,
        });
        let k = arena.alloc(Node::InputParameter {
            name: "k".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let rhs = arena.alloc(Node::Mul(k, y_full));
        let algebraic_sum = arena.alloc(Node::Add(y1, y2));
        let algebraic_prod = arena.alloc(Node::Mul(y1, y2));
        let alg = arena.alloc(Node::Concat(vec![algebraic_sum, algebraic_prod]));
        let mass = CsrData {
            shape: Shape::matrix(3, 3),
            indptr: vec![0, 1, 1, 1],
            indices: vec![0],
            data: vec![1.0],
        };

        let options = CompiledModelOptions::new()
            .with_sensitivities(&[0])
            .with_algebraic(CompiledModelAlgebraicBlock::new(alg, &[1, 2]));
        let mut model = ModelEvaluator::new_with_options(&arena, rhs, mass, 3, 1, options);

        assert_eq!(model.n_sens_params(), 1);
        assert!(model.has_algebraic());
        assert_eq!(model.n_algebraic(), 2);

        let y_vals = [3.0, 4.0, 5.0];
        let inputs = [2.0];

        let mut sens_out = [0.0; 3];
        model.eval_sens(0.0, &y_vals, &inputs, 0, &mut sens_out);
        for (actual, expected) in sens_out.iter().zip(y_vals.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }

        let mut algebraic_out = [0.0; 2];
        model.eval_algebraic_residual(0.0, &y_vals, &inputs, &mut algebraic_out);
        assert!((algebraic_out[0] - 9.0).abs() < 1e-12);
        assert!((algebraic_out[1] - 20.0).abs() < 1e-12);
    }

    #[test]
    fn test_algebraic_jac_action_uses_algebraic_index_mapping() {
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let y0 = arena.alloc(Node::Index {
            child: y_full,
            start: 0,
            end: 1,
        });
        let y1 = arena.alloc(Node::Index {
            child: y_full,
            start: 1,
            end: 2,
        });
        let y2 = arena.alloc(Node::Index {
            child: y_full,
            start: 2,
            end: 3,
        });
        let alg0_mul = arena.alloc(Node::Mul(y1, y2));
        let alg0 = arena.alloc(Node::Add(y0, alg0_mul));
        let two = arena.alloc(Node::Scalar(2.0));
        let y2_sq = arena.alloc(Node::Pow(y2, two));
        let alg1 = arena.alloc(Node::Add(y1, y2_sq));
        let alg = arena.alloc(Node::Concat(vec![alg0, alg1]));
        let mass = CsrData {
            shape: Shape::matrix(3, 3),
            indptr: vec![0, 1, 1, 1],
            indices: vec![0],
            data: vec![1.0],
        };

        let mut model =
            ModelEvaluator::new_with_algebraic(&arena, y_full, mass, 3, 0, Some(alg), &[1, 2]);
        let y_vals = [10.0, 2.0, 3.0];
        let v_alg = [5.0, 7.0];
        let mut jv = [0.0; 2];
        model.eval_algebraic_jacobian_action(0.0, &y_vals, &[], &v_alg, &mut jv);

        assert!((jv[0] - 29.0).abs() < 1e-12, "expected 29, got {}", jv[0]);
        assert!((jv[1] - 47.0).abs() < 1e-12, "expected 47, got {}", jv[1]);
    }

    #[test]
    fn test_assemble_alg_jacobian_into_matches_expected_sparse_values() {
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let y0 = arena.alloc(Node::Index {
            child: y_full,
            start: 0,
            end: 1,
        });
        let y1 = arena.alloc(Node::Index {
            child: y_full,
            start: 1,
            end: 2,
        });
        let y2 = arena.alloc(Node::Index {
            child: y_full,
            start: 2,
            end: 3,
        });
        let alg0_mul = arena.alloc(Node::Mul(y1, y2));
        let alg0 = arena.alloc(Node::Add(y0, alg0_mul));
        let two = arena.alloc(Node::Scalar(2.0));
        let y2_sq = arena.alloc(Node::Pow(y2, two));
        let alg1 = arena.alloc(Node::Add(y1, y2_sq));
        let alg = arena.alloc(Node::Concat(vec![alg0, alg1]));
        let mass = CsrData {
            shape: Shape::matrix(3, 3),
            indptr: vec![0, 1, 1, 1],
            indices: vec![0],
            data: vec![1.0],
        };

        let mut model =
            ModelEvaluator::new_with_algebraic(&arena, y_full, mass, 3, 0, Some(alg), &[1, 2]);

        // The block shares `JacobianData`'s CSC ordering, so the triplet runs
        // column-major: global column 1, then 2. Both C++ Newton drivers
        // bucket-sort it, so the order is ours to choose -- but it is published
        // alongside the values and must stay aligned with them.
        assert_eq!(model.algebraic_jacobian_row_indices(), &[0, 1, 0, 1]);
        assert_eq!(model.algebraic_jacobian_col_indices(), &[1, 1, 2, 2]);

        let mut jac = vec![0.0; model.algebraic_jacobian_nnz()];
        model.assemble_algebraic_jacobian_into(0.0, &[10.0, 2.0, 3.0], &[], &mut jac);
        // dg/d(y1, y2) = [[y2, y1], [1, 2*y2]] at (y1, y2) = (2, 3).
        assert_eq!(jac, vec![3.0, 1.0, 2.0, 6.0]);
    }

    #[test]
    fn test_jacobian_with_zero_elimination() {
        // Build a model where some Jacobian columns are structurally zero
        // Verify that zero propagation eliminates dead computation
        let mut arena = Arena::new();

        // f(y) = [y0 * y1, y0 + c], so df/dy = [[y1, y0], [1, 0]] and
        // df1/dy1 is a structural zero.
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = arena.alloc(Node::Index {
            child: y_full,
            start: 0,
            end: 1,
        });
        let y1 = arena.alloc(Node::Index {
            child: y_full,
            start: 1,
            end: 2,
        });
        let c = arena.alloc(Node::Scalar(5.0));
        let prod = arena.alloc(Node::Mul(y0, y1));
        let sum = arena.alloc(Node::Add(y0, c));
        let rhs = arena.alloc(Node::Concat(vec![prod, sum]));

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 2, 0);
        model.set_cj(0.0);

        // Test jac_mul with [1, 0] - should give [y1, 1]
        let y_vals = [2.0, 3.0];
        let v1 = [1.0, 0.0];
        let mut output = [0.0, 0.0];
        model.jac_mul(0.0, &y_vals, &[], &v1, &mut output);

        assert!(
            (output[0] - 3.0).abs() < 1e-12,
            "df1/dy0 = y1 = 3, got {}",
            output[0]
        );
        assert!(
            (output[1] - 1.0).abs() < 1e-12,
            "df2/dy0 = 1, got {}",
            output[1]
        );

        // Test jac_mul with [0, 1] - should give [y0, 0]
        let v2 = [0.0, 1.0];
        model.jac_mul(0.0, &y_vals, &[], &v2, &mut output);

        assert!(
            (output[0] - 2.0).abs() < 1e-12,
            "df1/dy1 = y0 = 2, got {}",
            output[0]
        );
        assert!(output[1].abs() < 1e-12, "df2/dy1 = 0, got {}", output[1]);
    }

    #[test]
    fn test_assemble_jacobian_linear() {
        // f(y) = 2*y, so df/dy = 2*I (diagonal)
        // With cj=0: J = df/dy = 2*I
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Mul(two, y));

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 2, 0);
        model.set_cj(0.0);

        let y_vals = [1.0, 2.0];
        let (rows, cols, vals) = model.assemble_jacobian(0.0, &y_vals, &[]);

        // Should have 2 non-zeros (diagonal)
        assert_eq!(rows.len(), 2);
        assert_eq!(cols.len(), 2);
        assert_eq!(vals.len(), 2);

        // Build dense matrix to verify
        let mut dense = [[0.0; 2]; 2];
        for i in 0..rows.len() {
            dense[rows[i]][cols[i]] = vals[i];
        }

        // J = 2*I
        assert!(
            (dense[0][0] - 2.0).abs() < 1e-12,
            "J[0,0] = {}",
            dense[0][0]
        );
        assert!(
            (dense[1][1] - 2.0).abs() < 1e-12,
            "J[1,1] = {}",
            dense[1][1]
        );
        assert!(dense[0][1].abs() < 1e-12, "J[0,1] = {}", dense[0][1]);
        assert!(dense[1][0].abs() < 1e-12, "J[1,0] = {}", dense[1][0]);
    }

    #[test]
    fn test_assemble_jacobian_with_mass() {
        // f(y) = 2*y, M = I
        // With cj=0.5: J = df/dy - cj*M = 2*I - 0.5*I = 1.5*I
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Mul(two, y));

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 2, 0);
        model.set_cj(0.5);

        let y_vals = [1.0, 2.0];
        let (rows, cols, vals) = model.assemble_jacobian(0.0, &y_vals, &[]);

        // Build dense matrix to verify
        let mut dense = [[0.0; 2]; 2];
        for i in 0..rows.len() {
            dense[rows[i]][cols[i]] = vals[i];
        }

        // J = 1.5*I
        assert!(
            (dense[0][0] - 1.5).abs() < 1e-12,
            "J[0,0] = {}",
            dense[0][0]
        );
        assert!(
            (dense[1][1] - 1.5).abs() < 1e-12,
            "J[1,1] = {}",
            dense[1][1]
        );
    }

    /// Square DAE whose last row is a dense algebraic equation:
    ///   `f_i = sin(y_i)` for `i in 0..n-1` (differential, mass diagonal 1) and
    ///   `f_{n-1} = sum_j y_j^2` (algebraic, mass diagonal 0), depending on every
    /// state. The dense row's `>= 16` columns also live on the diagonal rows
    /// (the aliasing trap), so the reduced coloring must not sum them.
    fn build_dense_row_dae(n: usize) -> (ModelEvaluator, Vec<f64>) {
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: n });
        // Differential block: sin over the first n-1 states (diagonal).
        let y_head = arena.alloc(Node::StateVector {
            start: 0,
            end: n - 1,
        });
        let diff = arena.alloc(Node::Sin(y_head));
        // Dense algebraic row: ones(1 x n) @ (y*y) = sum_j y_j^2.
        let gy = arena.alloc(Node::Mul(y_full, y_full));
        let ones = arena.alloc(Node::SparseMatrix(Box::new(CsrData {
            indptr: vec![0, n],
            indices: (0..n).collect(),
            data: vec![1.0; n],
            shape: Shape::matrix(1, n),
        })));
        let dense = arena.alloc(Node::MatMul(ones, gy));
        let rhs = arena.alloc(Node::Concat(vec![diff, dense]));
        // Mass: diag(1 for the n-1 differential rows, 0 for the algebraic row).
        let mass = CsrData {
            shape: Shape::matrix(n, n),
            indptr: (0..n).chain(std::iter::once(n - 1)).collect(),
            indices: (0..n - 1).collect(),
            data: vec![1.0; n - 1],
        };
        let model = ModelEvaluator::new(&arena, rhs, mass, n, 0);
        let y: Vec<f64> = (0..n)
            .map(|i| (i as f64).mul_add(0.1, 0.3).sin() + 0.7)
            .collect();
        (model, y)
    }

    /// Differential dense row whose `df/dy` excludes its own column
    /// (`f_0 = sum_{j>=1} y_j^2`), so with an identity mass the merged row 0
    /// carries column 0 as a mass-only slot the sub-Jacobian lacks, the
    /// "merged has more columns than the sub-row" remap case. Rows `1..n-1` are
    /// diagonal `sin(y_i)`.
    fn build_dense_row_dae_mass_only_column(n: usize) -> (ModelEvaluator, Vec<f64>) {
        let mut arena = Arena::new();
        let y_tail = arena.alloc(Node::StateVector { start: 1, end: n });
        let gy = arena.alloc(Node::Mul(y_tail, y_tail));
        let ones = arena.alloc(Node::SparseMatrix(Box::new(CsrData {
            indptr: vec![0, n - 1],
            indices: (0..n - 1).collect(),
            data: vec![1.0; n - 1],
            shape: Shape::matrix(1, n - 1),
        })));
        let dense = arena.alloc(Node::MatMul(ones, gy));
        let diag = arena.alloc(Node::Sin(y_tail));
        let rhs = arena.alloc(Node::Concat(vec![dense, diag]));
        let mass = identity_mass_matrix(n);
        let model = ModelEvaluator::new(&arena, rhs, mass, n, 0);
        let y: Vec<f64> = (0..n)
            .map(|i| (i as f64).mul_add(0.1, 0.4).cos() + 0.6)
            .collect();
        (model, y)
    }

    /// Dense df/dy via central finite differences of `eval_rhs`.
    fn finite_difference_jacobian(
        model: &mut ModelEvaluator,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        eps: f64,
    ) -> Vec<Vec<f64>> {
        let n = model.n_states();
        let mut fp = vec![0.0; n];
        let mut fm = vec![0.0; n];
        let mut dense = vec![vec![0.0; n]; n];
        for col in 0..n {
            let mut yp = y.to_vec();
            let mut ym = y.to_vec();
            yp[col] += eps;
            ym[col] -= eps;
            model.eval_rhs(t, &yp, inputs, &mut fp);
            model.eval_rhs(t, &ym, inputs, &mut fm);
            for row in 0..n {
                dense[row][col] = (fp[row] - fm[row]) / (2.0 * eps);
            }
        }
        dense
    }

    fn assert_matrices_match(assembled: &[Vec<f64>], expected: &[Vec<f64>], tol: f64, ctx: &str) {
        let n = expected.len();
        for row in 0..n {
            for col in 0..n {
                let err = (assembled[row][col] - expected[row][col]).abs();
                let scale = assembled[row][col]
                    .abs()
                    .max(expected[row][col].abs())
                    .max(1e-12);
                assert!(
                    err / scale < tol || err < tol,
                    "{ctx} J({row},{col}): assembled={}, expected={}",
                    assembled[row][col],
                    expected[row][col]
                );
            }
        }
    }

    fn coo_to_dense(rows: &[usize], cols: &[usize], vals: &[f64], n: usize) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; n]; n];
        for i in 0..rows.len() {
            dense[rows[i]][cols[i]] = vals[i];
        }
        dense
    }

    fn csc_to_dense(model: &ModelEvaluator, csc_vals: &[f64], n: usize) -> Vec<Vec<f64>> {
        let csc = model.csc_sparsity();
        let mut dense = vec![vec![0.0; n]; n];
        for (col, span) in csc.colptr.windows(2).enumerate() {
            for k in span[0]..span[1] {
                dense[csc.rowind[k]][col] = csc_vals[k];
            }
        }
        dense
    }

    #[test]
    fn test_all_assembly_paths_agree_with_dense_row_split() {
        let n = 20;
        let (mut model, y) = build_dense_row_dae(n);
        assert_eq!(
            model.compiled.jac.n_dense_rows(),
            1,
            "dense-row split must be active"
        );
        assert_eq!(model.compiled.jac.dense_rows()[0].rows[0], n - 1);
        let telemetry = model.jacobian_stats();
        assert_eq!(telemetry.n_dense_rows, 1);
        assert_eq!(telemetry.dense_row_entries, n);
        assert!(telemetry.dense_row_tape_instructions > 0);

        let dense_fd = finite_difference_jacobian(&mut model, 0.0, &y, &[], 1e-6);

        // Loop B (COO), cj = 0 -> J = df/dy.
        model.set_cj(0.0);
        let (rows, cols, vals) = model.assemble_jacobian(0.0, &y, &[]);
        assert_matrices_match(
            &coo_to_dense(&rows, &cols, &vals, n),
            &dense_fd,
            1e-5,
            "COO",
        );

        // Loop D (CSC, no mass) -> pure df/dy.
        let mut csc_no_mass = vec![0.0; model.nnz()];
        model.assemble_jacobian_csc_no_mass(0.0, &y, &[], &mut csc_no_mass);
        assert_matrices_match(
            &csc_to_dense(&model, &csc_no_mass, n),
            &dense_fd,
            1e-5,
            "CSC-no-mass",
        );

        // Loop C (CSC, with mass postpass), cj != 0 -> J = df/dy - cj*M. Mass is
        // diag(1 on the n-1 differential rows, 0 on the algebraic dense row).
        let cj = 0.7;
        model.set_cj(cj);
        let mut csc_mass = vec![0.0; model.nnz()];
        model.assemble_jacobian_csc_into_coloring(0.0, &y, &[], &mut csc_mass);
        let mut expected = dense_fd;
        for (i, row) in expected.iter_mut().enumerate().take(n - 1) {
            row[i] -= cj;
        }
        assert_matrices_match(
            &csc_to_dense(&model, &csc_mass, n),
            &expected,
            1e-5,
            "CSC-mass",
        );
    }

    #[test]
    fn test_dense_row_reverse_scales_linearly_not_quadratically() {
        // One reverse pass per dense row at any mesh size, with the value tape
        // growing ~linearly in state count rather than quadratically.
        let sizes = [20usize, 40, 80, 160];
        let mut tape_lens = Vec::new();
        for &n in &sizes {
            let (model, _y) = build_dense_row_dae(n);
            assert_eq!(
                model.jacobian_stats().n_dense_rows,
                1,
                "n={n}: reverse pass count must stay 1"
            );
            tape_lens.push(model.compiled.jac.max_adjoint_tape_len());
        }
        // Doubling n at most doubles a linear tape (b > 0 keeps every ratio < 2);
        // a quadratic tape would ~quadruple. `< 3.0` cleanly discriminates.
        for pair in tape_lens.windows(2) {
            let ratio = pair[1] as f64 / pair[0] as f64;
            assert!(
                ratio < 3.0,
                "adjoint tape grew {ratio:.1}x on doubling n (expected ~linear)"
            );
        }
    }

    #[test]
    fn test_dense_row_merged_remap_tolerates_mass_only_column() {
        let n = 20;
        let (mut model, y) = build_dense_row_dae_mass_only_column(n);
        assert_eq!(model.compiled.jac.n_dense_rows(), 1, "split must be active");
        assert_eq!(model.compiled.jac.dense_rows()[0].rows[0], 0);

        let dense_fd = finite_difference_jacobian(&mut model, 0.0, &y, &[], 1e-6);

        // Loop C with identity mass: J = df/dy - cj*I. Column 0 of row 0 is a
        // mass-only merged slot (df/dy has no entry) -> must become -cj.
        let cj = 0.5;
        model.set_cj(cj);
        let mut csc_mass = vec![0.0; model.nnz()];
        model.assemble_jacobian_csc_into_coloring(0.0, &y, &[], &mut csc_mass);
        let mut expected = dense_fd.clone();
        for (i, row) in expected.iter_mut().enumerate() {
            row[i] -= cj;
        }
        assert_matrices_match(
            &csc_to_dense(&model, &csc_mass, n),
            &expected,
            1e-5,
            "CSC-mass",
        );

        // Loop D (no mass): the mass-only slot (0,0) stays 0 (pure df/dy).
        let mut csc_no_mass = vec![0.0; model.nnz()];
        model.assemble_jacobian_csc_no_mass(0.0, &y, &[], &mut csc_no_mass);
        assert_matrices_match(
            &csc_to_dense(&model, &csc_no_mass, n),
            &dense_fd,
            1e-5,
            "CSC-no-mass",
        );

        // Loop B (COO) at cj != 0: the dense row's triplets must fold -cj*M,
        // emitting one triplet per merged nnz incl. the mass-only slot (0,0) = -cj.
        let (rows, cols, vals) = model.assemble_jacobian(0.0, &y, &[]);
        assert_eq!(rows.len(), model.nnz(), "one COO triplet per merged nnz");
        let coo = coo_to_dense(&rows, &cols, &vals, n);
        assert!(
            (coo[0][0] - (-cj)).abs() < 1e-12,
            "mass-only COO slot (0,0) must be -cj, got {}",
            coo[0][0]
        );
        assert_matrices_match(&coo, &expected, 1e-5, "COO-mass");
    }

    /// Differential dense row whose `df/dy` INCLUDES its own diagonal
    /// (`f_0 = sum_j y_j^2` over all states, identity mass), so merged slot
    /// (0,0) carries both a df/dy value and a mass entry. Rows `1..n-1` are
    /// diagonal `sin(y_i)`.
    fn build_dense_row_dae_shared_slot(n: usize) -> (ModelEvaluator, Vec<f64>) {
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: n });
        let gy = arena.alloc(Node::Mul(y_full, y_full));
        let ones = arena.alloc(Node::SparseMatrix(Box::new(CsrData {
            indptr: vec![0, n],
            indices: (0..n).collect(),
            data: vec![1.0; n],
            shape: Shape::matrix(1, n),
        })));
        let dense = arena.alloc(Node::MatMul(ones, gy));
        let y_tail = arena.alloc(Node::StateVector { start: 1, end: n });
        let diag = arena.alloc(Node::Sin(y_tail));
        let rhs = arena.alloc(Node::Concat(vec![dense, diag]));
        let mass = identity_mass_matrix(n);
        let model = ModelEvaluator::new(&arena, rhs, mass, n, 0);
        let y: Vec<f64> = (0..n)
            .map(|i| (i as f64).mul_add(0.2, 0.5).sin() + 0.8)
            .collect();
        (model, y)
    }

    #[test]
    fn test_dense_row_shared_dfdy_mass_slot_survives_postpass() {
        let n = 20;
        let (mut model, y) = build_dense_row_dae_shared_slot(n);
        assert_eq!(model.compiled.jac.n_dense_rows(), 1, "split must be active");
        assert_eq!(model.compiled.jac.dense_rows()[0].rows[0], 0);

        let dense_fd = finite_difference_jacobian(&mut model, 0.0, &y, &[], 1e-6);

        // Loop C at cj != 0: slot (0,0) carries BOTH the dense fill (2*y0) and
        // the mass postpass (-cj); a postpass-before-fill reorder clobbers -cj.
        let cj = 0.5;
        model.set_cj(cj);
        let mut csc_mass = vec![0.0; model.nnz()];
        model.assemble_jacobian_csc_into_coloring(0.0, &y, &[], &mut csc_mass);
        let assembled = csc_to_dense(&model, &csc_mass, n);
        let analytic = 2.0f64.mul_add(y[0], -cj);
        assert!(
            (assembled[0][0] - analytic).abs() < 1e-12,
            "shared slot (0,0) must be 2*y0 - cj = {analytic}, got {}",
            assembled[0][0]
        );
        let mut expected = dense_fd;
        for (i, row) in expected.iter_mut().enumerate() {
            row[i] -= cj;
        }
        assert_matrices_match(&assembled, &expected, 1e-5, "CSC-mass-shared");
    }

    /// Loop-B sparse scan: a differential row whose rhs does not depend on its
    /// own state has a mass-only merged slot `(r,r)`. Nothing in the df/dy
    /// sweep may reach that slot, which must carry a pure `-cj*M[r,r]` term.
    #[test]
    fn test_loop_b_sparse_scan_mass_only_slot_not_aliased() {
        // f = [sin(y1), sin(y1)]: col 0 is structurally absent from df/dy, and
        // identity mass adds slot (0,0).
        let mut arena = Arena::new();
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let f0 = arena.alloc(Node::Sin(y1));
        let f1 = arena.alloc(Node::Sin(y1));
        let rhs = arena.alloc(Node::Concat(vec![f0, f1]));
        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 2, 0);

        assert!(
            model.compiled.jac.dense_rows().is_empty(),
            "fixture must exercise the sparse scan, not the dense-row path"
        );
        // Precondition: col 0 has no df/dy entry, so no sweep ever writes it.
        assert_eq!(
            model.compiled.jac.coloring().colors[0],
            crate::coloring::UNSEEDED,
            "fixture requires col 0 to be absent from df/dy"
        );

        let y = [0.3, 0.5];
        let dcos = 0.5_f64.cos();

        // cj = 0 -> J = df/dy. The mass-only slot (0,0) has no df/dy entry, so it
        // must be exactly 0 (RED: old scan emitted df/dy(0,1) = cos(y1)).
        model.set_cj(0.0);
        let (rows, cols, vals) = model.assemble_jacobian(0.0, &y, &[]);
        assert_eq!(rows.len(), model.nnz(), "one triplet per merged nnz");
        let coo = coo_to_dense(&rows, &cols, &vals, 2);
        assert!(
            coo[0][0].abs() < 1e-12,
            "mass-only slot (0,0) must be 0 at cj=0, got {}",
            coo[0][0]
        );
        assert!(
            (coo[0][1] - dcos).abs() < 1e-12,
            "df/dy slot (0,1) must be cos(y1), got {}",
            coo[0][1]
        );

        // cj != 0 -> slot (0,0) = -cj*M[0,0] = -cj (identity mass).
        let cj = 0.5;
        model.set_cj(cj);
        let (rows, cols, vals) = model.assemble_jacobian(0.0, &y, &[]);
        let coo = coo_to_dense(&rows, &cols, &vals, 2);
        assert!(
            (coo[0][0] - (-cj)).abs() < 1e-12,
            "mass-only slot (0,0) must be -cj, got {}",
            coo[0][0]
        );
        assert!(
            (coo[0][1] - dcos).abs() < 1e-12,
            "df/dy slot (0,1) must stay cos(y1) at cj!=0, got {}",
            coo[0][1]
        );
    }

    #[test]
    fn test_assemble_jacobian_nonlinear() {
        // f(y) = y^2, so df/dy = diag([2*y0, 2*y1])
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Pow(y, two));

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 2, 0);
        model.set_cj(0.0);

        // At y = [3, 4], df/dy = diag([6, 8])
        let y_vals = [3.0, 4.0];
        let (rows, cols, vals) = model.assemble_jacobian(0.0, &y_vals, &[]);

        // Build dense matrix
        let mut dense = [[0.0; 2]; 2];
        for i in 0..rows.len() {
            dense[rows[i]][cols[i]] = vals[i];
        }

        assert!(
            (dense[0][0] - 6.0).abs() < 1e-12,
            "J[0,0] = {}",
            dense[0][0]
        );
        assert!(
            (dense[1][1] - 8.0).abs() < 1e-12,
            "J[1,1] = {}",
            dense[1][1]
        );
    }

    #[test]
    fn test_assemble_jacobian_coupled() {
        // f(y) = [y0 * y1, y0 + y1]
        // df/dy = [[y1, y0], [1, 1]]
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = arena.alloc(Node::Index {
            child: y_full,
            start: 0,
            end: 1,
        });
        let y1 = arena.alloc(Node::Index {
            child: y_full,
            start: 1,
            end: 2,
        });
        let prod = arena.alloc(Node::Mul(y0, y1));
        let sum = arena.alloc(Node::Add(y0, y1));
        let rhs = arena.alloc(Node::Concat(vec![prod, sum]));

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 2, 0);
        model.set_cj(0.0);

        // At y = [2, 3]
        // df/dy = [[3, 2], [1, 1]]
        let y_vals = [2.0, 3.0];
        let (rows, cols, vals) = model.assemble_jacobian(0.0, &y_vals, &[]);

        // Should have 4 non-zeros (dense 2x2)
        assert_eq!(rows.len(), 4, "Expected 4 non-zeros");

        // Build dense matrix
        let mut dense = [[0.0; 2]; 2];
        for i in 0..rows.len() {
            dense[rows[i]][cols[i]] = vals[i];
        }

        assert!(
            (dense[0][0] - 3.0).abs() < 1e-12,
            "J[0,0] = {}",
            dense[0][0]
        );
        assert!(
            (dense[0][1] - 2.0).abs() < 1e-12,
            "J[0,1] = {}",
            dense[0][1]
        );
        assert!(
            (dense[1][0] - 1.0).abs() < 1e-12,
            "J[1,0] = {}",
            dense[1][0]
        );
        assert!(
            (dense[1][1] - 1.0).abs() < 1e-12,
            "J[1,1] = {}",
            dense[1][1]
        );
    }

    #[test]
    fn test_assemble_jacobian_with_params() {
        // f(y) = k * y where k is a parameter
        // df/dy = k * I
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let k = arena.alloc(Node::InputParameter {
            name: "k".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let rhs = arena.alloc(Node::Mul(k, y));

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 2, 1);
        model.set_cj(1.0); // J = k*I - 1*I = (k-1)*I

        let y_vals = [1.0, 2.0];
        let inputs = [3.0]; // k = 3
        let (rows, cols, vals) = model.assemble_jacobian(0.0, &y_vals, &inputs);

        // Build dense matrix
        let mut dense = [[0.0; 2]; 2];
        for i in 0..rows.len() {
            dense[rows[i]][cols[i]] = vals[i];
        }

        // J = (3 - 1) * I = 2 * I
        assert!(
            (dense[0][0] - 2.0).abs() < 1e-12,
            "J[0,0] = {}",
            dense[0][0]
        );
        assert!(
            (dense[1][1] - 2.0).abs() < 1e-12,
            "J[1,1] = {}",
            dense[1][1]
        );
    }

    #[test]
    fn test_assemble_jacobian_sparse_pattern() {
        // f(y) = [y0, y1, y2] (identity)
        // df/dy = I (diagonal sparsity)
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let y2 = arena.alloc(Node::StateVector { start: 2, end: 3 });
        let rhs = arena.alloc(Node::Concat(vec![y0, y1, y2]));

        let mass = identity_mass_matrix(3);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 3, 0);
        model.set_cj(0.0);

        // Identity is fully constant, so the whole diagonal comes from the table.
        assert_eq!(model.coloring().n_colors, 0);

        let y_vals = [1.0, 2.0, 3.0];
        let (rows, cols, vals) = model.assemble_jacobian(0.0, &y_vals, &[]);

        // Should have exactly 3 non-zeros (diagonal)
        assert_eq!(rows.len(), 3, "Expected 3 non-zeros for diagonal");

        // All diagonal entries should be 1.0
        for i in 0..3 {
            assert_eq!(rows[i], cols[i], "Should be diagonal entry");
            assert!(
                (vals[i] - 1.0).abs() < 1e-12,
                "Diagonal entry {} = {}",
                i,
                vals[i]
            );
        }
    }

    #[test]
    fn test_assemble_jacobian_tridiagonal() {
        // Build tridiagonal-like structure
        // f(y) = [y0+y1, y0+y1+y2, y1+y2]
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let y2 = arena.alloc(Node::StateVector { start: 2, end: 3 });

        let f0 = arena.alloc(Node::Add(y0, y1));
        let f1_partial = arena.alloc(Node::Add(y0, y1));
        let f1 = arena.alloc(Node::Add(f1_partial, y2));
        let f2 = arena.alloc(Node::Add(y1, y2));
        let rhs = arena.alloc(Node::Concat(vec![f0, f1, f2]));

        let mass = identity_mass_matrix(3);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 3, 0);
        model.set_cj(0.0);

        // Should use <= 3 colors for tridiagonal
        assert!(
            model.coloring().n_colors <= 3,
            "Expected <= 3 colors, got {}",
            model.coloring().n_colors
        );

        let y_vals = [1.0, 2.0, 3.0];
        let (rows, cols, vals) = model.assemble_jacobian(0.0, &y_vals, &[]);

        // Build dense matrix
        let mut dense = [[0.0; 3]; 3];
        for i in 0..rows.len() {
            dense[rows[i]][cols[i]] = vals[i];
        }

        assert!((dense[0][0] - 1.0).abs() < 1e-12);
        assert!((dense[0][1] - 1.0).abs() < 1e-12);
        assert!(dense[0][2].abs() < 1e-12);
        assert!((dense[1][0] - 1.0).abs() < 1e-12);
        assert!((dense[1][1] - 1.0).abs() < 1e-12);
        assert!((dense[1][2] - 1.0).abs() < 1e-12);
        assert!(dense[2][0].abs() < 1e-12);
        assert!((dense[2][1] - 1.0).abs() < 1e-12);
        assert!((dense[2][2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_assemble_jacobian_consistency_with_jac_mul() {
        // J @ e_i == jac_mul(e_i) for each unit vector.
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let y0 = arena.alloc(Node::Index {
            child: y_full,
            start: 0,
            end: 1,
        });
        let y1 = arena.alloc(Node::Index {
            child: y_full,
            start: 1,
            end: 2,
        });
        let y2 = arena.alloc(Node::Index {
            child: y_full,
            start: 2,
            end: 3,
        });
        // f(y) = [y0*y1, y1*y2, y0+y2]
        let f0 = arena.alloc(Node::Mul(y0, y1));
        let f1 = arena.alloc(Node::Mul(y1, y2));
        let f2 = arena.alloc(Node::Add(y0, y2));
        let rhs = arena.alloc(Node::Concat(vec![f0, f1, f2]));

        let mass = identity_mass_matrix(3);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 3, 0);
        model.set_cj(0.5);

        let y_vals = [2.0, 3.0, 4.0];
        let (rows, cols, vals) = model.assemble_jacobian(0.0, &y_vals, &[]);

        // Build dense Jacobian
        let mut jac = [[0.0; 3]; 3];
        for i in 0..rows.len() {
            jac[rows[i]][cols[i]] = vals[i];
        }

        // Verify J @ e_i == jac_mul(e_i)
        for col in 0..3 {
            let mut e_i = [0.0, 0.0, 0.0];
            e_i[col] = 1.0;

            let mut jac_mul_result = [0.0, 0.0, 0.0];
            model.jac_mul(0.0, &y_vals, &[], &e_i, &mut jac_mul_result);

            for row in 0..3 {
                let expected = jac[row][col];
                let actual = jac_mul_result[row];
                assert!(
                    (expected - actual).abs() < 1e-12,
                    "Mismatch at ({row}, {col}): J={expected}, jac_mul={actual}"
                );
            }
        }
    }

    #[test]
    fn test_csc_pattern_from_csr() {
        // Create a simple CSR pattern for a 3x3 matrix with entries at (0,0), (1,1), (2,1), (2,2)
        let csr = SparsityPattern {
            nrows: 3,
            ncols: 3,
            indptr: vec![0, 1, 2, 4], // row 0: 1 entry, row 1: 1 entry, row 2: 2 entries
            indices: vec![0, 1, 1, 2], // col indices
        };

        let csc = CscPattern::from_csr(&csr);

        // Verify CSC structure
        assert_eq!(csc.nrows, 3);
        assert_eq!(csc.ncols, 3);
        assert_eq!(csc.nnz(), 4);

        // Column 0 has 1 entry (row 0), column 1 has 2 entries (rows 1, 2), column 2 has 1 entry (row 2)
        assert_eq!(csc.colptr, vec![0, 1, 3, 4]);
        // Row indices sorted by column
        assert_eq!(csc.rowind, vec![0, 1, 2, 2]);
    }

    #[test]
    fn test_csc_pattern_diagonal() {
        // Diagonal matrix: entries at (0,0), (1,1), (2,2)
        let csr = SparsityPattern {
            nrows: 3,
            ncols: 3,
            indptr: vec![0, 1, 2, 3],
            indices: vec![0, 1, 2],
        };

        let csc = CscPattern::from_csr(&csr);

        assert_eq!(csc.colptr, vec![0, 1, 2, 3]);
        assert_eq!(csc.rowind, vec![0, 1, 2]);
    }

    #[test]
    fn test_assemble_jacobian_csc_into_diagonal() {
        // f(y) = 2*y, so df/dy = 2*I (diagonal)
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Mul(two, y));

        let mass = identity_mass_matrix(3);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 3, 0);

        let y_vals = [1.0, 2.0, 3.0];
        let nnz = model.nnz();

        // Allocate CSC data buffer
        let mut jac_data = vec![0.0; nnz];

        // Assemble Jacobian with cj=0 (no mass term)
        model.set_cj(0.0);
        model.assemble_jacobian_csc_into(0.0, &y_vals, &[], &mut jac_data);

        // df/dy = 2*I, so all diagonal entries should be 2.0
        for &val in &jac_data {
            assert!((val - 2.0).abs() < 1e-14, "Expected 2.0, got {val}");
        }
    }

    #[test]
    fn test_assemble_jacobian_csc_into_with_cj() {
        // f(y) = 2*y, df/dy = 2*I, M = I
        // J = df/dy - cj*M = 2*I - 0.5*I = 1.5*I
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Mul(two, y));

        let mass = identity_mass_matrix(3);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 3, 0);

        let y_vals = [1.0, 2.0, 3.0];
        let nnz = model.nnz();

        let mut jac_data = vec![0.0; nnz];

        model.set_cj(0.5);
        model.assemble_jacobian_csc_into(0.0, &y_vals, &[], &mut jac_data);

        // J = 2*I - 0.5*I = 1.5*I
        for &val in &jac_data {
            assert!((val - 1.5).abs() < 1e-14, "Expected 1.5, got {val}");
        }
    }

    #[test]
    fn test_assemble_jacobian_csc_into_matches_coo() {
        // Compare CSC-into with the COO-returning version
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let y2 = arena.alloc(Node::StateVector { start: 2, end: 3 });

        // f(y) = [y0*y1, y1*y2, y0+y2] - coupled system
        let f0 = arena.alloc(Node::Mul(y0, y1));
        let f1 = arena.alloc(Node::Mul(y1, y2));
        let f2 = arena.alloc(Node::Add(y0, y2));
        let rhs = arena.alloc(Node::Concat(vec![f0, f1, f2]));

        let mass = identity_mass_matrix(3);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 3, 0);
        model.set_cj(0.5);

        let y_vals = [2.0, 3.0, 4.0];

        // Get COO result
        let (coo_rows, coo_cols, coo_vals) = model.assemble_jacobian(0.0, &y_vals, &[]);

        // Get CSC result
        let nnz = model.nnz();
        let mut csc_data = vec![0.0; nnz];
        model.assemble_jacobian_csc_into(0.0, &y_vals, &[], &mut csc_data);

        // Build dense matrices from both and compare
        let mut coo_dense = [[0.0; 3]; 3];
        for i in 0..coo_rows.len() {
            coo_dense[coo_rows[i]][coo_cols[i]] = coo_vals[i];
        }

        let csc = model.csc_sparsity();
        let mut csc_dense = [[0.0; 3]; 3];
        for (csc_idx, &val) in csc_data.iter().enumerate().take(nnz) {
            let (row, col) = csc.csc_to_csr_map[csc_idx];
            csc_dense[row][col] = val;
        }

        for row in 0..3 {
            for col in 0..3 {
                assert!(
                    (coo_dense[row][col] - csc_dense[row][col]).abs() < 1e-12,
                    "Mismatch at ({row}, {col}): COO={}, CSC={}",
                    coo_dense[row][col],
                    csc_dense[row][col]
                );
            }
        }
    }

    #[test]
    fn test_assemble_jacobian_csc_into_zero_allocation() {
        // Verify that repeated calls don't allocate (by checking buffer reuse)
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Mul(two, y));

        let mass = identity_mass_matrix(3);
        let mut model = ModelEvaluator::new(&arena, rhs, mass, 3, 0);

        let nnz = model.nnz();
        let mut jac_data = vec![0.0; nnz];

        // Call multiple times - should not allocate
        for i in 0..10_i32 {
            let y_vals = [1.0 + f64::from(i), 2.0, 3.0];
            model.assemble_jacobian_csc_into(0.0, &y_vals, &[], &mut jac_data);
            // All values should be 2.0 regardless of y (linear function)
            for &val in &jac_data {
                assert!((val - 2.0).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_jacobian_stats_report_the_compiled_strategy() {
        let mut arena = Arena::new();
        let x0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let x1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let f0 = arena.alloc(Node::Mul(x0, x1));
        let f1 = arena.alloc(Node::Add(x0, x1));
        let rhs = arena.alloc(Node::Concat(vec![f0, f1]));

        let mass = identity_mass_matrix(2);
        let model = ModelEvaluator::new(&arena, rhs, mass, 2, 0);

        let stats = model.jacobian_stats();
        assert_eq!(stats.strategy, JacobianStrategy::Coloring);
        assert_eq!(stats.n_colors, model.coloring().n_colors);
        assert_eq!(stats.nnz, model.nnz());
        assert_eq!(stats.n_dense_rows, 0);
        assert_eq!(stats.dense_row_entries, 0);
        assert_eq!(stats.dense_row_tape_instructions, 0);
    }

    #[test]
    fn test_n_events_zero_initially() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let model = ModelEvaluator::new(&arena, y, identity_mass_matrix(2), 2, 0);
        assert_eq!(model.n_events(), 0);
        assert_eq!(model.total_event_len(), 0);
    }

    #[test]
    fn test_add_event_compiles_and_evaluates() {
        // f(y) = y is the rhs; event = y[0] - 0.5 (triggers when y[0] crosses 0.5)
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = arena.alloc(Node::Index {
            child: y_full,
            start: 0,
            end: 1,
        });
        let threshold = arena.alloc(Node::Scalar(0.5));
        let event_expr = arena.alloc(Node::Sub(y0, threshold));

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, y_full, mass, 2, 0);
        model.add_event(&arena, event_expr);

        assert_eq!(model.n_events(), 1);
        assert_eq!(model.observables(ObservableKind::Events).len_at(0), 1);
        assert_eq!(model.total_event_len(), 1);

        // event(y=[0.7, 1.0]) = 0.7 - 0.5 = 0.2
        let mut out = [0.0; 1];
        let written = model.eval_event(0.0, &[0.7, 1.0], &[], 0, &mut out);
        assert_eq!(written, 1);
        assert!((out[0] - 0.2).abs() < 1e-12);

        // event(y=[0.5, 1.0]) = 0.5 - 0.5 = 0.0 (at threshold)
        let written = model.eval_event(0.0, &[0.5, 1.0], &[], 0, &mut out);
        assert_eq!(written, 1);
        assert!(out[0].abs() < 1e-12);

        // event(y=[0.3, 1.0]) = 0.3 - 0.5 = -0.2 (crossed threshold)
        let written = model.eval_event(0.0, &[0.3, 1.0], &[], 0, &mut out);
        assert_eq!(written, 1);
        assert!((out[0] + 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_eval_events_multiple_events() {
        // Two events: e0 = y[0] - 0.5, e1 = y[1] - 1.0
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = arena.alloc(Node::Index {
            child: y_full,
            start: 0,
            end: 1,
        });
        let y1 = arena.alloc(Node::Index {
            child: y_full,
            start: 1,
            end: 2,
        });
        let thresh0 = arena.alloc(Node::Scalar(0.5));
        let thresh1 = arena.alloc(Node::Scalar(1.0));
        let event0 = arena.alloc(Node::Sub(y0, thresh0));
        let event1 = arena.alloc(Node::Sub(y1, thresh1));

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, y_full, mass, 2, 0);
        model.add_event(&arena, event0);
        model.add_event(&arena, event1);

        assert_eq!(model.n_events(), 2);
        assert_eq!(model.total_event_len(), 2);

        // y = [0.7, 1.5] => e0 = 0.2, e1 = 0.5
        let mut out = [0.0; 2];
        model.eval_observables(ObservableKind::Events, 0.0, &[0.7, 1.5], &[], &mut out);
        assert!((out[0] - 0.2).abs() < 1e-12);
        assert!((out[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_events_compose_with_outputs() {
        // Verify events work alongside outputs on the same model
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = arena.alloc(Node::Index {
            child: y_full,
            start: 0,
            end: 1,
        });
        let two = arena.alloc(Node::Scalar(2.0));
        let output_expr = arena.alloc(Node::Mul(two, y0));
        let thresh = arena.alloc(Node::Scalar(0.5));
        let event_expr = arena.alloc(Node::Sub(y0, thresh));

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, y_full, mass, 2, 0);
        model.add_output(&arena, output_expr);
        model.add_event(&arena, event_expr);

        assert_eq!(model.n_outputs(), 1);
        assert_eq!(model.n_events(), 1);

        // y = [3.0, 4.0] => output = 6.0, event = 2.5
        let mut out_val = [0.0; 1];
        let mut event_val = [0.0; 1];
        model.eval_output(0.0, &[3.0, 4.0], &[], 0, &mut out_val);
        model.eval_event(0.0, &[3.0, 4.0], &[], 0, &mut event_val);

        assert!((out_val[0] - 6.0).abs() < 1e-12);
        assert!((event_val[0] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_fuse_events_matches_the_per_event_tapes() {
        // Two events over a shared subexpression `shared = y[0]*y[1] + y[0]`:
        //   e0 = shared - 0.5,  e1 = shared*2 - 1
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = arena.alloc(Node::Index {
            child: y_full,
            start: 0,
            end: 1,
        });
        let y1 = arena.alloc(Node::Index {
            child: y_full,
            start: 1,
            end: 2,
        });
        let prod = arena.alloc(Node::Mul(y0, y1));
        let shared = arena.alloc(Node::Add(prod, y0));
        let half = arena.alloc(Node::Scalar(0.5));
        let event0 = arena.alloc(Node::Sub(shared, half));
        let two = arena.alloc(Node::Scalar(2.0));
        let scaled = arena.alloc(Node::Mul(shared, two));
        let one = arena.alloc(Node::Scalar(1.0));
        let event1 = arena.alloc(Node::Sub(scaled, one));

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, y_full, mass, 2, 0);
        model.add_event(&arena, event0);
        model.add_event(&arena, event1);
        model.fuse_events(&mut arena, &[event0, event1]);

        // The fused tape is the one the hot loop now uses.
        assert!(
            model
                .observables(ObservableKind::Events)
                .fused_expr()
                .is_some()
        );
        assert_eq!(model.total_event_len(), 2);

        // The fused tape must match the per-event tapes BITWISE: same
        // nodes, same instruction semantics.
        for &y in &[[0.7, 1.5], [0.3, -2.0], [1.25, 0.0]] {
            let mut e0 = [0.0; 1];
            let mut e1 = [0.0; 1];
            model.eval_event(0.0, &y, &[], 0, &mut e0);
            model.eval_event(0.0, &y, &[], 1, &mut e1);

            let mut fused = [0.0; 2];
            model.eval_observables(ObservableKind::Events, 0.0, &y, &[], &mut fused);
            // Compare raw bits: the fused tape must reproduce the per-event
            // values exactly (identical nodes and instruction semantics).
            assert_eq!(fused[0].to_bits(), e0[0].to_bits());
            assert_eq!(fused[1].to_bits(), e1[0].to_bits());

            // `eval_events` (diffsol path) shares the same fused tape.
            let mut fused_diffsol = [0.0; 2];
            model.eval_observables(ObservableKind::Events, 0.0, &y, &[], &mut fused_diffsol);
            assert_eq!(fused_diffsol.map(f64::to_bits), fused.map(f64::to_bits));
        }
    }

    #[test]
    fn test_fuse_events_noop_for_single_event() {
        // One event: fusion is a no-op and the per-event path is used unchanged.
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = arena.alloc(Node::Index {
            child: y_full,
            start: 0,
            end: 1,
        });
        let half = arena.alloc(Node::Scalar(0.5));
        let event0 = arena.alloc(Node::Sub(y0, half));

        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, y_full, mass, 2, 0);
        model.add_event(&arena, event0);
        model.fuse_events(&mut arena, &[event0]);

        assert!(
            model
                .observables(ObservableKind::Events)
                .fused_expr()
                .is_none()
        );
        let mut out = [0.0; 1];
        model.eval_observables(ObservableKind::Events, 0.0, &[0.7, 1.0], &[], &mut out);
        assert!((out[0] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn mass_kind_identity() {
        let mass = CsrData {
            indptr: vec![0, 1, 2],
            indices: vec![0, 1],
            data: vec![1.0, 1.0],
            shape: Shape::matrix(2, 2),
        };
        assert!(matches!(classify_mass_matrix(&mass), MassKind::Identity));
    }

    #[test]
    fn mass_kind_diagonal_selector() {
        let mass = CsrData {
            indptr: vec![0, 1, 1],
            indices: vec![0],
            data: vec![1.0],
            shape: Shape::matrix(2, 2),
        };
        match classify_mass_matrix(&mass) {
            MassKind::DiagonalSelector(mask) => assert_eq!(mask, vec![true, false]),
            other => panic!("Expected DiagonalSelector, got {other:?}"),
        }
    }

    #[test]
    fn mass_kind_general_offdiag() {
        let mass = CsrData {
            indptr: vec![0, 2, 3],
            indices: vec![0, 1, 1],
            data: vec![1.0, 0.5, 1.0],
            shape: Shape::matrix(2, 2),
        };
        assert!(matches!(classify_mass_matrix(&mass), MassKind::General));
    }

    /// Build a `CompiledModel` with one output and one sensitivity parameter.
    ///
    /// Model: f(y) = k * y (RHS), H(y, p) = k * y[0]^2 (output).
    /// `n_states` = 2, `n_params` = 1, sens on param 0.
    fn build_test_model_with_output_and_sens() -> CompiledModel {
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = arena.alloc(Node::Index {
            child: y_full,
            start: 0,
            end: 1,
        });
        let k = arena.alloc(Node::InputParameter {
            name: "k".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let two = arena.alloc(Node::Scalar(2.0));
        // RHS: k * y (2-vector)
        let rhs = arena.alloc(Node::Mul(k, y_full));
        // Output: k * y[0]^2 (scalar)
        let y0_sq = arena.alloc(Node::Pow(y0, two));
        let out_node = arena.alloc(Node::Mul(k, y0_sq));

        let mass = identity_mass_matrix(2);
        let mut data = CompiledModel::new_with_sens(&arena, rhs, mass, 2, 1, &[0]);
        data.add_output(&arena, out_node);
        data
    }

    /// State vector for `build_test_model_with_output_and_sens`.
    fn test_state(model: &CompiledModel) -> Vec<f64> {
        vec![3.0; model.n_states]
    }

    /// Input vector for `build_test_model_with_output_and_sens`.
    fn test_inputs(_model: &CompiledModel) -> Vec<f64> {
        vec![2.0] // k = 2
    }

    #[test]
    fn an_output_sens_action_matches_finite_difference() {
        // Build a compiled model with >=1 output and >=1 sensitivity parameter.
        let model = build_test_model_with_output_and_sens();
        let mut ws = model.create_workspace();
        let t = 0.3;
        let y = test_state(&model);
        let inputs = test_inputs(&model);

        // Analytic dH/dp . e_k for the first sensitivity parameter.
        let mut v = vec![0.0; model.n_sens_params()];
        v[0] = 1.0;
        let mut got = vec![0.0; model.total_output_len()];
        model.observable_sens_action(
            &mut ws,
            ObservableKind::Outputs,
            t,
            &y,
            &inputs,
            model.sens_param_indices(),
            &v,
            &mut got,
        );

        // Finite-difference reference: perturb that parameter, re-eval outputs.
        let h = 1e-6;
        let pidx = model.sens_param_indices()[0];
        let mut ip = inputs.clone();
        let mut o_plus = vec![0.0; model.total_output_len()];
        let mut o_minus = vec![0.0; model.total_output_len()];
        ip[pidx] += h;
        model.eval_observables(&mut ws, ObservableKind::Outputs, t, &y, &ip, &mut o_plus);
        ip[pidx] -= 2.0 * h;
        model.eval_observables(&mut ws, ObservableKind::Outputs, t, &y, &ip, &mut o_minus);
        for j in 0..got.len() {
            let fd = (o_plus[j] - o_minus[j]) / (2.0 * h);
            assert!(
                (got[j] - fd).abs() < 1e-5,
                "out[{j}]: got {} fd {}",
                got[j],
                fd
            );
        }
    }

    #[test]
    fn an_output_jac_action_matches_finite_difference() {
        let model = build_test_model_with_output_and_sens();
        let mut ws = model.create_workspace();
        let (t, y, inputs) = (0.3, test_state(&model), test_inputs(&model));
        let mut v = vec![0.0; model.n_states];
        v[0] = 1.0; // dH/dy . e_0
        let mut got = vec![0.0; model.total_output_len()];
        model.observable_jac_action(
            &mut ws,
            ObservableKind::Outputs,
            t,
            &y,
            &inputs,
            &v,
            &mut got,
        );
        let h = 1e-6;
        let (mut yp, mut ym) = (y.clone(), y.clone());
        yp[0] += h;
        ym[0] -= h;
        let mut op = vec![0.0; model.total_output_len()];
        let mut om = vec![0.0; model.total_output_len()];
        model.eval_observables(&mut ws, ObservableKind::Outputs, t, &yp, &inputs, &mut op);
        model.eval_observables(&mut ws, ObservableKind::Outputs, t, &ym, &inputs, &mut om);
        for j in 0..got.len() {
            let fd = (op[j] - om[j]) / (2.0 * h);
            assert!(
                (got[j] - fd).abs() < 1e-5,
                "jac[{j}]: got {} fd {}",
                got[j],
                fd
            );
        }
    }

    #[test]
    fn output_sens_project_equals_action_sum() {
        // projection[k] must equal dH/dp . e_k + dH/dy . y_sens_k
        let model = build_test_model_with_output_and_sens();
        let mut ws = model.create_workspace();
        let (t, y, inputs) = (0.3, test_state(&model), test_inputs(&model));
        let n_s = model.n_sens_params();
        let n_states = model.n_states;
        let n_out = model.total_output_len();

        let mut y_sens = vec![0.0; n_s * n_states];
        for (i, v) in y_sens.iter_mut().enumerate() {
            *v = 0.1 * (i as f64 + 1.0);
        }
        let mut got = vec![0.0; n_s * n_out];
        model.output_sens_project(&mut ws, t, &y, &inputs, &y_sens, &mut got);

        for k in 0..n_s {
            let mut e_k = vec![0.0; n_s];
            e_k[k] = 1.0;
            let mut sens_term = vec![0.0; n_out];
            model.observable_sens_action(
                &mut ws,
                ObservableKind::Outputs,
                t,
                &y,
                &inputs,
                model.sens_param_indices(),
                &e_k,
                &mut sens_term,
            );
            let mut jac_term = vec![0.0; n_out];
            model.observable_jac_action(
                &mut ws,
                ObservableKind::Outputs,
                t,
                &y,
                &inputs,
                &y_sens[k * n_states..(k + 1) * n_states],
                &mut jac_term,
            );
            for o in 0..n_out {
                let expected = sens_term[o] + jac_term[o];
                assert!(
                    (got[k * n_out + o] - expected).abs() < 1e-12,
                    "k={k} o={o}: got {} want {}",
                    got[k * n_out + o],
                    expected
                );
            }
        }
    }

    /// Build a `CompiledModel` with one event and one sensitivity parameter.
    ///
    /// Model: f(y) = k * y (RHS), g(y, p) = k * y[0]^2 - 1.0 (event).
    /// `n_states` = 2, `n_params` = 1, sens on param 0.
    fn build_test_model_with_event_and_sens() -> CompiledModel {
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = arena.alloc(Node::Index {
            child: y_full,
            start: 0,
            end: 1,
        });
        let k = arena.alloc(Node::InputParameter {
            name: "k".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let two = arena.alloc(Node::Scalar(2.0));
        let one = arena.alloc(Node::Scalar(1.0));
        // RHS: k * y (2-vector)
        let rhs = arena.alloc(Node::Mul(k, y_full));
        // Event: k * y[0]^2 - 1.0 (scalar)
        let y0_sq = arena.alloc(Node::Pow(y0, two));
        let ky0_sq = arena.alloc(Node::Mul(k, y0_sq));
        let event_node = arena.alloc(Node::Sub(ky0_sq, one));

        let mass = identity_mass_matrix(2);
        let mut data = CompiledModel::new_with_sens(&arena, rhs, mass, 2, 1, &[0]);
        data.add_event(&arena, event_node);
        data
    }

    #[test]
    fn event_sens_and_jac_actions_match_finite_difference() {
        let model = build_test_model_with_event_and_sens();
        let mut ws = model.create_workspace();
        let (t, y, inputs) = (0.3, test_state(&model), test_inputs(&model));

        // dg/dp . e_0
        let mut vp = vec![0.0; model.n_sens_params()];
        vp[0] = 1.0;
        let mut got_p = vec![0.0; model.total_event_len()];
        model.observable_sens_action(
            &mut ws,
            ObservableKind::Events,
            t,
            &y,
            &inputs,
            model.sens_param_indices(),
            &vp,
            &mut got_p,
        );
        let h = 1e-6;
        let pidx = model.sens_param_indices()[0];
        let (mut ip_p, mut ip_m) = (inputs.clone(), inputs.clone());
        ip_p[pidx] += h;
        ip_m[pidx] -= h;
        let mut gp = vec![0.0; model.total_event_len()];
        let mut gm = vec![0.0; model.total_event_len()];
        model.eval_observables(&mut ws, ObservableKind::Events, t, &y, &ip_p, &mut gp);
        model.eval_observables(&mut ws, ObservableKind::Events, t, &y, &ip_m, &mut gm);
        for j in 0..got_p.len() {
            let fd = (gp[j] - gm[j]) / (2.0 * h);
            assert!((got_p[j] - fd).abs() < 1e-5);
        }

        // dg/dy . e_0
        let mut vy = vec![0.0; model.n_states];
        vy[0] = 1.0;
        let mut got_y = vec![0.0; model.total_event_len()];
        model.observable_jac_action(
            &mut ws,
            ObservableKind::Events,
            t,
            &y,
            &inputs,
            &vy,
            &mut got_y,
        );
        let (mut yp, mut ym) = (y.clone(), y.clone());
        yp[0] += h;
        ym[0] -= h;
        let mut gyp = vec![0.0; model.total_event_len()];
        let mut gym = vec![0.0; model.total_event_len()];
        model.eval_observables(&mut ws, ObservableKind::Events, t, &yp, &inputs, &mut gyp);
        model.eval_observables(&mut ws, ObservableKind::Events, t, &ym, &inputs, &mut gym);
        for j in 0..got_y.len() {
            let fd = (gyp[j] - gym[j]) / (2.0 * h);
            assert!((got_y[j] - fd).abs() < 1e-5);
        }
    }

    #[test]
    #[should_panic(expected = "sens_param_indices contains a repeated index")]
    fn duplicate_sens_param_indices_are_rejected() {
        // A repeated index would turn the subset scatter into an accumulate and
        // break the set_params/get_params round trip.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let a = arena.alloc(Node::InputParameter {
            name: "a".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let rhs = arena.alloc(Node::Mul(a, y));
        let mass = CsrData {
            indptr: vec![0, 1],
            indices: vec![0],
            data: vec![1.0],
            shape: Shape::matrix(1, 1),
        };
        ModelEvaluator::new_with_options(
            &arena,
            rhs,
            mass,
            1,
            1,
            CompiledModelOptions::new().with_sensitivities(&[0, 0]),
        );
    }
}
