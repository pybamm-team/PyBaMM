//! `CompiledModel`: a DAE `M y' = f(t, y)` with a constant, often singular, mass matrix `M`.
//!
//! Construction, accessors and the `df/dy` entry points; the mass algebra, sensitivities,
//! algebraic block and observables each live in a submodule.

mod algebraic;
mod mass;
mod observables;
#[cfg(test)]
mod poison_probe;
mod sens;
#[cfg(test)]
pub(crate) mod test_fixtures;

use std::cell::RefCell;
use std::sync::Arc;

use crate::arena::{Arena, DagRef, NodeId, ReadExtents, resolve_state_count};
use crate::error::CoreError;
use crate::eval::{CompiledExpr, SplitCompiledExpr};
use crate::ir::TypedIr;
use crate::jacobian::{CscPattern, JacobianData, JacobianLayout, JacobianScratch, JvpAt};
use crate::node::CsrData;
use crate::observable::{ObservableScratch, ObservableSet};
use crate::params::{PackedWidth, ParamIdx, ParamRegistry, resolve_input_width};
use crate::sparsity::SparsityPattern;

pub use algebraic::AlgebraicBlock;
use algebraic::AlgebraicScratch;
pub use mass::StateKinds;
use mass::{MassKind, classify_mass_matrix};

/// Lightweight Jacobian-assembly stats exposed to benchmarks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JacobianStats {
    /// Colors in the adopted coloring, so tangent sweeps per assembly.
    pub n_colors: usize,
    pub nnz: usize,
    /// Rows filled by reverse mode instead of the coloring.
    pub n_dense_rows: usize,
    /// Rows `detect_dense_rows` nominated; above `n_dense_rows` means the split was declined.
    pub n_dense_row_candidates: usize,
    /// Entries proven constant and written from a table rather than swept.
    pub n_constant_entries: usize,
    pub n_swept_columns: usize,
    pub dense_row_entries: usize,
    pub dense_row_tape_instructions: usize,
    /// Instructions in the assembly tape's shared primal section.
    pub split_eval_primal_instructions: usize,
    /// Common tape plus dispatch: what runs regardless of branch.
    pub split_eval_total_instructions: usize,
    /// Raw tape length, branch blocks included.
    pub split_eval_raw_instructions: usize,
    /// One per short-circuited conditional, in each of the primal and tangent halves.
    pub split_eval_dispatch_count: usize,
    /// Per-branch block lengths, in tape order.
    pub branch_block_lens: Vec<usize>,
    /// Colours the first, widest walk of an assembly carries; 1 when each colour walks alone.
    pub jac_lane_width: usize,
}

/// The algebraic sub-block to compile alongside the right-hand side.
#[derive(Debug, Clone, Copy)]
pub struct AlgebraicBlockSpec<'a> {
    /// The algebraic residual `g(t, y)`.
    pub residual: NodeId,
    /// Global state indices `g` solves for: strictly ascending, each one algebraic.
    pub variable_indices: &'a [usize],
}

/// Optional artifacts to compile with a model; the default asks for none.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompiledModelOptions<'a> {
    /// Registration indices to compile `df/dp` for, in sensitivity column order.
    pub sensitivities: &'a [ParamIdx],
    /// An algebraic sub-block to compile for consistent initialisation.
    pub algebraic: Option<AlgebraicBlockSpec<'a>>,
}

/// Panic message for a [`Workspace`] handed to a model other than the one that created it.
const FOREIGN_WORKSPACE: &str = "the workspace was created by another model";

/// Per-solve mutable scratch for evaluating a [`CompiledModel`].
#[derive(Debug)]
pub struct Workspace {
    primal_scratch: Vec<f64>,
    jac_scratch: JacobianScratch,
    sens_scratch: Option<Vec<f64>>,
    output_scratch: ObservableScratch,
    event_scratch: ObservableScratch,
    algebraic: Option<AlgebraicScratch>,
    sens_dp_buffer: Vec<f64>,
    #[cfg(test)]
    pub(crate) sens_primal_passes: usize,
}

#[cfg(test)]
impl Workspace {
    /// Overwrite every buffer with `value`, as an unwound call could leave them.
    pub(crate) fn poison(&mut self, value: f64) {
        self.primal_scratch.fill(value);
        self.jac_scratch.poison(value);
        if let Some(scratch) = &mut self.sens_scratch {
            scratch.fill(value);
        }
        self.output_scratch.poison(value);
        self.event_scratch.poison(value);
        if let Some(scratch) = &mut self.algebraic {
            scratch.poison(value);
        }
        self.sens_dp_buffer.fill(value);
    }
}

/// A shared [`CompiledModel`] and one caller's [`Workspace`]: the `user_data` behind every
/// [`crate::ffi`] entry point, and what the diffsol solver's operators evaluate through.
#[derive(Debug)]
pub struct ModelEvaluator {
    pub(crate) compiled: Arc<CompiledModel>,
    /// A `RefCell` so operators sharing the evaluator borrow it per call; the FFI, which
    /// holds the evaluator exclusively, reaches it with `get_mut`.
    pub(crate) workspace: RefCell<Workspace>,
}

impl ModelEvaluator {
    /// Bind a fresh [`Workspace`] to an already-compiled model, without recompiling.
    pub fn from_compiled(compiled: Arc<CompiledModel>) -> Self {
        let workspace = RefCell::new(compiled.create_workspace());
        Self {
            compiled,
            workspace,
        }
    }
}

/// A compiled DAE system: primal expression, symbolic Jacobian, mass matrix.
///
/// Immutable and shareable across concurrent solves; scratch lives in a [`Workspace`].
pub struct CompiledModel {
    primal_expr: Arc<CompiledExpr>,
    jac: Arc<JacobianData>,
    mass_matrix: CsrData,
    mass_kind: MassKind,
    sparsity: SparsityPattern,
    csc_sparsity: CscPattern,
    n_states: usize,
    /// Declared length of the `inputs` slice, which may exceed what the expression reads.
    input_width: PackedWidth,
    /// Registration-index space: where each parameter sits in `inputs`.
    params: ParamRegistry,
    /// Registration extent the tangent tapes index `dp` by, widened by later observables.
    param_extent: usize,
    /// `jac`'s entries on the merged CSC, which also carries the mass pattern's slots.
    jac_layout: JacobianLayout,
    mass_to_csc_map: Vec<usize>,
    state_kinds: StateKinds,
    sens_expr: Option<SplitCompiledExpr>,
    /// Registration index of each sensitivity column, in column order.
    sens_param_indices: Vec<ParamIdx>,
    /// Output variables H(t, y; p) and their tangent tapes.
    outputs: ObservableSet,
    /// Event functions g(t, y; p), fused into one tape when there are two or more.
    events: ObservableSet,
    algebraic: Option<AlgebraicBlock>,
}

impl CompiledModel {
    /// Compile `M y' = f(t, y; p)` with `rhs` as `f`, plus what `options` asks for.
    ///
    /// Refuses declared sizes below what the graph reads, a mass matrix that is not a canonical
    /// `n_states x n_states` CSR, and sensitivity or algebraic specs the model cannot honour.
    pub fn new(
        arena: &Arena,
        rhs: NodeId,
        mass_matrix: CsrData,
        n_states: usize,
        input_width: PackedWidth,
        options: CompiledModelOptions<'_>,
    ) -> Result<Self, CoreError> {
        let params = verify_widths(arena, rhs, n_states, input_width)?;
        let mass_shape = mass_matrix.shape();
        if mass_shape.rows != n_states || mass_shape.cols != n_states {
            return Err(CoreError::MassMatrixShape {
                rows: mass_shape.rows,
                cols: mass_shape.cols,
                n_states,
            });
        }
        // Up front, so a mass row that repeats or reorders columns fails before the Jacobian build.
        let mass_sparsity = SparsityPattern::try_from(&mass_matrix)?;
        // Compiled raw: a ULP-shifting `simplify` fold stalls IDA's Newton.
        let primal_ir = TypedIr::from_dag(DagRef::new(arena, rhs));
        let n_outputs = primal_ir.output_len();
        if n_outputs != n_states {
            return Err(CoreError::ResidualLength {
                got: n_outputs,
                n_states,
            });
        }

        let jac = Arc::new(JacobianData::new_wrt_states(
            arena, rhs, n_outputs, n_states,
        ));

        let mut sparsity = jac.sparsity().clone();
        sparsity.merge_with(&mass_sparsity);

        let csc_sparsity = CscPattern::from_csr(&sparsity);

        let mass_to_csc_map = csc_sparsity.slots_of(&mass_sparsity);
        let jac_layout = jac.layout_in(&csc_sparsity);

        let primal_expr = Arc::new(CompiledExpr::from_ir(primal_ir));

        let state_kinds = StateKinds::from_mass_matrix(&mass_matrix);

        let mass_kind = classify_mass_matrix(&mass_matrix);

        let mut model = Self {
            primal_expr,
            jac,
            mass_matrix,
            mass_kind,
            sparsity,
            csc_sparsity,
            n_states,
            input_width,
            param_extent: params.index_extent(),
            params,
            jac_layout,
            mass_to_csc_map,
            state_kinds,
            sens_expr: None,
            sens_param_indices: Vec::new(),
            outputs: ObservableSet::default(),
            events: ObservableSet::default(),
            algebraic: None,
        };
        model.compile_sensitivities(arena, rhs, options.sensitivities)?;
        if let Some(spec) = options.algebraic {
            model.compile_algebraic_block(arena, spec)?;
        }
        Ok(model)
    }

    /// Whether `df/dp` was compiled, and with it every observable's tangent tapes.
    #[inline]
    const fn has_sensitivities(&self) -> bool {
        self.sens_expr.is_some()
    }

    /// Allocate a fresh [`Workspace`] sized for this model's expressions.
    pub fn create_workspace(&self) -> Workspace {
        Workspace {
            primal_scratch: vec![0.0; self.primal_expr.scratch_len()],
            jac_scratch: JacobianScratch::new(&self.jac),
            sens_scratch: self.sens_expr.as_ref().map(|e| vec![0.0; e.scratch_len()]),
            output_scratch: self.outputs.create_scratch(),
            event_scratch: self.events.create_scratch(),
            algebraic: self.algebraic.as_ref().map(AlgebraicScratch::new),
            // `dp` is indexed by `ParamIdx`: registration extent, not packed width.
            sens_dp_buffer: vec![0.0; self.param_extent.max(1)],
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

    #[inline]
    pub const fn n_states(&self) -> usize {
        self.n_states
    }

    #[inline]
    pub const fn input_width(&self) -> PackedWidth {
        self.input_width
    }

    /// The input parameters the graph registers: where each sits in the packed `inputs`.
    #[inline]
    pub const fn params(&self) -> &ParamRegistry {
        &self.params
    }

    /// Length of `f(t, y)`, which construction pins to `n_states`.
    #[inline]
    pub const fn output_len(&self) -> usize {
        self.n_states
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
        self.primal_expr
            .eval_into(&mut ws.primal_scratch, t, y, &[], inputs, output);
    }

    /// Compute pure Jacobian-vector product: df/dy @ v (no mass term).
    pub(crate) fn jac_action(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        v: &[f64],
        jv: &mut [f64],
    ) {
        self.jac
            .action_into(&mut ws.jac_scratch, t, y, &[], inputs, v, jv);
    }

    /// Run the primal pass at `(t, y)` and return a [`JvpAt`] for computing
    /// [`Self::jac_action`] there in any number of directions.
    pub(crate) fn jac_action_at<'a>(
        &'a self,
        ws: &'a mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
    ) -> JvpAt<'a> {
        self.jac.action_at(&mut ws.jac_scratch, t, y, &[], inputs)
    }

    /// CSR pattern of the assembled Jacobian `df/dy - cj*M` (union of both patterns).
    #[inline]
    pub const fn sparsity(&self) -> &SparsityPattern {
        &self.sparsity
    }

    /// Colours the `df/dy` sweep uses; with a dense-row split it leaves those rows out.
    #[inline]
    pub fn n_colors(&self) -> usize {
        self.jac.n_colors()
    }

    #[inline]
    pub const fn mass_matrix(&self) -> &CsrData {
        &self.mass_matrix
    }

    #[inline]
    pub const fn state_kinds(&self) -> &StateKinds {
        &self.state_kinds
    }

    /// CSC pattern of the assembled Jacobian `df/dy - cj*M`, the layout KLU and diffsol expect.
    #[inline]
    pub const fn csc_sparsity(&self) -> &CscPattern {
        &self.csc_sparsity
    }

    /// Non-zeros in the assembled Jacobian, the buffer size `assemble_jacobian_csc_*` needs.
    #[inline]
    pub const fn nnz(&self) -> usize {
        self.sparsity.nnz()
    }

    /// Compile-time Jacobian metrics for benchmark attribution.
    pub fn jacobian_stats(&self) -> JacobianStats {
        let tape = self.jac.assembly_tape();
        let ir = tape.ir();
        JacobianStats {
            n_colors: self.jac.n_colors(),
            nnz: self.nnz(),
            n_dense_rows: self.jac.n_dense_rows(),
            n_dense_row_candidates: self.jac.n_candidate_rows(),
            n_constant_entries: self.jac.constant_csr_entries().len(),
            n_swept_columns: self.jac.coloring().n_seeded_columns(),
            dense_row_entries: self.jac.dense_row_entries(),
            dense_row_tape_instructions: self.jac.dense_row_tape_instructions(),
            split_eval_primal_instructions: tape.split_info().primal_end,
            split_eval_total_instructions: ir.common_instruction_count(),
            split_eval_raw_instructions: ir.instructions().len(),
            split_eval_dispatch_count: ir.dispatch_count(),
            branch_block_lens: ir.branch_block_lens(),
            jac_lane_width: self.jac.lane_width(),
        }
    }

    /// Assemble `J = df/dy - cj * M` into a pre-allocated CSC data buffer.
    pub fn assemble_jacobian_csc_into(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        cj: f64,
        jac_data: &mut [f64],
    ) {
        self.assemble_jacobian_csc_no_mass(ws, t, y, inputs, jac_data);
        self.apply_mass_postpass(cj, jac_data);
    }

    /// Assemble `df/dy` alone in CSC, without the mass post-pass (diffsol applies mass itself).
    pub fn assemble_jacobian_csc_no_mass(
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
}

/// Refuse an expression that reads more of `y` than `n_states` declares.
///
/// Scans only what `node` reaches, so discarded wider expressions in the arena are ignored.
fn verify_state_extent(arena: &Arena, node: NodeId, n_states: usize) -> Result<(), CoreError> {
    resolve_state_count(Some(n_states), ReadExtents::scan(arena, node).n_states())?;
    Ok(())
}

/// Refuse declared widths below what the graph reads, returning the arena's parameters.
fn verify_widths(
    arena: &Arena,
    rhs: NodeId,
    n_states: usize,
    input_width: PackedWidth,
) -> Result<ParamRegistry, CoreError> {
    verify_state_extent(arena, rhs, n_states)?;
    let registry = ParamRegistry::from_arena(arena)?;
    resolve_input_width(Some(input_width), registry.packed_width())?;
    Ok(registry)
}

impl std::fmt::Debug for CompiledModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledModel")
            .field("n_states", &self.n_states)
            .field("input_width", &self.input_width)
            .field("output_len", &self.output_len())
            .field("sparsity_nnz", &self.sparsity.nnz())
            .field("n_colors", &self.jac.n_colors())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::test_fixtures::{
        assemble_dense, assemble_dense_no_mass, assert_matrices_match, compile, dense_row_model,
        doubling_model, element, finite_difference_jacobian, identity_mass_matrix, input,
        less_cj_on_the_diagonal, try_compile,
    };
    use super::*;
    use crate::jacobian::tests::banded_nonlinear;
    use crate::node::{Node, Shape};

    #[test]
    fn data_workspace_roundtrip_eval_rhs() {
        let mut arena = Arena::new();
        let sv0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sv1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let neg_one = arena.alloc(Node::Scalar(-1.0));
        let neg_two = arena.alloc(Node::Scalar(-2.0));
        let r0 = arena.alloc(Node::Mul(neg_one, sv0));
        let r1 = arena.alloc(Node::Mul(neg_two, sv1));
        let rhs = arena.alloc(Node::Concat(vec![r0, r1]));
        let compiled = compile(&arena, rhs, identity_mass_matrix(2), 2, 0);

        let mut ws = compiled.create_workspace();
        let mut out = vec![0.0; 2];
        compiled.eval_rhs(&mut ws, 0.0, &[3.0, 5.0], &[], &mut out);
        assert!((out[0] - (-3.0)).abs() < 1e-12);
        assert!((out[1] - (-10.0)).abs() < 1e-12);

        assert_eq!(ws.primal_scratch.len(), compiled.primal_expr.scratch_len());
    }

    #[test]
    fn a_model_reports_its_sizes() {
        let model = doubling_model(2);

        assert_eq!(model.n_states(), 2);
        assert_eq!(model.input_width(), PackedWidth::new(0));
        assert_eq!(model.output_len(), 2);
        let debug = format!("{model:?}");
        assert!(debug.contains("CompiledModel") && debug.contains("n_states: 2"));
    }

    #[test]
    fn test_compiled_model_sparsity() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let rhs = arena.alloc(Node::Sin(y));
        let model = compile(&arena, rhs, identity_mass_matrix(2), 2, 0);

        assert_eq!(model.n_colors(), 1, "a diagonal needs one colour");
    }

    /// A small identity-mass model and its analytic `df/dy` at the point evaluated.
    struct AnalyticCase {
        name: &'static str,
        model: CompiledModel,
        y: Vec<f64>,
        inputs: Vec<f64>,
        /// `df/dy` at `(y, inputs)`, row-major.
        jacobian: Vec<Vec<f64>>,
    }

    /// `build` receives the arena and the whole state vector, and returns the rhs.
    fn analytic_case(
        name: &'static str,
        (y, inputs): (Vec<f64>, Vec<f64>),
        jacobian: Vec<Vec<f64>>,
        build: impl FnOnce(&mut Arena, NodeId) -> NodeId,
    ) -> AnalyticCase {
        let n = y.len();
        let mut arena = Arena::new();
        let state = arena.alloc(Node::StateVector { start: 0, end: n });
        let rhs = build(&mut arena, state);
        let model = compile(&arena, rhs, identity_mass_matrix(n), n, inputs.len());
        AnalyticCase {
            name,
            model,
            y,
            inputs,
            jacobian,
        }
    }

    fn analytic_cases() -> Vec<AnalyticCase> {
        vec![
            analytic_case(
                "2y",
                (vec![1.0, 2.0], vec![]),
                vec![vec![2.0, 0.0], vec![0.0, 2.0]],
                |arena, y| {
                    let two = arena.alloc(Node::Scalar(2.0));
                    arena.alloc(Node::Mul(two, y))
                },
            ),
            analytic_case(
                "y^2",
                (vec![3.0, 4.0], vec![]),
                vec![vec![6.0, 0.0], vec![0.0, 8.0]],
                |arena, y| {
                    let two = arena.alloc(Node::Scalar(2.0));
                    arena.alloc(Node::Pow(y, two))
                },
            ),
            analytic_case(
                "k*y",
                (vec![1.0, 2.0], vec![3.0]),
                vec![vec![3.0, 0.0], vec![0.0, 3.0]],
                |arena, y| {
                    let k = input(arena, "k", 0, 0, 1);
                    arena.alloc(Node::Mul(k, y))
                },
            ),
            analytic_case(
                "identity",
                (vec![1.0, 2.0, 3.0], vec![]),
                vec![
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 1.0],
                ],
                |_, y| y,
            ),
            analytic_case(
                "[y0*y1, y0+y1]",
                (vec![2.0, 3.0], vec![]),
                vec![vec![3.0, 2.0], vec![1.0, 1.0]],
                |arena, y| {
                    let (y0, y1) = (element(arena, y, 0), element(arena, y, 1));
                    let prod = arena.alloc(Node::Mul(y0, y1));
                    let sum = arena.alloc(Node::Add(y0, y1));
                    arena.alloc(Node::Concat(vec![prod, sum]))
                },
            ),
            analytic_case(
                "[y0*y1, y0+5]",
                (vec![2.0, 3.0], vec![]),
                vec![vec![3.0, 2.0], vec![1.0, 0.0]],
                |arena, y| {
                    let (y0, y1) = (element(arena, y, 0), element(arena, y, 1));
                    let five = arena.alloc(Node::Scalar(5.0));
                    let prod = arena.alloc(Node::Mul(y0, y1));
                    let sum = arena.alloc(Node::Add(y0, five));
                    arena.alloc(Node::Concat(vec![prod, sum]))
                },
            ),
            analytic_case(
                "tridiagonal sums",
                (vec![1.0, 2.0, 3.0], vec![]),
                vec![
                    vec![1.0, 1.0, 0.0],
                    vec![1.0, 1.0, 1.0],
                    vec![0.0, 1.0, 1.0],
                ],
                |arena, y| {
                    let [y0, y1, y2] = [0, 1, 2].map(|i| element(arena, y, i));
                    let f0 = arena.alloc(Node::Add(y0, y1));
                    let f1_partial = arena.alloc(Node::Add(y0, y1));
                    let f1 = arena.alloc(Node::Add(f1_partial, y2));
                    let f2 = arena.alloc(Node::Add(y1, y2));
                    arena.alloc(Node::Concat(vec![f0, f1, f2]))
                },
            ),
            analytic_case(
                "[y0*y1, y1*y2, y0+y2]",
                (vec![2.0, 3.0, 4.0], vec![]),
                vec![
                    vec![3.0, 2.0, 0.0],
                    vec![0.0, 4.0, 3.0],
                    vec![1.0, 0.0, 1.0],
                ],
                |arena, y| {
                    let [y0, y1, y2] = [0, 1, 2].map(|i| element(arena, y, i));
                    let f0 = arena.alloc(Node::Mul(y0, y1));
                    let f1 = arena.alloc(Node::Mul(y1, y2));
                    let f2 = arena.alloc(Node::Add(y0, y2));
                    arena.alloc(Node::Concat(vec![f0, f1, f2]))
                },
            ),
        ]
    }

    /// Pattern, matrix-free actions and CSC assembly at two `cj`, against analytic `df/dy`.
    #[test]
    fn small_models_match_their_analytic_jacobians() {
        for case in analytic_cases() {
            let AnalyticCase {
                name,
                model,
                y,
                inputs,
                jacobian,
            } = case;
            let n = y.len();
            let mut ws = model.create_workspace();

            // Identity mass: the merged pattern is df/dy's non-zeros plus the diagonal.
            let csc = model.csc_sparsity();
            for (col, span) in csc.colptr().windows(2).enumerate() {
                let rows = &csc.rowind()[span[0]..span[1]];
                for (row, analytic) in jacobian.iter().enumerate() {
                    let expected = analytic[col] != 0.0 || row == col;
                    assert_eq!(rows.contains(&row), expected, "{name}: slot ({row}, {col})");
                }
            }

            let mut column = vec![f64::NAN; n];
            for col in 0..n {
                let mut unit = vec![0.0; n];
                unit[col] = 1.0;
                model.jac_action(&mut ws, 0.0, &y, &inputs, &unit, &mut column);
                for (row, analytic) in jacobian.iter().enumerate() {
                    assert!(
                        (column[row] - analytic[col]).abs() < 1e-12,
                        "{name}: J v at ({row}, {col}) = {}",
                        column[row]
                    );
                }
            }

            for cj in [0.0, 0.5] {
                let assembled = assemble_dense(&model, cj, &y, &inputs);
                for (row, analytic) in jacobian.iter().enumerate() {
                    for (col, &value) in analytic.iter().enumerate() {
                        let expected = if row == col { value - cj } else { value };
                        assert!(
                            (assembled[row][col] - expected).abs() < 1e-12,
                            "{name}, cj = {cj}: J({row}, {col}) = {}",
                            assembled[row][col]
                        );
                    }
                }
            }
        }
    }

    /// Nonlinear ODE with a one-sided stencil of `reach + 1` colours, so the count can be even.
    fn build_one_sided_model(n: usize, reach: usize) -> CompiledModel {
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
        compile(&arena, rhs, identity_mass_matrix(n), n, 0)
    }

    /// Nonlinear banded ODE of a chosen half-bandwidth.
    fn build_banded_model(n: usize, half_width: usize) -> CompiledModel {
        let (arena, rhs) = banded_nonlinear(n, half_width);
        compile(&arena, rhs, identity_mass_matrix(n), n, 0)
    }

    #[test]
    fn a_narrow_tail_block_matches_the_unbatched_tape() {
        // Symmetric half-widths give odd colour counts, one-sided reaches even ones.
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
            let model = if symmetric {
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
            let mut ws = model.create_workspace();
            let y: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.037, 0.41)).collect();
            let mut assembled = vec![f64::NAN; model.nnz()];
            model.assemble_jacobian_csc_into(&mut ws, 0.0, &y, &[], 0.0, &mut assembled);

            let (colptr, rowind) = {
                let csc = model.csc_sparsity();
                (csc.colptr(), csc.rowind())
            };
            let mut column = vec![0.0; n];
            let mut seed = vec![0.0; n];
            for col in 0..n {
                seed.fill(0.0);
                seed[col] = 1.0;
                model.jac_action(&mut ws, 0.0, &y, &[], &seed, &mut column);
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
    fn a_linear_model_assembles_without_sweeping() {
        // Every entry folds at compile time, so the coloring seeds nothing.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let model = compile(&arena, y, identity_mass_matrix(2), 2, 0);

        let stats = model.jacobian_stats();
        assert_eq!(stats.n_colors, 0);
        assert_eq!(stats.n_swept_columns, 0);
        assert_eq!(stats.n_constant_entries, 2);

        let mut data = vec![f64::NAN; model.nnz()];
        let mut ws = model.create_workspace();
        model.assemble_jacobian_csc_into(&mut ws, 0.0, &[7.0, -3.0], &[], 0.25, &mut data);
        assert_eq!(data, vec![0.75, 0.75]);
    }

    /// The extent is whole-arena: `wide` is registered but never read.
    #[test]
    fn an_under_declared_input_width_is_rejected() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let k = input(&mut arena, "k", 0, 0, 1);
        let _wide = input(&mut arena, "wide", 1, 1, 3);
        let rhs = arena.alloc(Node::Mul(k, y));

        let err = try_compile(&arena, rhs, identity_mass_matrix(2), 2, 1)
            .expect_err("a width below the registered inputs must be refused");
        assert!(
            matches!(
                err,
                CoreError::InputWidthTooSmall {
                    declared: 1,
                    extent: 4
                }
            ),
            "unexpected error: {err}"
        );

        let model = compile(&arena, rhs, identity_mass_matrix(2), 2, 6);
        assert_eq!(model.input_width(), PackedWidth::new(6));
    }

    #[test]
    fn a_state_count_below_the_tape_extent_is_rejected() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 4 });
        let rhs = arena.alloc(Node::Neg(y));

        let err = try_compile(&arena, rhs, identity_mass_matrix(4), 2, 0)
            .expect_err("n_states below the state extent must be refused");
        assert!(
            matches!(
                err,
                CoreError::StateCountTooSmall {
                    declared: 2,
                    extent: 4
                }
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn a_residual_shorter_than_the_state_count_is_rejected() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let rhs = element(&mut arena, y, 0);

        let err = try_compile(&arena, rhs, identity_mass_matrix(2), 2, 0)
            .expect_err("a residual row count below n_states must be refused");
        assert!(
            matches!(
                err,
                CoreError::ResidualLength {
                    got: 1,
                    n_states: 2
                }
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn a_mass_matrix_of_another_size_is_rejected() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let rhs = arena.alloc(Node::Neg(y));

        let err = try_compile(&arena, rhs, identity_mass_matrix(3), 2, 0)
            .expect_err("a mass matrix that is not n_states x n_states must be refused");
        assert!(
            matches!(
                err,
                CoreError::MassMatrixShape {
                    rows: 3,
                    cols: 3,
                    n_states: 2
                }
            ),
            "unexpected error: {err}"
        );
    }

    /// The merged CSC pattern needs each mass row's columns strictly ascending.
    #[test]
    fn a_mass_row_with_unsorted_or_repeated_columns_is_rejected() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let rhs = arena.alloc(Node::Neg(y));
        let build = |row0: Vec<usize>| {
            let nnz = row0.len();
            let mass = CsrData::try_new(
                vec![0, nnz, nnz + 1],
                row0.into_iter().chain([1]).collect(),
                vec![1.0; nnz + 1],
                Shape::matrix(2, 2),
            )
            .expect("structurally valid CSR");
            try_compile(&arena, rhs, mass, 2, 0).map(|_| ())
        };

        for row0 in [vec![1, 0], vec![0, 0]] {
            let err = build(row0.clone()).expect_err("a non-canonical mass row must be refused");
            assert!(
                matches!(&err, CoreError::Csr(message) if message.contains("row 0")),
                "{row0:?}: unexpected error: {err}"
            );
        }
        build(vec![0, 1]).expect("strictly ascending columns are accepted");
    }

    /// DAE of `n-1` differential `sin(y_i)` rows and a dense algebraic row `sum_j y_j^2`.
    ///
    /// Its columns also live on the diagonal rows, so the reduced coloring must not sum them.
    fn build_dense_row_dae(n: usize) -> (CompiledModel, Vec<f64>) {
        let mass = CsrData {
            shape: Shape::matrix(n, n),
            indptr: (0..n).chain(std::iter::once(n - 1)).collect(),
            indices: (0..n - 1).collect(),
            data: vec![1.0; n - 1],
        };
        let y = (0..n)
            .map(|i| (i as f64).mul_add(0.1, 0.3).sin() + 0.7)
            .collect();
        (dense_row_model(n, 0..n, false, mass), y)
    }

    #[test]
    fn test_all_assembly_paths_agree_with_dense_row_split() {
        let n = 20;
        let (model, y) = build_dense_row_dae(n);
        assert_eq!(
            model.jac.n_dense_rows(),
            1,
            "dense-row split must be active"
        );
        assert_eq!(model.jac.dense_row_indices().next(), Some(n - 1));
        assert!(
            model.n_colors() <= 4,
            "dense-row split must reduce the coloring, got {}",
            model.n_colors()
        );
        let telemetry = model.jacobian_stats();
        assert_eq!(telemetry.n_dense_rows, 1);
        assert_eq!(telemetry.dense_row_entries, n);
        assert!(telemetry.dense_row_tape_instructions > 0);

        let dense_fd = finite_difference_jacobian(&model, &y, 1e-6);
        let no_mass = assemble_dense_no_mass(&model, &y);
        assert_matrices_match(&no_mass, &dense_fd, 1e-5, "CSC-no-mass");

        let cj = 0.7;
        let expected = less_cj_on_the_diagonal(dense_fd, cj, n - 1);
        let with_mass = assemble_dense(&model, cj, &y, &[]);
        assert_matrices_match(&with_mass, &expected, 1e-5, "CSC-mass");
    }

    #[test]
    fn test_dense_row_reverse_scales_linearly_not_quadratically() {
        let sizes = [20usize, 40, 80, 160];
        let mut tape_lens = Vec::new();
        for &n in &sizes {
            let (model, _y) = build_dense_row_dae(n);
            assert_eq!(
                model.jacobian_stats().n_dense_rows,
                1,
                "n={n}: reverse pass count must stay 1"
            );
            tape_lens.push(model.jac.max_adjoint_tape_len());
        }
        // Doubling n at most doubles a linear tape and quadruples a quadratic one.
        for pair in tape_lens.windows(2) {
            let ratio = pair[1] as f64 / pair[0] as f64;
            assert!(
                ratio < 3.0,
                "adjoint tape grew {ratio:.1}x on doubling n (expected ~linear)"
            );
        }
    }
}
