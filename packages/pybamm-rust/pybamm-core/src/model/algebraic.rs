//! The algebraic sub-block `g(t, y) = 0` and `dg/dy_alg`, for consistent initialisation.

use std::sync::Arc;

use crate::arena::{Arena, DagRef};
use crate::error::CoreError;
use crate::eval::CompiledExpr;
use crate::ir::TypedIr;
use crate::jacobian::{JacobianData, JacobianScratch};
use crate::simplify::{dce, simplify_pipeline};

use super::{AlgebraicBlockSpec, CompiledModel, FOREIGN_WORKSPACE, Workspace, verify_state_extent};

/// The compiled algebraic residual `g(t, y)` and `dg/dy_alg`, for consistent initialisation.
#[derive(Debug)]
pub struct AlgebraicBlock {
    residual: Arc<CompiledExpr>,
    /// `dg/dy_alg` over the algebraic columns only.
    jac: Arc<JacobianData>,
}

impl AlgebraicBlock {
    /// Algebraic variables the block solves for, which is also its residual length.
    #[inline]
    pub fn n_variables(&self) -> usize {
        self.jac.n_cols()
    }

    /// The residual `g(t, y)`, shareable without recompiling.
    #[inline]
    pub const fn residual_expr(&self) -> &Arc<CompiledExpr> {
        &self.residual
    }

    /// The prepared `dg/dy_alg`, shareable without recompiling.
    #[inline]
    pub const fn jacobian(&self) -> &Arc<JacobianData> {
        &self.jac
    }

    /// COO `(rows, cols)` of `dg/dy_alg` in the order the block assembles it, columns as
    /// state indices.
    #[inline]
    pub fn jacobian_coo(&self) -> (&[usize], &[usize]) {
        self.jac.coo_global_indices()
    }

    /// Evaluate `g(t, y)` into `output`.
    pub(crate) fn eval_residual(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        output: &mut [f64],
    ) {
        self.residual
            .eval_into(&mut Self::scratch(ws).residual, t, y, &[], inputs, output);
    }

    /// Assemble `dg/dy_alg` into `jac_data`, in [`jacobian_coo`](Self::jacobian_coo) order.
    pub(crate) fn assemble_jacobian_into(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        jac_data: &mut [f64],
    ) {
        let jac = &self.jac;
        let scratch = &mut Self::scratch(ws).jac;
        jac.assemble_into(scratch, jac.layout(), t, y, &[], inputs, jac_data);
    }

    /// `ws`'s buffers for this block.
    const fn scratch(ws: &mut Workspace) -> &mut AlgebraicScratch {
        ws.algebraic.as_mut().expect(FOREIGN_WORKSPACE)
    }
}

/// A [`Workspace`]'s buffers for an [`AlgebraicBlock`].
#[derive(Debug)]
pub struct AlgebraicScratch {
    residual: Vec<f64>,
    jac: JacobianScratch,
}

impl AlgebraicScratch {
    /// Buffers sized for `block`.
    pub fn new(block: &AlgebraicBlock) -> Self {
        Self {
            residual: vec![0.0; block.residual.scratch_len()],
            jac: JacobianScratch::new(&block.jac),
        }
    }

    /// Overwrite every buffer with `value`, as an unwound call could leave them.
    #[cfg(test)]
    pub fn poison(&mut self, value: f64) {
        self.residual.fill(value);
        self.jac.poison(value);
    }
}

impl CompiledModel {
    /// Compile the algebraic residual and its Jacobian over the algebraic columns.
    ///
    /// The block must be square: the Newton drivers factorise `dg/dy_alg` as `n_alg x n_alg`.
    pub(super) fn compile_algebraic_block(
        &mut self,
        arena: &Arena,
        spec: AlgebraicBlockSpec<'_>,
    ) -> Result<(), CoreError> {
        let AlgebraicBlockSpec {
            residual: algebraic,
            variable_indices,
        } = spec;
        verify_state_extent(arena, algebraic, self.n_states)?;
        self.verify_algebraic_indices(variable_indices)?;
        let algebraic_dag = simplify_pipeline(dce(DagRef::new(arena, algebraic)));
        let algebraic_ir = TypedIr::from_dag(algebraic_dag.as_ref());
        let n_rows = algebraic_ir.output_len();
        if n_rows != variable_indices.len() {
            return Err(CoreError::AlgebraicBlockNotSquare {
                rows: n_rows,
                columns: variable_indices.len(),
            });
        }
        let residual = Arc::new(CompiledExpr::from_ir(algebraic_ir));

        let jac = Arc::new(JacobianData::new_wrt_state_subset(
            arena,
            algebraic,
            n_rows,
            self.n_states,
            variable_indices,
        ));
        self.algebraic = Some(AlgebraicBlock { residual, jac });
        Ok(())
    }

    /// Refuse algebraic indices that are out of range, unsorted, or name a differential state.
    fn verify_algebraic_indices(&self, indices: &[usize]) -> Result<(), CoreError> {
        for &index in indices {
            if index >= self.n_states {
                return Err(CoreError::AlgebraicVariableIndex {
                    index,
                    n_states: self.n_states,
                });
            }
            if self.state_kinds.is_differential(index) {
                return Err(CoreError::AlgebraicVariableDifferential { index });
            }
        }
        for pair in indices.windows(2) {
            let (previous, index) = (pair[0], pair[1]);
            if previous >= index {
                return Err(CoreError::AlgebraicVariableOrder { previous, index });
            }
        }
        Ok(())
    }

    /// The algebraic sub-block, or `None` when the model was built without one.
    #[inline]
    pub const fn algebraic(&self) -> Option<&AlgebraicBlock> {
        self.algebraic.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::{
        algebraic_options, all_algebraic_mass_matrix, compile, compile_with, doubling_model,
        element, identity_mass_matrix, input, split_algebraic_model,
    };
    use super::*;
    use crate::arena::NodeId;
    use crate::jacobian::tests::banded_nonlinear;
    use crate::model::CompiledModelOptions;
    use crate::node::{CsrData, Node, Shape};
    use crate::params::{PackedWidth, ParamIdx};

    /// Assemble `jac` directly, through its own canonical layout.
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

    /// A fully algebraic banded model, whose `dg/dy_alg` is wide enough to batch.
    fn build_batched_algebraic_model(n: usize, half_width: usize) -> CompiledModel {
        let (arena, root) = banded_nonlinear(n, half_width);
        let indices: Vec<usize> = (0..n).collect();
        compile_with(
            &arena,
            root,
            all_algebraic_mass_matrix(n),
            n,
            0,
            algebraic_options(root, &indices),
        )
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact equality is the point
    fn the_algebraic_block_inherits_lane_batching() {
        let (n, half_width) = (64usize, 3usize);
        let model = build_batched_algebraic_model(n, half_width);
        let block = model
            .algebraic()
            .expect("the fixture compiles an algebraic block");
        let jac = block.jacobian();
        assert!(
            JacobianScratch::new(jac).lane_width() > 1,
            "the algebraic fixture must batch"
        );

        let y: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.037, 0.41)).collect();
        let mut batched = vec![f64::NAN; jac.nnz()];
        block.assemble_jacobian_into(&mut model.create_workspace(), 0.5, &y, &[], &mut batched);

        let scalar = assemble_through(jac, JacobianScratch::scalar(jac), 0.5, &y);
        assert_eq!(batched, scalar);
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact equality is the point
    fn the_algebraic_block_assembles_in_its_published_order() {
        let n = 24usize;
        let model = build_batched_algebraic_model(n, 2);
        let block = model.algebraic().expect("algebraic block");
        let jac = block.jacobian();
        assert_eq!(block.jacobian_coo(), jac.coo_global_indices());

        let y: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.13, 0.7)).collect();
        let mut through_block = vec![f64::NAN; jac.nnz()];
        block.assemble_jacobian_into(
            &mut model.create_workspace(),
            0.25,
            &y,
            &[],
            &mut through_block,
        );

        let through_artifact = assemble_through(jac, JacobianScratch::new(jac), 0.25, &y);
        assert_eq!(through_block, through_artifact);
    }

    #[test]
    #[should_panic(expected = "created by another model")]
    fn a_workspace_from_another_model_is_refused() {
        let algebraic = build_batched_algebraic_model(4, 1);
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 4 });
        let plain = compile(&arena, y, identity_mass_matrix(4), 4, 0);

        let mut out = [0.0; 4];
        algebraic
            .algebraic()
            .expect("algebraic block")
            .eval_residual(&mut plain.create_workspace(), 0.0, &[0.5; 4], &[], &mut out);
    }

    #[test]
    fn an_algebraic_block_past_the_declared_state_count_is_rejected() {
        let mut arena = Arena::new();
        let narrow = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let rhs = arena.alloc(Node::Neg(narrow));
        let wide = arena.alloc(Node::StateVector { start: 0, end: 8 });

        let err = CompiledModel::new(
            &arena,
            rhs,
            identity_mass_matrix(2),
            2,
            PackedWidth::new(0),
            algebraic_options(wide, &[0, 1]),
        )
        .expect_err("an algebraic root past n_states must be refused");
        assert!(
            matches!(err, CoreError::StateCountTooSmall { declared: 2, .. }),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn an_algebraic_variable_index_outside_the_state_count_is_rejected() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let rhs = arena.alloc(Node::Neg(y));

        let build = |indices: &[usize]| {
            CompiledModel::new(
                &arena,
                rhs,
                all_algebraic_mass_matrix(2),
                2,
                PackedWidth::new(0),
                algebraic_options(rhs, indices),
            )
            .map(|_| ())
        };

        assert!(matches!(
            build(&[0, 2]).expect_err("index 2 is outside n_states=2"),
            CoreError::AlgebraicVariableIndex {
                index: 2,
                n_states: 2
            }
        ));
        assert!(matches!(
            build(&[1, 0]).expect_err("descending indices are refused"),
            CoreError::AlgebraicVariableOrder {
                previous: 1,
                index: 0
            }
        ));
        assert!(matches!(
            build(&[0, 0]).expect_err("repeated indices are refused"),
            CoreError::AlgebraicVariableOrder {
                previous: 0,
                index: 0
            }
        ));
        build(&[0, 1]).expect("a strictly ascending in-range subset is accepted");
    }

    #[test]
    fn a_non_square_algebraic_block_is_rejected() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let rhs = arena.alloc(Node::Neg(y));

        let err = CompiledModel::new(
            &arena,
            rhs,
            all_algebraic_mass_matrix(2),
            2,
            PackedWidth::new(0),
            algebraic_options(rhs, &[0]),
        )
        .expect_err("two residual rows over one variable is not square");
        assert!(
            matches!(
                err,
                CoreError::AlgebraicBlockNotSquare {
                    rows: 2,
                    columns: 1
                }
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn an_algebraic_variable_index_naming_a_differential_state_is_rejected() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let rhs = arena.alloc(Node::Neg(y));
        let algebraic_row = element(&mut arena, y, 1);
        // State 0 differential, state 1 algebraic.
        let mass = CsrData {
            indptr: vec![0, 1, 1],
            indices: vec![0],
            data: vec![1.0],
            shape: Shape::matrix(2, 2),
        };
        let build = |block: NodeId, indices: &[usize]| {
            CompiledModel::new(
                &arena,
                rhs,
                mass.clone(),
                2,
                PackedWidth::new(0),
                algebraic_options(block, indices),
            )
            .map(|_| ())
        };

        let err = build(rhs, &[0, 1]).expect_err("state 0 is differential");
        assert!(
            matches!(err, CoreError::AlgebraicVariableDifferential { index: 0 }),
            "unexpected error: {err}"
        );
        build(algebraic_row, &[1]).expect("the algebraic state alone is accepted");
    }

    #[test]
    fn test_options_compile_algebraic_and_sensitivities() {
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let y1 = element(&mut arena, y_full, 1);
        let y2 = element(&mut arena, y_full, 2);
        let k = input(&mut arena, "k", 0, 0, 1);
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

        let options = CompiledModelOptions {
            sensitivities: &[ParamIdx::new(0)],
            ..algebraic_options(alg, &[1, 2])
        };
        let model = compile_with(&arena, rhs, mass, 3, 1, options);
        let mut ws = model.create_workspace();

        assert_eq!(model.n_sens_params(), 1);
        let block = model
            .algebraic()
            .expect("the options compile an algebraic block");
        assert_eq!(block.n_variables(), 2);

        let y_vals = [3.0, 4.0, 5.0];
        let inputs = [2.0];

        let mut sens_out = [0.0; 3];
        model.eval_sens_all(&mut ws, 0.0, &y_vals, &inputs, &mut sens_out);
        for (actual, expected) in sens_out.iter().zip(y_vals.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }

        let mut algebraic_out = [0.0; 2];
        block.eval_residual(&mut ws, 0.0, &y_vals, &inputs, &mut algebraic_out);
        assert!((algebraic_out[0] - 9.0).abs() < 1e-12);
        assert!((algebraic_out[1] - 20.0).abs() < 1e-12);
    }

    #[test]
    fn test_algebraic_jac_action_uses_algebraic_index_mapping() {
        let model = split_algebraic_model();
        let jac = model.algebraic().expect("algebraic block").jacobian();
        let y_vals = [10.0, 2.0, 3.0];
        let v_alg = [5.0, 7.0];
        let mut jv = [0.0; 2];
        let mut scratch = JacobianScratch::new(jac);
        jac.action_into(&mut scratch, 0.0, &y_vals, &[], &[], &v_alg, &mut jv);

        assert!((jv[0] - 29.0).abs() < 1e-12, "expected 29, got {}", jv[0]);
        assert!((jv[1] - 47.0).abs() < 1e-12, "expected 47, got {}", jv[1]);
    }

    #[test]
    fn test_assemble_alg_jacobian_into_matches_expected_sparse_values() {
        let model = split_algebraic_model();
        let block = model.algebraic().expect("algebraic block");

        // Column-major, matching `JacobianData`'s CSC.
        let (rows, cols) = block.jacobian_coo();
        assert_eq!(rows, &[0, 1, 0, 1]);
        assert_eq!(cols, &[1, 1, 2, 2]);

        let mut jac = vec![0.0; block.jacobian().nnz()];
        let mut ws = model.create_workspace();
        block.assemble_jacobian_into(&mut ws, 0.0, &[10.0, 2.0, 3.0], &[], &mut jac);
        // dg/d(y1, y2) = [[y2, y1], [1, 2*y2]] at (y1, y2) = (2, 3).
        assert_eq!(jac, vec![3.0, 1.0, 2.0, 6.0]);
    }

    #[test]
    fn an_ode_has_no_algebraic_block() {
        assert!(doubling_model(2).algebraic().is_none());
    }
}
