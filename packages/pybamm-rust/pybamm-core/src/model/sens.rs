//! Forward sensitivities: the shared `df/dp` tape and its columns.

use crate::arena::{Arena, NodeId};
use crate::error::CoreError;
use crate::eval::{SplitCompiledExpr, TangentInputs};
use crate::jacobian::tangent_dag;
use crate::observable::seed_param_tangent;
use crate::params::ParamIdx;
use crate::simplify::simplify_pipeline;
use crate::tangent::tangent_wrt_params;

use super::{CompiledModel, FOREIGN_WORKSPACE, Workspace};

/// A [`CompiledModel`]'s `df/dp` tape paired with a workspace's sensitivity buffers.
#[derive(Debug)]
struct SensModel<'a> {
    expr: &'a SplitCompiledExpr,
    /// Registration index of each sensitivity column, in column order.
    columns: &'a [ParamIdx],
    scratch: &'a mut [f64],
    /// The parameter seed vector, in registration space.
    dp: &'a mut [f64],
    #[cfg(test)]
    primal_passes: &'a mut usize,
}

impl SensModel<'_> {
    /// Every column at one point, sharing a single primal pass.
    ///
    /// Column `i` lands in `output[i * stride..]`; `output` must hold one stride per column.
    fn all_columns(self, t: f64, y: &[f64], inputs: &[f64], stride: usize, output: &mut [f64]) {
        assert!(
            output.len() >= self.columns.len() * stride,
            "df/dp output holds {} elements but {} columns of {stride} were requested",
            output.len(),
            self.columns.len(),
        );
        #[cfg(test)]
        {
            *self.primal_passes += 1;
        }
        let mut primal = self.expr.eval_primal(self.scratch, t, y, &[], inputs);
        for (&param, column) in self.columns.iter().zip(output.chunks_mut(stride)) {
            self.dp.fill(0.0);
            self.dp[param.index()] = 1.0;
            let result = primal.eval_tangent(&TangentInputs {
                dy: None,
                dp: Some(self.dp),
            });
            column[..result.len()].copy_from_slice(result);
        }
    }

    /// `df/dp @ v`, where `v[i]` weighs sensitivity column `i`.
    fn action(self, t: f64, y: &[f64], inputs: &[f64], v: &[f64], output: &mut [f64]) {
        seed_param_tangent(self.dp, self.columns, v);
        let tangent = TangentInputs {
            dy: None,
            dp: Some(self.dp),
        };
        let result = self
            .expr
            .eval_with_tangent(self.scratch, t, y, &[], inputs, &tangent);
        output[..result.len()].copy_from_slice(result);
    }
}

impl CompiledModel {
    /// Compile the shared `df/dp` tape for the requested sensitivity columns.
    ///
    /// Each index must name a registered width-1 parameter; any other slot is not one column.
    pub(super) fn compile_sensitivities(
        &mut self,
        arena: &Arena,
        rhs: NodeId,
        sens_param_indices: &[ParamIdx],
    ) -> Result<(), CoreError> {
        if sens_param_indices.is_empty() {
            return Ok(());
        }
        let mut seen = vec![false; self.params.index_extent()];
        for &idx in sens_param_indices {
            let Some(slot) = self.params.slot(idx) else {
                return Err(CoreError::SensParamIndex {
                    index: idx.index(),
                    n_registered: self.params.n_registered(),
                });
            };
            if slot.width.get() != 1 {
                return Err(CoreError::SensParamVector {
                    index: idx.index(),
                    width: slot.width.index(),
                });
            }
            if std::mem::replace(&mut seen[idx.index()], true) {
                return Err(CoreError::SensParamRepeated { index: idx.index() });
            }
        }

        let df_dp = simplify_pipeline(tangent_dag(arena, rhs, tangent_wrt_params));
        self.sens_expr = Some(SplitCompiledExpr::compile(df_dp.as_ref()));
        self.sens_param_indices = sens_param_indices.to_vec();
        Ok(())
    }

    /// Number of sensitivity columns; zero without sensitivities.
    #[inline]
    pub const fn n_sens_params(&self) -> usize {
        self.sens_param_indices.len()
    }

    /// Registration index of each configured sensitivity column, in column order.
    #[inline]
    pub fn sens_param_indices(&self) -> &[ParamIdx] {
        &self.sens_param_indices
    }

    /// This model's sensitivity half paired with `ws`'s buffers.
    fn sens<'a>(&'a self, ws: &'a mut Workspace) -> SensModel<'a> {
        let expr = self
            .sens_expr
            .as_ref()
            .expect("this model was compiled without sensitivities");
        SensModel {
            expr,
            columns: &self.sens_param_indices,
            scratch: ws.sens_scratch.as_mut().expect(FOREIGN_WORKSPACE),
            dp: &mut ws.sens_dp_buffer,
            #[cfg(test)]
            primal_passes: &mut ws.sens_primal_passes,
        }
    }

    /// Evaluate ∂f/∂p for all configured sensitivity parameters into `output`.
    ///
    /// Layout: `output[i*n_states + j] = ∂f_j/∂p_i`.
    pub(crate) fn eval_sens_all(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        output: &mut [f64],
    ) {
        if self.n_sens_params() == 0 {
            return;
        }
        let n_states = self.n_states;
        self.sens(ws).all_columns(t, y, inputs, n_states, output);
    }

    /// Compute the sensitivity action `df/dp @ v`, where `v[i]` weighs sensitivity column `i`.
    pub fn sens_action(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        inputs: &[f64],
        v: &[f64],
        output: &mut [f64],
    ) {
        self.sens(ws).action(t, y, inputs, v, output);
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::{
        build_trailing_param_sens_model, compile_with_sens, doubling_model, identity_mass_matrix,
        input,
    };
    use super::*;
    use crate::model::CompiledModelOptions;
    use crate::node::Node;
    use crate::params::PackedWidth;

    /// f(y) = k * y^2 with k as `InputParameter` index 0.
    fn build_sens_test_model() -> (CompiledModel, [f64; 2]) {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let k = input(&mut arena, "k", 0, 0, 1);
        let two = arena.alloc(Node::Scalar(2.0));
        let y_sq = arena.alloc(Node::Pow(y, two));
        let rhs = arena.alloc(Node::Mul(k, y_sq));

        let model = compile_with_sens(
            &arena,
            rhs,
            identity_mass_matrix(2),
            2,
            1,
            &[ParamIdx::new(0)],
        );
        (model, [3.0, 4.0])
    }

    /// f(y) = k1 * y, with sensitivities for both k1 (idx 0) and k2 (idx 1).
    fn build_two_param_sens_model() -> (CompiledModel, [f64; 2]) {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let k1 = input(&mut arena, "k1", 0, 0, 1);
        let _k2 = input(&mut arena, "k2", 1, 1, 1);
        let rhs = arena.alloc(Node::Mul(k1, y));

        let model = compile_with_sens(
            &arena,
            rhs,
            identity_mass_matrix(2),
            2,
            2,
            &[ParamIdx::new(0), ParamIdx::new(1)],
        );
        (model, [3.0, 4.0])
    }

    /// Index 2 is within the packed width (4) but past the last registration index (1).
    #[test]
    fn a_sens_index_past_the_last_named_input_is_rejected() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let _wide = input(&mut arena, "wide", 0, 0, 3);
        let k = input(&mut arena, "k", 1, 3, 1);
        let rhs = arena.alloc(Node::Mul(k, y));

        let options = CompiledModelOptions {
            sensitivities: &[ParamIdx::new(2)],
            algebraic: None,
        };
        let err = CompiledModel::new(
            &arena,
            rhs,
            identity_mass_matrix(2),
            2,
            PackedWidth::new(4),
            options,
        )
        .expect_err("index 2 names no input");
        assert!(
            matches!(
                err,
                CoreError::SensParamIndex {
                    index: 2,
                    n_registered: 2
                }
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn a_sens_column_on_an_input_not_one_wide_is_rejected() {
        // A zero-width input was let through, and its derivative came out one row wide.
        for width in [3, 0] {
            let mut arena = Arena::new();
            let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
            let _other = input(&mut arena, "other", 0, 0, width);
            let k = input(&mut arena, "k", 1, width, 1);
            let rhs = arena.alloc(Node::Mul(k, y));

            let options = CompiledModelOptions {
                sensitivities: &[ParamIdx::new(0)],
                algebraic: None,
            };
            let err = CompiledModel::new(
                &arena,
                rhs,
                identity_mass_matrix(2),
                2,
                PackedWidth::new(width + 1),
                options,
            )
            .expect_err("only a width-1 input is one sensitivity column");
            assert!(
                matches!(err, CoreError::SensParamVector { index: 0, width: w } if w == width),
                "unexpected error: {err}"
            );
        }
    }

    #[test]
    fn duplicate_sens_param_indices_are_rejected() {
        // A repeated index would turn the subset scatter into an accumulate.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let a = input(&mut arena, "a", 0, 0, 1);
        let rhs = arena.alloc(Node::Mul(a, y));
        let options = CompiledModelOptions {
            sensitivities: &[ParamIdx::new(0), ParamIdx::new(0)],
            algebraic: None,
        };
        let err = CompiledModel::new(
            &arena,
            rhs,
            identity_mass_matrix(1),
            1,
            PackedWidth::new(1),
            options,
        )
        .expect_err("a repeated sensitivity index must be refused");
        assert!(
            matches!(err, CoreError::SensParamRepeated { index: 0 }),
            "unexpected error: {err}"
        );
    }

    /// The `dp` buffer is indexed in registration space, behind a vector input.
    #[test]
    fn a_sens_column_behind_a_vector_input_is_seeded_correctly() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let _wide = input(&mut arena, "wide", 0, 0, 3);
        let k = input(&mut arena, "k", 1, 3, 1);
        let rhs = arena.alloc(Node::Mul(k, y));

        let model = compile_with_sens(
            &arena,
            rhs,
            identity_mass_matrix(2),
            2,
            4,
            &[ParamIdx::new(1)],
        );

        let y_vals = [3.0, 4.0];
        let inputs = [1.0, 2.0, 3.0, 7.0];
        let mut out = [0.0; 2];
        model.eval_sens_all(
            &mut model.create_workspace(),
            0.0,
            &y_vals,
            &inputs,
            &mut out,
        );
        assert!((out[0] - y_vals[0]).abs() < 1e-12);
        assert!((out[1] - y_vals[1]).abs() < 1e-12);
    }

    #[test]
    fn test_sens_expr_runs_one_primal_pass_per_point() {
        let (model, y) = build_two_param_sens_model();
        let mut ws = model.create_workspace();

        let mut out = vec![0.0; 2 * model.n_states()];
        model.eval_sens_all(&mut ws, 0.0, &y, &[7.0, 1.0], &mut out);
        assert_eq!(ws.sens_primal_passes, 1);
    }

    #[test]
    fn test_eval_sens_matches_analytical() {
        let (model, y_vals) = build_sens_test_model();
        assert_eq!(model.n_sens_params(), 1);

        // y = [3, 4], k = 7 -> ∂f/∂k = y^2 = [9, 16]
        let mut out = [0.0; 2];
        model.eval_sens_all(
            &mut model.create_workspace(),
            0.0,
            &y_vals,
            &[7.0],
            &mut out,
        );
        assert!((out[0] - 9.0).abs() < 1e-10, "expected 9, got {}", out[0]);
        assert!((out[1] - 16.0).abs() < 1e-10, "expected 16, got {}", out[1]);
    }

    #[test]
    fn test_eval_sens_all_layout() {
        let (model, y_vals) = build_two_param_sens_model();

        let mut out = vec![-1.0; 4];
        let mut ws = model.create_workspace();
        model.eval_sens_all(&mut ws, 0.0, &y_vals, &[5.0, 9.0], &mut out);

        assert!((out[0] - 3.0).abs() < 1e-10);
        assert!((out[1] - 4.0).abs() < 1e-10);
        assert!(out[2].abs() < 1e-10);
        assert!(out[3].abs() < 1e-10);
    }

    #[test]
    fn test_eval_sens_all_matches_the_unit_sens_actions() {
        let (model, y) = build_two_param_sens_model();
        let mut ws = model.create_workspace();
        let (n, k) = (model.n_states(), model.n_sens_params());
        let inputs = [1.5, 2.5];
        let mut batched = vec![0.0; n * k];
        model.eval_sens_all(&mut ws, 0.0, &y, &inputs, &mut batched);
        for i in 0..k {
            let mut unit = vec![0.0; k];
            unit[i] = 1.0;
            let mut single = vec![0.0; n];
            model.sens_action(&mut ws, 0.0, &y, &inputs, &unit, &mut single);
            assert_eq!(
                &batched[i * n..(i + 1) * n],
                &single[..],
                "column {i} diverged"
            );
        }
    }

    #[test]
    fn sens_action_seeds_its_columns_in_parameter_space() {
        // Column 0 is k1; a column-space read of the seed would differentiate by k0 instead.
        let (model, y) = build_trailing_param_sens_model();
        let mut ws = model.create_workspace();
        let mut got = [0.0; 2];
        model.sens_action(&mut ws, 0.0, &y, &[5.0, 9.0], &[1.0], &mut got);
        assert!((got[0] - 9.0).abs() < 1e-10, "expected 9, got {}", got[0]);
        assert!((got[1] - 16.0).abs() < 1e-10, "expected 16, got {}", got[1]);
    }

    #[test]
    #[should_panic(expected = "tangent length must match")]
    fn a_sens_action_direction_of_another_length_is_refused() {
        // Silently zipped down to the column count before, dropping the second entry.
        let (model, y) = build_trailing_param_sens_model();
        let mut out = [0.0; 2];
        let mut ws = model.create_workspace();
        model.sens_action(&mut ws, 0.0, &y, &[5.0, 9.0], &[1.0, 1.0], &mut out);
    }

    #[test]
    fn test_eval_sens_all_noop_without_sensitivities() {
        let model = doubling_model(2);
        assert_eq!(model.n_sens_params(), 0);
        let mut out: [f64; 0] = [];
        model.eval_sens_all(
            &mut model.create_workspace(),
            0.0,
            &[3.0, 4.0],
            &[],
            &mut out,
        );
    }
}
