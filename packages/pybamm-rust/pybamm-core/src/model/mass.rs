//! The mass matrix `M`: which states it makes algebraic, and its products.

use crate::node::CsrData;

use super::{CompiledModel, Workspace};

/// Which states a mass matrix makes differential and which algebraic.
///
/// State `i` is algebraic iff column `i` stores no non-zero, as `M y'` reads `y'_i` only there.
#[derive(Debug, PartialEq, Eq)]
pub struct StateKinds {
    algebraic: Vec<bool>,
}

impl StateKinds {
    /// Classify every state of a square CSR mass matrix.
    // Exact `== 0.0`: an epsilon would reclassify a stiff-but-real row.
    #[allow(clippy::float_cmp)]
    pub(crate) fn from_mass_matrix(mass: &CsrData) -> Self {
        debug_assert_eq!(
            mass.shape().rows,
            mass.shape().cols,
            "a mass matrix classifies states only while it is square",
        );
        let mut algebraic = vec![true; mass.shape().cols];
        for (&col, &value) in mass.indices().iter().zip(mass.data()) {
            if value != 0.0 {
                algebraic[col] = false;
            }
        }
        Self { algebraic }
    }

    pub(crate) fn is_differential(&self, state: usize) -> bool {
        !self.algebraic[state]
    }

    /// One flag per state, `true` where algebraic.
    pub(crate) fn algebraic_mask(&self) -> &[bool] {
        &self.algebraic
    }

    pub fn n_algebraic(&self) -> usize {
        self.algebraic.iter().filter(|&&a| a).count()
    }

    /// Write IDA's `id` vector: `1.0` differential, `0.0` algebraic.
    pub fn write_ida_id(&self, output: &mut [f64]) {
        for (out, &is_algebraic) in output.iter_mut().zip(&self.algebraic) {
            *out = if is_algebraic { 0.0 } else { 1.0 };
        }
    }
}

/// Classification of the mass matrix for hot-path dispatch.
#[derive(Debug)]
pub enum MassKind {
    /// `M = I` (pure ODE).
    Identity,
    /// `M = diag(mask)`; the mask is read from the model's [`StateKinds`].
    DiagonalSelector,
    /// Arbitrary sparse `M`, via CSR matvec.
    General,
}

/// Classify a CSR mass matrix into the fastest applicable [`MassKind`].
// Exact compares: an epsilon would misclassify a near-identity matrix as `Identity`.
#[allow(clippy::float_cmp)]
pub fn classify_mass_matrix(mass: &CsrData) -> MassKind {
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

    MassKind::DiagonalSelector
}

impl CompiledModel {
    /// Compute DAE residual r = M*y' - f(t, y).
    pub fn eval_residual(
        &self,
        ws: &mut Workspace,
        t: f64,
        y: &[f64],
        yp: &[f64],
        inputs: &[f64],
        r: &mut [f64],
    ) {
        let f = self
            .primal_expr
            .eval(&mut ws.primal_scratch, t, y, &[], inputs);
        let n = self.n_states;
        let (rows, yp) = (r[..n].iter_mut().zip(&f[..n]), &yp[..n]);
        match &self.mass_kind {
            MassKind::Identity => {
                for ((r_i, &f_i), &yp_i) in rows.zip(yp) {
                    *r_i = yp_i - f_i;
                }
            },
            MassKind::DiagonalSelector => {
                let mask = self.state_kinds.algebraic_mask();
                for (((r_i, &f_i), &yp_i), &algebraic) in rows.zip(yp).zip(mask) {
                    *r_i = if algebraic { 0.0 } else { yp_i } - f_i;
                }
            },
            MassKind::General => {
                for (row, (r_i, &f_i)) in rows.enumerate() {
                    *r_i = csr_row_dot(&self.mass_matrix, row, yp) - f_i;
                }
            },
        }
    }

    /// Compute mass matrix action: M @ x (no beta term).
    pub(crate) fn mass_action(&self, x: &[f64], mv: &mut [f64]) {
        let n = self.n_states;
        match &self.mass_kind {
            MassKind::Identity => mv[..n].copy_from_slice(&x[..n]),
            MassKind::DiagonalSelector => {
                let mask = self.state_kinds.algebraic_mask();
                for ((mv_i, &x_i), &algebraic) in mv[..n].iter_mut().zip(&x[..n]).zip(mask) {
                    *mv_i = if algebraic { 0.0 } else { x_i };
                }
            },
            MassKind::General => csr_matvec(&self.mass_matrix, x, mv),
        }
    }

    /// Fold `-cj*M` into the merged CSC; must run after every derivative fill.
    pub(super) fn apply_mass_postpass(&self, cj: f64, jac_data: &mut [f64]) {
        if cj == 0.0 {
            return;
        }
        for (mass_idx, &csc_idx) in self.mass_to_csc_map.iter().enumerate() {
            jac_data[csc_idx] -= cj * self.mass_matrix.data[mass_idx];
        }
    }
}

/// CSR matrix-vector product: y = A @ x
#[inline]
fn csr_matvec(csr: &CsrData, x: &[f64], y: &mut [f64]) {
    for (row, y_row) in y[..csr.shape.rows].iter_mut().enumerate() {
        *y_row = csr_row_dot(csr, row, x);
    }
}

/// `A[row, :] @ x`, accumulated from `0.0` in stored order.
#[inline]
fn csr_row_dot(csr: &CsrData, row: usize, x: &[f64]) -> f64 {
    let span = csr.indptr[row]..csr.indptr[row + 1];
    let mut sum = 0.0;
    for (&value, &col) in csr.data[span.clone()].iter().zip(&csr.indices[span]) {
        sum += value * x[col];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::{
        assemble_dense, assemble_dense_no_mass, assert_matrices_match, compile, dense_row_model,
        doubling_model, finite_difference_jacobian, identity_mass_matrix, less_cj_on_the_diagonal,
    };
    use super::*;
    use crate::arena::Arena;
    use crate::node::{Node, Shape};

    #[test]
    fn csr_matvec_takes_each_rows_dot_product() {
        // diag(2, 3), then [[1, 2, 0], [3, 4, 5], [0, 6, 7]].
        let cases = [
            (vec![0, 1, 2], vec![0, 1], vec![2.0, 3.0], vec![2.0, 6.0]),
            (
                vec![0, 2, 5, 7],
                vec![0, 1, 0, 1, 2, 1, 2],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                vec![5.0, 26.0, 33.0],
            ),
        ];
        for (indptr, indices, data, expected) in cases {
            let n = expected.len();
            let csr = CsrData {
                shape: Shape::matrix(n, n),
                indptr,
                indices,
                data,
            };
            let x: Vec<f64> = (1..=n).map(|i| i as f64).collect();
            let mut y = vec![0.0; n];
            csr_matvec(&csr, &x, &mut y);
            for (got, want) in y.iter().zip(&expected) {
                assert!((got - want).abs() < 1e-14, "got {y:?}, want {expected:?}");
            }
        }
    }

    #[test]
    fn the_residual_is_m_yp_minus_f() {
        // f = 2y under identity mass, then f = y under diag(2, 3).
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let scaled_mass = CsrData {
            shape: Shape::matrix(2, 2),
            indptr: vec![0, 1, 2],
            indices: vec![0, 1],
            data: vec![2.0, 3.0],
        };
        let cases = [
            (doubling_model(2), [1.0, 0.0]),
            (compile(&arena, y, scaled_mass, 2, 0), [5.0, 10.0]),
        ];
        for (model, expected) in cases {
            let mut r = [0.0, 0.0];
            let mut ws = model.create_workspace();
            model.eval_residual(&mut ws, 0.0, &[1.0, 2.0], &[3.0, 4.0], &[], &mut r);
            for (got, want) in r.iter().zip(expected) {
                assert!((got - want).abs() < 1e-14, "got {r:?}, want {expected:?}");
            }
        }
    }

    #[test]
    fn a_zero_mass_column_makes_its_state_algebraic() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let mass = CsrData {
            shape: Shape::matrix(3, 3),
            indptr: vec![0, 1, 1, 2],
            indices: vec![0, 2],
            data: vec![1.0, 1.0],
        };
        let model = compile(&arena, y, mass, 3, 0);
        assert_eq!(model.state_kinds().algebraic_mask(), &[false, true, false]);
        let mut ids = vec![0.0; 3];
        model.state_kinds().write_ida_id(&mut ids);
        assert_eq!(ids, vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn an_explicitly_stored_zero_diagonal_is_algebraic() {
        let mass = CsrData {
            indptr: vec![0, 1, 2],
            indices: vec![0, 1],
            data: vec![1.0, 0.0],
            shape: Shape::matrix(2, 2),
        };
        let kinds = StateKinds::from_mass_matrix(&mass);

        assert_eq!(kinds.algebraic_mask(), &[false, true]);
        assert_eq!(kinds.n_algebraic(), 1);
        let mut ids = [f64::NAN; 2];
        kinds.write_ida_id(&mut ids);
        assert_eq!(ids.map(f64::to_bits), [1.0_f64, 0.0].map(f64::to_bits));
    }

    #[test]
    fn test_algebraic_ids_identity_mass_all_differential() {
        let model = doubling_model(4);
        assert_eq!(
            model.state_kinds().algebraic_mask(),
            &[false, false, false, false]
        );
    }

    /// A row test would call state 1 differential; nothing in `M` multiplies `y'_1`.
    #[test]
    fn an_off_diagonal_only_row_leaves_its_own_state_algebraic() {
        let mass = CsrData::try_new(
            vec![0, 1, 2, 2],
            vec![0, 2],
            vec![1.0, 1.0],
            Shape::matrix(3, 3),
        )
        .expect("valid mass matrix");

        let kinds = StateKinds::from_mass_matrix(&mass);
        assert_eq!(kinds.algebraic_mask(), &[false, true, false]);
        assert_eq!(kinds.n_algebraic(), 1);

        assert!(kinds.algebraic_mask()[1]);
        assert!(kinds.is_differential(2));
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
        assert!(matches!(
            classify_mass_matrix(&mass),
            MassKind::DiagonalSelector
        ));
        assert_eq!(
            StateKinds::from_mass_matrix(&mass).algebraic_mask(),
            &[false, true]
        );
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

    /// One 2x2 mass matrix of each [`MassKind`].
    fn masses_of_every_kind() -> [CsrData; 3] {
        [
            (vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]),
            (vec![0, 1, 1], vec![0], vec![1.0]),
            (vec![0, 2, 3], vec![0, 1, 1], vec![1.0, 0.5, 1.0]),
        ]
        .map(|(indptr, indices, data)| CsrData {
            indptr,
            indices,
            data,
            shape: Shape::matrix(2, 2),
        })
    }

    #[test]
    fn residual_and_assembly_apply_mass_action_on_every_mass_kind() {
        let (y, yp, cj) = ([0.3, -1.2], [2.0, -0.7], 0.25);
        for mass in masses_of_every_kind() {
            let mut arena = Arena::new();
            let state = arena.alloc(Node::StateVector { start: 0, end: 2 });
            let rhs = arena.alloc(Node::Sin(state));
            let model = compile(&arena, rhs, mass, 2, 0);
            let mut ws = model.create_workspace();

            let (mut f, mut m_yp, mut residual) = ([0.0; 2], [0.0; 2], [0.0; 2]);
            model.eval_rhs(&mut ws, 0.0, &y, &[], &mut f);
            model.mass_action(&yp, &mut m_yp);
            model.eval_residual(&mut ws, 0.0, &y, &yp, &[], &mut residual);
            let expected = [m_yp[0] - f[0], m_yp[1] - f[1]];
            assert_eq!(residual.map(f64::to_bits), expected.map(f64::to_bits));

            let assembled = assemble_dense(&model, cj, &y, &[]);
            for col in 0..2 {
                let mut unit = [0.0; 2];
                unit[col] = 1.0;
                let (mut jv, mut m_v) = ([0.0; 2], [0.0; 2]);
                model.jac_action(&mut ws, 0.0, &y, &[], &unit, &mut jv);
                model.mass_action(&unit, &mut m_v);
                for row in 0..2 {
                    let expected = jv[row] - cj * m_v[row];
                    assert!(
                        (assembled[row][col] - expected).abs() < 1e-15,
                        "({row}, {col}): assembled {}, actions {expected}",
                        assembled[row][col]
                    );
                }
            }
        }
    }

    #[test]
    fn test_dense_row_merged_remap_tolerates_mass_only_column() {
        // Dense row `sum_{j>=1} y_j^2` first: merged slot (0,0) is mass-only.
        let n = 20;
        let model = dense_row_model(n, 1..n, true, identity_mass_matrix(n));
        let y: Vec<f64> = (0..n)
            .map(|i| (i as f64).mul_add(0.1, 0.4).cos() + 0.6)
            .collect();
        assert_eq!(model.jac.n_dense_rows(), 1, "split must be active");
        assert_eq!(model.jac.dense_row_indices().next(), Some(0));

        let dense_fd = finite_difference_jacobian(&model, &y, 1e-6);

        let cj = 0.5;
        let assembled = assemble_dense(&model, cj, &y, &[]);
        assert!(
            (assembled[0][0] - (-cj)).abs() < 1e-12,
            "mass-only slot (0,0) must be -cj, got {}",
            assembled[0][0]
        );
        let expected = less_cj_on_the_diagonal(dense_fd.clone(), cj, n);
        assert_matrices_match(&assembled, &expected, 1e-5, "CSC-mass");

        let no_mass = assemble_dense_no_mass(&model, &y);
        assert_matrices_match(&no_mass, &dense_fd, 1e-5, "CSC-no-mass");
    }

    #[test]
    fn test_dense_row_shared_dfdy_mass_slot_survives_postpass() {
        // Dense row `sum_j y_j^2` first: merged slot (0,0) holds `df/dy` and mass.
        let n = 20;
        let model = dense_row_model(n, 0..n, true, identity_mass_matrix(n));
        let y: Vec<f64> = (0..n)
            .map(|i| (i as f64).mul_add(0.2, 0.5).sin() + 0.8)
            .collect();
        assert_eq!(model.jac.n_dense_rows(), 1, "split must be active");
        assert_eq!(model.jac.dense_row_indices().next(), Some(0));

        let dense_fd = finite_difference_jacobian(&model, &y, 1e-6);

        // A postpass-before-fill reorder would clobber the -cj in slot (0,0).
        let cj = 0.5;
        let assembled = assemble_dense(&model, cj, &y, &[]);
        let analytic = 2.0f64.mul_add(y[0], -cj);
        assert!(
            (assembled[0][0] - analytic).abs() < 1e-12,
            "shared slot (0,0) must be 2*y0 - cj = {analytic}, got {}",
            assembled[0][0]
        );
        let expected = less_cj_on_the_diagonal(dense_fd, cj, n);
        assert_matrices_match(&assembled, &expected, 1e-5, "CSC-mass-shared");
    }

    #[test]
    fn test_loop_b_sparse_scan_mass_only_slot_not_aliased() {
        let mut arena = Arena::new();
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let f0 = arena.alloc(Node::Sin(y1));
        let f1 = arena.alloc(Node::Sin(y1));
        let rhs = arena.alloc(Node::Concat(vec![f0, f1]));
        let model = compile(&arena, rhs, identity_mass_matrix(2), 2, 0);

        assert!(
            model.jac.n_dense_rows() == 0,
            "fixture must exercise the sparse scan, not the dense-row path"
        );
        assert_eq!(
            model.jac.coloring().colors()[0],
            crate::coloring::UNSEEDED,
            "fixture requires col 0 to be absent from df/dy"
        );

        let y = [0.3, 0.5];
        let dcos = 0.5_f64.cos();

        let dense = assemble_dense(&model, 0.0, &y, &[]);
        assert!(
            dense[0][0].abs() < 1e-12,
            "mass-only slot (0,0) must be 0 at cj=0, got {}",
            dense[0][0]
        );
        assert!(
            (dense[0][1] - dcos).abs() < 1e-12,
            "df/dy slot (0,1) must be cos(y1), got {}",
            dense[0][1]
        );

        let cj = 0.5;
        let dense = assemble_dense(&model, cj, &y, &[]);
        assert!(
            (dense[0][0] - (-cj)).abs() < 1e-12,
            "mass-only slot (0,0) must be -cj, got {}",
            dense[0][0]
        );
        assert!(
            (dense[0][1] - dcos).abs() < 1e-12,
            "df/dy slot (0,1) must stay cos(y1) at cj!=0, got {}",
            dense[0][1]
        );
    }
}
