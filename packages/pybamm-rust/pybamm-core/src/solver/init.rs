//! Initial-condition operator, with a per-solve `dy0/dp` seed (empty means zero).

use diffsol::matrix::sparse_faer::FaerSparseMat;
use diffsol::vector::faer_serial::FaerVec;
use diffsol::{ConstantOp, ConstantOpSens, FaerContext, Op};

use super::{FaerSparsity, dense_faer_sparsity, lane, lane_mut, values_mut};

/// Initial-condition operator returning the stored `y0`.
#[derive(Debug)]
pub struct InitOp<'a> {
    pub y0: &'a [f64],
    /// `dy0/dp`, column-major `n_states x n_sens_params`; empty means zero.
    pub y0_sens: &'a [f64],
    pub n_sens_params: usize,
    pub context: FaerContext,
}

impl Op for InitOp<'_> {
    type T = f64;
    type V = FaerVec<f64>;
    type M = FaerSparseMat<f64>;
    type C = FaerContext;

    fn nstates(&self) -> usize {
        0
    }
    fn nout(&self) -> usize {
        self.y0.len()
    }
    fn nparams(&self) -> usize {
        self.n_sens_params
    }
    fn context(&self) -> &Self::C {
        &self.context
    }
}

impl ConstantOp for InitOp<'_> {
    fn call_inplace(&self, _t: f64, y: &mut FaerVec<f64>) {
        lane_mut(y).copy_from_slice(self.y0);
    }
}

impl ConstantOpSens for InitOp<'_> {
    /// Write `dy0/dp · v` into `y`, or zeros if no seed was given.
    fn sens_mul_inplace(&self, _t: f64, v: &FaerVec<f64>, y: &mut FaerVec<f64>) {
        let out = lane_mut(y);
        out.fill(0.0);
        if self.y0_sens.is_empty() {
            return;
        }
        let n_states = self.y0.len();
        for (col, &v_col) in lane(v).iter().enumerate() {
            if v_col == 0.0 {
                continue;
            }
            let offset = col * n_states;
            for (out_i, &s_i) in out.iter_mut().zip(&self.y0_sens[offset..offset + n_states]) {
                *out_i += v_col * s_i;
            }
        }
    }

    /// Write the full `dy0/dp` matrix into `y`, or zeros if no seed was given.
    ///
    /// `y` uses the dense column-major pattern of `y0_sens`, so values copy directly.
    fn sens_inplace(&self, _t: f64, y: &mut FaerSparseMat<f64>) {
        let values = values_mut(y);
        if self.y0_sens.is_empty() {
            values.fill(0.0);
        } else {
            values.copy_from_slice(self.y0_sens);
        }
    }

    fn sens_sparsity(&self) -> Option<FaerSparsity> {
        Some(dense_faer_sparsity(self.y0.len(), self.n_sens_params))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diffsol::{Context, Matrix, Vector};

    fn init_op<'a>(y0: &'a [f64], y0_sens: &'a [f64], n_sens_params: usize) -> InitOp<'a> {
        InitOp {
            y0,
            y0_sens,
            n_sens_params,
            context: FaerContext::default(),
        }
    }

    fn sens_column(op: &InitOp<'_>, col: usize) -> Vec<f64> {
        let ctx = FaerContext::default();
        let mut v = ctx.vector_zeros::<FaerVec<f64>>(op.n_sens_params);
        v.set_index(col, 1.0);
        let mut y = ctx.vector_zeros::<FaerVec<f64>>(op.y0.len());
        op.sens_mul_inplace(0.0, &v, &mut y);
        lane(&y).to_vec()
    }

    #[test]
    fn empty_seed_is_all_zero() {
        let y0 = [0.0; 3];
        let op = init_op(&y0, &[], 2);
        assert_eq!(sens_column(&op, 0), vec![0.0; 3]);
        assert_eq!(sens_column(&op, 1), vec![0.0; 3]);
    }

    #[test]
    fn mixed_direction_is_the_matvec() {
        let y0 = [0.0; 3];
        let seed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let op = init_op(&y0, &seed, 2);
        let ctx = FaerContext::default();
        let mut v = ctx.vector_zeros::<FaerVec<f64>>(2);
        v.set_index(0, 2.0);
        v.set_index(1, -1.0);
        let mut y = ctx.vector_zeros::<FaerVec<f64>>(3);
        op.sens_mul_inplace(0.0, &v, &mut y);
        assert_eq!(lane(&y), &[2.0 - 4.0, 4.0 - 5.0, 6.0 - 6.0]);
    }

    fn dense_sens_matrix(op: &InitOp<'_>) -> FaerSparseMat<f64> {
        FaerSparseMat::new_from_sparsity(
            op.y0.len(),
            op.n_sens_params,
            op.sens_sparsity(),
            FaerContext::default(),
        )
    }

    #[test]
    fn sens_inplace_copies_the_seed_the_probing_default_reads() {
        let y0 = [0.0; 3];
        let seed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let op = init_op(&y0, &seed, 2);
        let mut copied = dense_sens_matrix(&op);
        let mut probed = dense_sens_matrix(&op);
        op.sens_inplace(0.0, &mut copied);
        op._default_sens_inplace(0.0, &mut probed);
        assert_eq!(values_mut(&mut copied), &seed);
        assert_eq!(values_mut(&mut copied), values_mut(&mut probed));
    }

    #[test]
    fn sens_inplace_of_an_empty_seed_overwrites_with_zeros() {
        let y0 = [0.0; 3];
        let op = init_op(&y0, &[], 2);
        let mut m = dense_sens_matrix(&op);
        values_mut(&mut m).fill(9.0);
        op.sens_inplace(0.0, &mut m);
        assert_eq!(values_mut(&mut m), &[0.0; 6]);
    }

    #[test]
    fn output_is_overwritten_not_accumulated() {
        let y0 = [0.0; 3];
        let seed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let op = init_op(&y0, &seed, 2);
        let ctx = FaerContext::default();
        let mut v = ctx.vector_zeros::<FaerVec<f64>>(2);
        v.set_index(0, 1.0);
        let mut y = ctx.vector_zeros::<FaerVec<f64>>(3);
        y.set_index(0, 99.0);
        op.sens_mul_inplace(0.0, &v, &mut y);
        assert_eq!(lane(&y), &[1.0, 2.0, 3.0]);
    }
}
