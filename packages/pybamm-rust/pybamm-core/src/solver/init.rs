//! Initial-condition operator.
//!
//! `y0` is supplied per solve, and so is its parameter derivative `dy0/dp`: a
//! parameter that feeds an initial condition seeds the sensitivity system with
//! a non-zero column, exactly as IDAS does. An empty seed means `dy0/dp = 0`.

use diffsol::matrix::sparse_faer::FaerSparseMat;
use diffsol::vector::faer_serial::FaerVec;
use diffsol::{ConstantOp, ConstantOpSens, FaerContext, Op, VectorHost};

/// Initial-condition operator: returns the stored y0 at solve time, borrowed
/// from the [`Equations`](super::equations::Equations) that owns the solve.
#[derive(Debug)]
pub struct InitOp<'a> {
    pub y0: &'a [f64],
    /// `dy0/dp`, column-major `n_states x n_sens_params`; empty means zero.
    pub y0_sens: &'a [f64],
    pub n_states: usize,
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
        self.n_states
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
        y.as_mut_slice().copy_from_slice(self.y0);
    }
}

impl ConstantOpSens for InitOp<'_> {
    /// `dy0/dp · v`. diffsol drives this with unit vectors to read off one
    /// column at a time, then its consistent-IC augmentation overwrites the
    /// algebraic rows; the differential rows are taken as seeded here.
    fn sens_mul_inplace(&self, _t: f64, v: &FaerVec<f64>, y: &mut FaerVec<f64>) {
        let out = y.as_mut_slice();
        out.fill(0.0);
        if self.y0_sens.is_empty() {
            return;
        }
        for (col, &v_col) in v.as_slice().iter().enumerate() {
            if v_col == 0.0 {
                continue;
            }
            let offset = col * self.n_states;
            for (out_i, &s_i) in out
                .iter_mut()
                .zip(&self.y0_sens[offset..offset + self.n_states])
            {
                *out_i += v_col * s_i;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diffsol::{Context, Vector};

    fn init_op<'a>(y0: &'a [f64], y0_sens: &'a [f64], n_sens_params: usize) -> InitOp<'a> {
        InitOp {
            y0,
            y0_sens,
            n_states: y0.len(),
            n_sens_params,
            context: FaerContext::default(),
        }
    }

    fn sens_column(op: &InitOp<'_>, col: usize) -> Vec<f64> {
        let ctx = FaerContext::default();
        let mut v = ctx.vector_zeros::<FaerVec<f64>>(op.n_sens_params);
        v.set_index(col, 1.0);
        let mut y = ctx.vector_zeros::<FaerVec<f64>>(op.n_states);
        op.sens_mul_inplace(0.0, &v, &mut y);
        y.as_slice().to_vec()
    }

    #[test]
    fn empty_seed_is_all_zero() {
        let y0 = [0.0; 3];
        let op = init_op(&y0, &[], 2);
        assert_eq!(sens_column(&op, 0), vec![0.0; 3]);
        assert_eq!(sens_column(&op, 1), vec![0.0; 3]);
    }

    #[test]
    fn unit_directions_read_off_columns() {
        // Column-major 3x2: column 0 = [1,2,3], column 1 = [4,5,6].
        let y0 = [0.0; 3];
        let seed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let op = init_op(&y0, &seed, 2);
        assert_eq!(sens_column(&op, 0), vec![1.0, 2.0, 3.0]);
        assert_eq!(sens_column(&op, 1), vec![4.0, 5.0, 6.0]);
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
        assert_eq!(y.as_slice(), &[2.0 - 4.0, 4.0 - 5.0, 6.0 - 6.0]);
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
        assert_eq!(y.as_slice(), &[1.0, 2.0, 3.0]);
    }
}
