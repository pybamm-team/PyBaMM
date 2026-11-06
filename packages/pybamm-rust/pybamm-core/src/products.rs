//! Matrix products, each a [`Product`] that [`dispatch`](crate::accumulate::dispatch) runs
//! fused or unfused.
//!
//! Every sparse product walks its rows through [`CsrData::rows`], so the scalar, lane-batched
//! and transposed forms sum in the same stored order.

// Products must inline into `pulp`'s feature-enabled wrapper to compile as FMA instructions.
#![allow(clippy::inline_always)]

use crate::accumulate::{Accumulate, Product};
use crate::node::CsrData;

impl CsrData {
    /// Each row as its stored `(columns, values)`, in row order.
    ///
    /// The one row walk behind every sparse product; [`CsrData::try_new`] guarantees the splits
    /// fit.
    #[inline(always)]
    pub(crate) fn rows(&self) -> impl Iterator<Item = (&[usize], &[f64])> {
        let (mut cols, mut vals) = (&self.indices[..], &self.data[..]);
        self.indptr.windows(2).map(move |w| {
            let (row_cols, rest_cols) = cols.split_at(w[1] - w[0]);
            let (row_vals, rest_vals) = vals.split_at(w[1] - w[0]);
            (cols, vals) = (rest_cols, rest_vals);
            (row_cols, row_vals)
        })
    }

    /// `A[row, :] @ x`, accumulated unfused from `0.0` in stored order.
    #[inline]
    pub(crate) fn row_dot(&self, row: usize, x: &[f64]) -> f64 {
        let span = self.indptr[row]..self.indptr[row + 1];
        let mut sum = 0.0;
        for (&value, &col) in self.data[span.clone()].iter().zip(&self.indices[span]) {
            sum += value * x[col];
        }
        sum
    }

    /// `y = A @ x`, each row summed as [`row_dot`](Self::row_dot) sums it.
    #[inline]
    pub(crate) fn matvec(&self, x: &[f64], y: &mut [f64]) {
        for (row, y_row) in y[..self.shape.rows].iter_mut().enumerate() {
            *y_row = self.row_dot(row, x);
        }
    }
}

/// `out = A @ x` for a CSR `A`, each row summed from `0.0` in stored order.
pub struct CsrProduct<'a> {
    pub csr: &'a CsrData,
    pub x: &'a [f64],
    pub out: &'a mut [f64],
}

impl Product for CsrProduct<'_> {
    type Output = ();

    #[inline(always)]
    fn run<A: Accumulate>(self) {
        for (o, (cols, vals)) in self.out.iter_mut().zip(self.csr.rows()) {
            let mut sum = 0.0;
            for (&col, &val) in cols.iter().zip(vals) {
                sum = A::mul_acc(sum, val, self.x[col]);
            }
            *o = sum;
        }
    }
}

/// `buf[dst..] = M @ buf[vec..]` for the row-major `rows x cols` matrix at `buf[mat..]`.
pub struct DenseProduct<'a> {
    pub buf: &'a mut [f64],
    pub mat: usize,
    pub vec: usize,
    pub dst: usize,
    pub rows: usize,
    pub cols: usize,
}

impl Product for DenseProduct<'_> {
    type Output = ();

    #[inline(always)]
    fn run<A: Accumulate>(self) {
        let Self {
            buf,
            mat,
            vec,
            dst,
            rows,
            cols,
        } = self;
        for row in 0..rows {
            let mat_row = &buf[mat + row * cols..][..cols];
            let mut sum = 0.0;
            for (&m, &v) in mat_row.iter().zip(&buf[vec..vec + cols]) {
                sum = A::mul_acc(sum, m, v);
            }
            buf[dst + row] = sum;
        }
    }
}

/// `out = A @ x` across `k` lanes, each lane accumulated as the scalar `CsrProduct` does.
pub struct CsrLanesProduct<'a> {
    pub csr: &'a CsrData,
    pub x: &'a [f64],
    pub out: &'a mut [f64],
    pub k: usize,
}

impl Product for CsrLanesProduct<'_> {
    type Output = ();

    #[inline(always)]
    fn run<A: Accumulate>(self) {
        let Self { csr, x, out, k } = self;
        for (acc, (cols, vals)) in out.chunks_exact_mut(k).zip(csr.rows()) {
            acc.fill(0.0);
            for (&col, &val) in cols.iter().zip(vals) {
                for (a, &xi) in acc.iter_mut().zip(&x[col * k..][..k]) {
                    *a = A::mul_acc(*a, val, xi);
                }
            }
        }
    }
}

/// `out = M @ v` across `k` lanes for a row-major `M` with `cols` columns.
pub struct DenseLanesProduct<'a> {
    pub m: &'a [f64],
    pub v: &'a [f64],
    pub out: &'a mut [f64],
    pub cols: usize,
    pub k: usize,
}

impl Product for DenseLanesProduct<'_> {
    type Output = ();

    #[inline(always)]
    fn run<A: Accumulate>(self) {
        let Self { m, v, out, cols, k } = self;
        for (row, acc) in out.chunks_exact_mut(k).enumerate() {
            acc.fill(0.0);
            let m_row = &m[row * cols * k..][..cols * k];
            for (m_lanes, v_lanes) in m_row.chunks_exact(k).zip(v.chunks_exact(k)) {
                for ((a, &mi), &vi) in acc.iter_mut().zip(m_lanes).zip(v_lanes) {
                    *a = A::mul_acc(*a, mi, vi);
                }
            }
        }
    }
}

/// `out = A @ x` for `K` tangent directions, each accumulated as the scalar `CsrProduct` does.
pub struct CsrTangentLanesProduct<'a, const K: usize> {
    pub csr: &'a CsrData,
    pub x: &'a [f64],
    pub out: &'a mut [f64],
}

impl<const K: usize> Product for CsrTangentLanesProduct<'_, K> {
    type Output = ();

    #[inline(always)]
    fn run<A: Accumulate>(self) {
        let Self { csr, x, out } = self;
        for (out, (cols, vals)) in out.as_chunks_mut::<K>().0.iter_mut().zip(csr.rows()) {
            let mut acc = [0.0_f64; K];
            for (&col, &value) in cols.iter().zip(vals) {
                for (a, &v) in acc.iter_mut().zip(&x[col * K..][..K]) {
                    *a = A::mul_acc(*a, value, v);
                }
            }
            out.copy_from_slice(&acc);
        }
    }
}

/// `bar_vec += Aᵀ @ bar_dst` for a CSR `A`, skipping rows with no adjoint.
pub struct CsrTransposedProduct<'a> {
    pub csr: &'a CsrData,
    pub bar_dst: &'a [f64],
    pub bar_vec: &'a mut [f64],
}

impl Product for CsrTransposedProduct<'_> {
    type Output = ();

    #[inline(always)]
    fn run<A: Accumulate>(self) {
        let Self {
            csr,
            bar_dst,
            bar_vec,
        } = self;
        for (&bd, (cols, vals)) in bar_dst.iter().zip(csr.rows()) {
            if bd == 0.0 {
                continue;
            }
            for (&col, &value) in cols.iter().zip(vals) {
                bar_vec[col] = A::mul_acc(bar_vec[col], value, bd);
            }
        }
    }
}

/// `bar[vec..] += Mᵀ @ bar[dst..]` for the row-major `rows x cols` matrix at `scratch[mat..]`.
pub struct DenseTransposedProduct<'a> {
    pub scratch: &'a [f64],
    pub bar: &'a mut [f64],
    pub mat: usize,
    pub vec: usize,
    pub dst: usize,
    pub rows: usize,
    pub cols: usize,
}

impl Product for DenseTransposedProduct<'_> {
    type Output = ();

    #[inline(always)]
    fn run<A: Accumulate>(self) {
        let Self {
            scratch,
            bar,
            mat,
            vec,
            dst,
            rows,
            cols,
        } = self;
        for row in 0..rows {
            let bd = bar[dst + row];
            if bd == 0.0 {
                continue;
            }
            let base = mat + row * cols;
            for j in 0..cols {
                bar[vec + j] = A::mul_acc(bar[vec + j], scratch[base + j], bd);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::node::{CsrData, Shape};

    #[test]
    #[allow(clippy::float_cmp)]
    fn matvec_takes_each_rows_dot_product() {
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
            csr.matvec(&x, &mut y);
            // Small integers: every product and sum is exact.
            assert_eq!(y, expected);
        }
    }
}
