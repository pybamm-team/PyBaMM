from __future__ import annotations

from dataclasses import dataclass

import casadi
import numpy as np
from scipy.sparse import issparse


def try_repeated_row_matmul(
    left_symbol, converted_right: casadi.MX
) -> casadi.MX | None:
    """
    Optimize M @ x when M has repeated rows.

    Detects two patterns:
    - MatMulIdenticalRows: All rows are the same
    - MatMulBoundaryDiffers: Interior rows identical, edges differ

    This reduces O(m*n) to O(m+n) for the repeated section.
    """
    entries = getattr(left_symbol, "entries", None)
    if entries is None:
        return None

    m = entries.shape[0]

    if m < 2:
        return None

    if issparse(entries):
        opt = _check_identical_rows_sparse(entries.tocsr())
    else:
        opt = _check_identical_rows_dense(np.asarray(entries))

    if opt:
        return opt.apply(converted_right)

    return None


@dataclass
class MatMulIdenticalRows:
    """All rows are identical: M @ x = ones(m,1) * (row @ x)"""

    row: np.ndarray
    m: int

    def apply(self, converted_right: casadi.MX) -> casadi.MX:
        return casadi.DM.ones(self.m, 1) * (casadi.DM(self.row).T @ converted_right)


@dataclass
class MatMulBoundaryDiffers:
    """Interior rows identical, boundary rows differ."""

    interior_row: np.ndarray
    first_row: np.ndarray | None  # None if same as interior
    last_row: np.ndarray | None  # None if same as interior
    m: int

    def apply(self, converted_right: casadi.MX) -> casadi.MX:
        first = self.first_row if self.first_row is not None else self.interior_row
        last = self.last_row if self.last_row is not None else self.interior_row

        first_result = casadi.DM(first).T @ converted_right
        interior_result = casadi.DM.ones(self.m - 2, 1) * (
            casadi.DM(self.interior_row).T @ converted_right
        )
        last_result = casadi.DM(last).T @ converted_right
        return casadi.vertcat(first_result, interior_result, last_result)


def _csr_rows_equal(indptr, indices, data, i, j):
    """Check if rows i and j of a CSR matrix are identical."""
    si, ei = indptr[i], indptr[i + 1]
    sj, ej = indptr[j], indptr[j + 1]
    if ei - si != ej - sj:
        return False
    return np.array_equal(indices[si:ei], indices[sj:ej]) and np.array_equal(
        data[si:ei], data[sj:ej]
    )


def _csr_row_to_dense(csr, i, n):
    """Extract a single row from CSR as a dense 1D array."""
    row = np.zeros(n)
    s, e = csr.indptr[i], csr.indptr[i + 1]
    row[csr.indices[s:e]] = csr.data[s:e]
    return row


def _check_identical_rows_sparse(
    csr,
) -> MatMulIdenticalRows | MatMulBoundaryDiffers | None:
    """Check if interior rows are identical, operating directly on CSR data."""
    m, n = csr.shape
    indptr, indices, data = csr.indptr, csr.indices, csr.data

    row_nnz = np.diff(indptr)
    interior_nnz = row_nnz[1]

    if not (row_nnz[1:-1] == interior_nnz).all():
        return None

    if m == 2:
        if _csr_rows_equal(indptr, indices, data, 0, 1):
            return MatMulIdenticalRows(row=_csr_row_to_dense(csr, 0, n), m=m)
        return None

    for i in range(2, m - 1):
        if not _csr_rows_equal(indptr, indices, data, 1, i):
            return None

    first_same = _csr_rows_equal(indptr, indices, data, 0, 1)
    last_same = _csr_rows_equal(indptr, indices, data, m - 1, 1)

    interior_row = _csr_row_to_dense(csr, 1, n)
    if first_same and last_same:
        return MatMulIdenticalRows(row=interior_row, m=m)
    else:
        return MatMulBoundaryDiffers(
            interior_row=interior_row,
            first_row=None if first_same else _csr_row_to_dense(csr, 0, n),
            last_row=None if last_same else _csr_row_to_dense(csr, m - 1, n),
            m=m,
        )


def _check_identical_rows_dense(
    dense: np.ndarray,
) -> MatMulIdenticalRows | MatMulBoundaryDiffers | None:
    """Check if interior rows are identical, with optional boundary differences."""
    first_row = dense[0, :]
    m = dense.shape[0]

    if m == 2:
        if np.array_equal(dense[1, :], first_row):
            return MatMulIdenticalRows(row=first_row, m=m)
        return None

    interior_row = dense[1, :]

    if not (dense[2:-1] == interior_row).all():
        return None

    last_row = dense[-1, :]
    first_same = np.array_equal(first_row, interior_row)
    last_same = np.array_equal(last_row, interior_row)

    if first_same and last_same:
        return MatMulIdenticalRows(row=interior_row, m=m)
    else:
        return MatMulBoundaryDiffers(
            interior_row=interior_row,
            first_row=None if first_same else first_row,
            last_row=None if last_same else last_row,
            m=m,
        )
