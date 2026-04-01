import numpy as np

from pybammsolvers import idaklu


class LinearSolver:
    """Thin Python wrapper around the C++ LinearSolver.

    Wraps SUNDIALS direct/iterative linear solvers via pybammsolvers.
    Supports factorize/solve for direct solvers and iterative solve.

    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse matrix
        Sparsity pattern (any scipy.sparse format; converted to CSC internally).
    options : dict, optional
        Options passed to the C++ LinearSolver constructor:

        - ``"jacobian"``: ``"sparse"`` (default) or ``"dense"``
        - ``"linear_solver"``: ``"SUNLinSol_KLU"`` (default), ``"SUNLinSol_Dense"``, etc.
    """

    def __init__(self, matrix: np.ndarray, options: dict | None = None):
        options = options or {}
        self._solver = idaklu.create_linear_solver(matrix, options)

    def factorize(self, A: np.ndarray):
        """
        Perform in-place factorization of a square matrix A.

        Parameters
        ----------
        A : numpy array
            Square matrix (nnz for sparse, n*n for dense).
        """
        A = np.asarray(A, dtype=np.float64)
        self._solver.factorize(A)

    def solve(self, b: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Solve A*x = b using existing factorization.

        Parameters
        ----------
        b : numpy array
            Right-hand side vector (length n).
        out : numpy array, optional
            Pre-allocated output array (length n) for in-place result.

        Returns
        -------
        numpy array
            Solution vector x.
        """
        b = np.asarray(b, dtype=np.float64).ravel()
        return self._solver.solve(b, out)

    def solve_batched(self, B: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Solve A@X = B, where B is a matrix (n, k) and X is a matrix (n, k).

        Parameters
        ----------
        B : numpy array
            Right-hand side matrix (n, k).
        out : numpy array, optional
            Pre-allocated output matrix (n, k) for in-place result.

        Returns
        -------
        numpy array
            Solution matrix X (n, k).
        """
        B = np.asarray(B, dtype=np.float64)
        return self._solver.solve_batched(B, out)

    @property
    def can_factorize(self):
        """True for direct solvers, False for iterative."""
        return self._solver.can_factorize
