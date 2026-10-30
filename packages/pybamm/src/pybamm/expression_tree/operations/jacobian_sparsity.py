#
# Structural Jacobian sparsity of a discretised expression, and a CasADi Jacobian
# assembled from it
#
from __future__ import annotations

import casadi
import numpy as np
import scipy.sparse

import pybamm


class JacobianSparsity:
    """
    Structural sparsity pattern of the Jacobian of discretised expressions with
    respect to the state vector.

    The pattern is propagated up the expression tree without differentiating, by
    each node's :meth:`pybamm.Symbol._jacobian_sparsity` rule. Patterns of
    subexpressions are cached, so expressions that share them are cheap.

    Parameters
    ----------
    n_states : int
        The length of the state vector.
    """

    def __init__(self, n_states: int):
        self.n_states = n_states
        # Symbol id -> (size, pattern), with None for no dependence on the states
        self._known: dict[int, tuple[int, scipy.sparse.csr_matrix | None]] = {}

    def pattern(self, symbol: pybamm.Symbol) -> scipy.sparse.csr_matrix:
        """
        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The discretised expression.

        Returns
        -------
        :class:`scipy.sparse.csr_matrix`
            Boolean matrix of shape ``(symbol.size, n_states)``, true where the
            Jacobian can be nonzero.

        Raises
        ------
        NotImplementedError
            If the expression contains a node with no structural rule.
        """
        size, pattern = self.size_and_pattern(symbol)
        if pattern is None:
            return scipy.sparse.csr_matrix((size, self.n_states), dtype=bool)
        return pattern

    def size_and_pattern(self, symbol: pybamm.Symbol):
        """
        The size of ``symbol`` and its pattern, None for no dependence on the
        states. Sizes come from the children, since ``Symbol.size`` evaluates the
        whole subtree.
        """
        if symbol.id not in self._known:
            self._known[symbol.id] = symbol._jacobian_sparsity(self)
        return self._known[symbol.id]

    def union(self, children, size: int | None = None):
        """
        Size and pattern of a node that depends on every entry its arguments'
        patterns reach: elementwise for ``size`` None, with scalars broadcast.
        """
        sized = [self.size_and_pattern(child) for child in children]
        if size is None:
            size = max(child_size for child_size, _ in sized)
        result = None
        seen = set()
        for child_size, pattern in sized:
            # Arguments often share a pattern, as in x * f(x)
            if pattern is None or id(pattern) in seen:
                continue
            seen.add(id(pattern))
            if child_size == size:
                term = pattern
            elif child_size == 1:
                # A scalar argument reaches every entry
                ones = scipy.sparse.csr_matrix(np.ones((size, 1), dtype=bool))
                term = ones @ pattern
            elif size == 1:
                # A reduction depends on every entry of its argument
                term = scipy.sparse.csr_matrix(pattern.sum(axis=0) > 0)
            else:
                raise NotImplementedError(
                    f"Cannot combine arguments of sizes {child_size} and {size}"
                )
            result = (
                term if result is None else (result + term).astype(bool, copy=False)
            )
        return size, result

    def stack(self, children):
        """Size and pattern of the children stacked vertically."""
        sized = [self.size_and_pattern(child) for child in children]
        size = sum(child_size for child_size, _ in sized)
        if all(pattern is None for _, pattern in sized):
            return size, None
        return size, scipy.sparse.vstack(
            [self.pattern(child) for child in children], format="csr"
        )

    def independent(self, children):
        """Size of an elementwise node of ``children`` whose Jacobian is zero."""
        return max(self.size_and_pattern(child)[0] for child in children), None


def casadi_jacobian(
    expression: casadi.MX,
    y: casadi.MX,
    sparsity: scipy.sparse.spmatrix,
    entries: np.ndarray | None = None,
) -> casadi.MX:
    """
    Jacobian of a CasADi expression with a known sparsity pattern.

    The rows are split between forward and reverse mode (bidirectional colouring).
    Rows that depend on many states, such as a mean over a field, are
    differentiated in reverse mode, one sweep per colour, and the rest in forward
    mode with columns grouped by colour, where a state that every row depends on
    costs a single colour. Only the columns of the requested entries are seeded,
    so entries outside them, such as dense couplings that are not needed, cost
    nothing.

    Parameters
    ----------
    expression : :class:`casadi.MX`
        The expression, a column vector.
    y : :class:`casadi.MX`
        The state vector symbol.
    sparsity : :class:`scipy.sparse.spmatrix`
        Structural sparsity pattern of the Jacobian, such as from
        :class:`JacobianSparsity`.
    entries : numpy.ndarray, optional
        Positions, in the CSC data of ``sparsity``, of the entries to compute; the
        rest are zero. Default is every entry.

    Returns
    -------
    :class:`casadi.MX`
        The Jacobian, with the given sparsity.
    """
    jacobian = scipy.sparse.csc_matrix(sparsity, dtype=bool)
    jacobian.sum_duplicates()
    jacobian.sort_indices()
    n_rows, n_columns = jacobian.shape
    coo = jacobian.tocoo()
    rows, columns = coo.row, coo.col
    wanted = np.ones(len(rows), dtype=bool)
    if entries is not None:
        wanted[:] = False
        wanted[entries] = True

    # Colour only the rows and columns of wanted entries; the rest never mix in
    needed_rows = np.zeros(n_rows, dtype=bool)
    needed_rows[rows[wanted]] = True
    seeded = np.zeros(n_columns, dtype=bool)
    seeded[columns[wanted]] = True
    used = np.flatnonzero(seeded)
    local = np.full(n_columns, -1, dtype=np.int64)
    local[used] = np.arange(len(used))
    keep = needed_rows[rows] & seeded[columns]
    resolved = scipy.sparse.csr_matrix(
        (np.ones(keep.sum(), dtype=bool), (rows[keep], local[columns[keep]])),
        shape=(n_rows, len(used)),
    )
    reverse = _reverse_rows(resolved)
    forward_colour, n_forward = _colour_columns(_keep_rows(resolved, ~reverse))
    reverse_colour, n_reverse = _colour_columns(_keep_rows(resolved, reverse).T)
    reverse_colour[~reverse] = -1

    sparsity_casadi = _casadi_sparsity(jacobian)
    if n_forward + n_reverse == 0:
        return casadi.MX(sparsity_casadi, 0)
    resolved = resolved.tocoo()
    in_forward = ~reverse[resolved.row]
    blocks, offset = [], 0
    position = np.full(len(rows), -1, dtype=np.int64)
    if n_forward:
        # Rows x colours, with column j of the Jacobian in colour forward_colour
        coloured = forward_colour >= 0
        seeds = casadi.Sparsity.triplet(
            n_columns,
            n_forward,
            used[coloured].tolist(),
            forward_colour[coloured].tolist(),
        )
        product, keys = _product_pattern(
            resolved.row[in_forward],
            forward_colour[resolved.col[in_forward]],
            n_rows,
            n_forward,
        )
        blocks.append(
            casadi.vec(
                casadi.project(
                    casadi.jtimes(expression, y, casadi.MX(casadi.DM(seeds, 1))),
                    product,
                )
            )
        )
        take = wanted & ~reverse[rows]
        position[take] = offset + np.searchsorted(
            keys, forward_colour[local[columns[take]]] * n_rows + rows[take]
        )
        offset += product.nnz()
    if n_reverse:
        # Columns x colours, with row i of the Jacobian in colour reverse_colour
        reverse_rows = np.flatnonzero(reverse_colour >= 0)
        seeds = casadi.Sparsity.triplet(
            n_rows,
            n_reverse,
            reverse_rows.tolist(),
            reverse_colour[reverse_rows].tolist(),
        )
        product, keys = _product_pattern(
            used[resolved.col[~in_forward]],
            reverse_colour[resolved.row[~in_forward]],
            n_columns,
            n_reverse,
        )
        blocks.append(
            casadi.vec(
                casadi.project(
                    casadi.jtimes(expression, y, casadi.MX(casadi.DM(seeds, 1)), True),
                    product,
                )
            )
        )
        take = wanted & reverse[rows]
        position[take] = offset + np.searchsorted(
            keys, reverse_colour[rows[take]] * n_columns + columns[take]
        )
    # Unwanted entries read a trailing zero
    zero = casadi.MX(casadi.DM.zeros(1, 1))
    values = casadi.vertcat(*blocks, zero)
    position[position < 0] = values.nnz() - 1
    return casadi.MX(sparsity_casadi, values.nz[position.tolist()])


def _reverse_rows(pattern: scipy.sparse.csr_matrix) -> np.ndarray:
    """
    Rows to differentiate in reverse mode: the ``k`` densest rows, with ``k``
    minimising ``k`` reverse sweeps plus the nonzeros of the densest remaining row,
    which forward mode needs at least as many colours as.
    """
    row_counts = np.diff(pattern.indptr)
    order = np.argsort(-row_counts, kind="stable")
    densest_remaining = np.append(row_counts[order], 0)
    k = int(np.argmin(np.arange(len(densest_remaining)) + densest_remaining))
    reverse = np.zeros(len(row_counts), dtype=bool)
    reverse[order[:k]] = True
    return reverse


def _keep_rows(pattern: scipy.sparse.csr_matrix, keep: np.ndarray):
    coo = pattern.tocoo()
    kept = keep[coo.row]
    return scipy.sparse.csr_matrix(
        (coo.data[kept], (coo.row[kept], coo.col[kept])), shape=pattern.shape
    )


def _colour_columns(pattern) -> tuple[np.ndarray, int]:
    """Colour of each column, such that no row has two entries of one colour, and
    the number of colours. Columns without entries take no colour (-1)."""
    pattern = scipy.sparse.csc_matrix(pattern)
    colour = np.full(pattern.shape[1], -1, dtype=np.int64)
    if pattern.nnz == 0:
        return colour, 0
    colouring = _casadi_sparsity(pattern).uni_coloring()
    index, assigned = (np.asarray(a, dtype=np.int64) for a in colouring.get_triplet())
    colour[index] = assigned
    colour[np.diff(pattern.indptr) == 0] = -1
    occupied = np.flatnonzero(np.bincount(colour[colour >= 0]))
    # Renumber so the colours in use are 0, 1, ...
    renumber = np.full(colouring.size2(), -1, dtype=np.int64)
    renumber[occupied] = np.arange(len(occupied))
    colour[colour >= 0] = renumber[colour[colour >= 0]]
    return colour, len(occupied)


def _product_pattern(rows, colours, n_rows, n_colours):
    """CasADi pattern of a compressed product with the given entries, and the key
    ``colour * n_rows + row`` of each of its nonzeros, in order. A colouring puts no
    two entries of one row in one colour, so the keys are distinct."""
    keys = np.sort(colours.astype(np.int64) * n_rows + rows)
    product = casadi.Sparsity.triplet(
        n_rows, n_colours, (keys % n_rows).tolist(), (keys // n_rows).tolist()
    )
    return product, keys


def _casadi_sparsity(pattern) -> casadi.Sparsity:
    csc = scipy.sparse.csc_matrix(pattern)
    csc.eliminate_zeros()
    csc.sort_indices()
    return casadi.Sparsity(
        csc.shape[0], csc.shape[1], csc.indptr.tolist(), csc.indices.tolist()
    )
