from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from pybammsolvers import idaklu
from scipy.sparse import csc_matrix

from pybamm.solvers.preconditioners.tree import Node, specification


def analyse(tree: Node, jacobian_pattern: csc_matrix) -> Any:
    """
    Analyse a preconditioner tree for the block preconditioner backend.

    The backend computes every block's pattern, the Schur complements' structure
    and the incomplete factorisations' fill from the tree and the pattern.

    Parameters
    ----------
    tree : :class:`Leaf`, :class:`Additive`, :class:`Multiplicative` or :class:`Schur`
        The tree, from :func:`pybamm.preconditioners.compile_tree`.
    jacobian_pattern : :class:`scipy.sparse.csc_matrix`
        Pattern of the iteration matrix, in the CSC order of the data the solver
        sets the preconditioner up from.

    Returns
    -------
    :class:`pybammsolvers.idaklu.BlockAnalysis`
        The analysis. Its ``exact`` attribute says whether applying the tree
        solves the linear system exactly, and ``read_entries`` gives the positions
        of the pattern's CSC data that the tree reads.
    """
    pattern = csc_matrix(jacobian_pattern)
    return idaklu.BlockAnalysis(
        specification(tree),
        pattern.indptr.astype(np.int64),
        pattern.indices.astype(np.int64),
    )


class JacobianAnalysis(NamedTuple):
    """A tree analysed on an iteration matrix, from :func:`analyse_jacobian`."""

    #: The analysis, from :func:`analyse`
    analysis: Any
    #: Pattern of the iteration matrix ``J - cj * M``
    pattern: csc_matrix
    #: Sorted positions, in the CSC data of ``J``, of the entries the tree reads
    entries: npt.NDArray[np.int64]


def analyse_jacobian(
    tree: Node, jacobian_pattern: csc_matrix, mass_matrix: csc_matrix
) -> JacobianAnalysis:
    """
    Analyse a preconditioner tree on the iteration matrix ``J - cj * M``, and find
    the entries of ``J`` it reads.

    Parameters
    ----------
    tree : :class:`Leaf`, :class:`Additive`, :class:`Multiplicative` or :class:`Schur`
        The tree, from :func:`pybamm.preconditioners.compile_tree`.
    jacobian_pattern : :class:`scipy.sparse.spmatrix`
        Pattern of the Jacobian ``J``.
    mass_matrix : :class:`scipy.sparse.spmatrix`
        The mass matrix ``M``.

    Returns
    -------
    :class:`JacobianAnalysis`
    """
    jacobian = csc_matrix(jacobian_pattern, dtype=bool)
    jacobian.sort_indices()
    iteration = csc_matrix(jacobian + (csc_matrix(mass_matrix) != 0))
    iteration.sort_indices()
    analysis = analyse(tree, iteration)
    read = np.zeros(iteration.nnz)
    read[analysis.read_entries] = 1.0
    read = csc_matrix((read, iteration.indices, iteration.indptr), iteration.shape)
    # Positions in the Jacobian's CSC data, offset by one so none are zero
    position = csc_matrix(
        (np.arange(1, jacobian.nnz + 1), jacobian.indices, jacobian.indptr),
        jacobian.shape,
    )
    entries = np.sort(position.multiply(read).tocsc().data).astype(np.int64) - 1
    return JacobianAnalysis(analysis, iteration, entries)
