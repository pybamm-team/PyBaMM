#ifndef PYBAMM_IDAKLU_BLOCK_PRECONDITIONER_HPP
#define PYBAMM_IDAKLU_BLOCK_PRECONDITIONER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

#include "Preconditioner.hpp"
#include "../include/pybamm_block_solver.h"

namespace py = pybind11;

using IndexVector = std::vector<int64_t>;

/**
 * @brief One node of an analysed preconditioner tree. Shared read-only by every
 * solver.
 *
 * A node reads its matrix from a source array: the iteration matrix's CSC data
 * at the root, or the Schur complement owned by the nearest Schur ancestor.
 */
struct BlockNodeData {
  enum Type { LEAF, ADDITIVE, MULTIPLICATIVE, SCHUR };
  enum Solver { KLU, ILU, JACOBI, TRIDIAGONAL, EXTERNAL };
  enum SchurComplement { EXACT, DIAGONAL, NONE };

  Type type = LEAF;
  int64_t size = 0;
  bool exact = false;

  // Leaf: stored pattern (CSC for KLU and external, CSR for ILU) and the
  // position of every stored entry in the source array (-1 for a zero)
  Solver solver = KLU;
  IndexVector indptr, indices, diagonal, gather;
  // External: the plugin's functions and its options as JSON
  pybamm_block_solver external{};
  std::string external_options;

  // Additive, multiplicative and Schur: children and their states' positions
  std::vector<std::shared_ptr<const BlockNodeData>> children;
  std::vector<IndexVector> positions;

  // Multiplicative: entries coupling each child to the children before it
  IndexVector sweep_ptr, sweep_row, sweep_col, sweep_slot;

  // Schur: off-diagonal blocks and the complement's own value array
  SchurComplement schur_complement = EXACT;
  IndexVector m01_row, m01_col, m01_slot, m10_row, m10_col, m10_slot, s_gather;
  // Exact: one right-hand side of M01 columns per colour, then its updates
  IndexVector colour_ptr, colour_row, colour_slot;
  IndexVector update_ptr, update_s, update_m10, update_state;
  // Diagonal: S -= M10[r, i] M01[i, j] / M00[i, i]
  IndexVector diagonal_s, diagonal_m10, diagonal_m01, diagonal_m00;
};

/**
 * @brief Symbolic analysis of a preconditioner tree against the sparsity pattern
 * of the iteration matrix
 */
class BlockAnalysis {
public:
  /**
   * @param tree Nested dictionaries: leaves give their states' indices and
   *     block solver, composite nodes their children
   * @param indptr, indices CSC pattern of the iteration matrix, in the order of
   *     the data the preconditioner is set up from
   */
  BlockAnalysis(const py::dict &tree, const IndexVector &indptr,
                 const IndexVector &indices);

  std::shared_ptr<const BlockNodeData> root;
  int64_t size;
  // Sorted positions of the iteration matrix's data that the tree reads
  IndexVector read_entries;
  bool exact() const { return root->exact; }
};

class BlockNode;

/**
 * @brief The block preconditioner backend: applies an analysed preconditioner tree
 */
class BlockPreconditioner : public Preconditioner {
public:
  explicit BlockPreconditioner(std::shared_ptr<const BlockAnalysis> analysis);
  ~BlockPreconditioner() override;
  BlockPreconditioner(const BlockPreconditioner &) = delete;
  BlockPreconditioner &operator=(const BlockPreconditioner &) = delete;

  int setup(const double *matrix_data) override;
  void solve(const double *r, double *z) override;

private:
  std::shared_ptr<const BlockAnalysis> analysis_;
  std::unique_ptr<BlockNode> root_;
  int64_t size_;
};

#endif // PYBAMM_IDAKLU_BLOCK_PRECONDITIONER_HPP
