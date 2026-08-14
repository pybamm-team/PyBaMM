#ifndef PYBAMM_BLOCK_PARTITION_HPP
#define PYBAMM_BLOCK_PARTITION_HPP

#include "common.hpp"
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief How NonlinearSolver damps the Newton step.
 *
 * COUPLED    one block over every solved state, one damping factor (no analysis)
 * DECOUPLED  one block per independent subsystem, each damped separately
 */
enum class BlockMode { COUPLED, DECOUPLED };

inline BlockMode block_mode_from_string(const std::string& name) {
  if (name == "coupled") return BlockMode::COUPLED;
  if (name == "decoupled") return BlockMode::DECOUPLED;
  throw std::invalid_argument("Unknown block mode '" + name + "'");
}

inline const char* block_mode_name(BlockMode mode) {
  return mode == BlockMode::DECOUPLED ? "decoupled" : "coupled";
}

/**
 * @brief Which states move together under one damping factor.
 *
 * `blocks` holds the state indices of each independent subsystem and the solver
 * carries one damping factor per block. States absent from every block are frozen
 * for the whole solve.
 */
struct BlockPartition {
  BlockMode mode = BlockMode::COUPLED;
  int n_vars = 0;
  std::vector<int> block_of;              // n_vars entries; -1 means frozen
  std::vector<std::vector<int>> blocks;   // state indices per block

  int n_blocks() const { return static_cast<int>(blocks.size()); }

  /** @brief Single block over `solve_idx`; every other state is frozen. */
  static BlockPartition coupled(int n_vars, const std::vector<int>& solve_idx) {
    BlockPartition part;
    part.n_vars = n_vars;
    part.block_of.assign(n_vars, -1);
    part.blocks.push_back(solve_idx);
    for (int i : solve_idx) part.block_of[i] = 0;
    return part;
  }
};

/**
 * @brief Split the states of a square Jacobian into independent subsystems.
 *
 * Connected components of the structural sparsity, by union-find. Two states share
 * a block whenever some equation reads both, so distinct blocks cannot influence
 * each other and per-block damping is exact. Falls back to a single coupled block
 * when there is nothing to split.
 *
 * @param requested   Mode asked for; COUPLED skips the analysis entirely.
 * @param n_vars      Length of the solver's state vector.
 * @param solve_idx   State indices covered by the Jacobian; all others stay frozen.
 * @param colptrs     CSC column pointers of the square Jacobian, length solve_idx+1.
 * @param rowvals     CSC row indices, local to the Jacobian.
 */
inline BlockPartition build_block_partition(
  BlockMode requested,
  int n_vars,
  const std::vector<int>& solve_idx,
  const std::vector<sunindextype>& colptrs,
  const std::vector<sunindextype>& rowvals)
{
  const int n = static_cast<int>(solve_idx.size());
  if (requested == BlockMode::COUPLED || n <= 1 || rowvals.empty() ||
      static_cast<int>(colptrs.size()) != n + 1) {
    return BlockPartition::coupled(n_vars, solve_idx);
  }

  std::vector<int> parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&parent](int a) {
    while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
    return a;
  };
  for (int c = 0; c < n; c++)
    for (auto k = colptrs[c]; k < colptrs[c + 1]; k++)
      parent[find(static_cast<int>(rowvals[k]))] = find(c);

  std::vector<int> component(n, -1);
  int n_components = 0;
  for (int i = 0; i < n; i++) {
    int root = find(i);
    if (component[root] < 0) component[root] = n_components++;
    component[i] = component[root];
  }
  if (n_components <= 1) return BlockPartition::coupled(n_vars, solve_idx);

  BlockPartition part;
  part.mode = BlockMode::DECOUPLED;
  part.n_vars = n_vars;
  part.block_of.assign(n_vars, -1);
  part.blocks.resize(n_components);
  for (int i = 0; i < n; i++) {
    part.blocks[component[i]].push_back(solve_idx[i]);
    part.block_of[solve_idx[i]] = component[i];
  }
  return part;
}

#endif // PYBAMM_BLOCK_PARTITION_HPP
