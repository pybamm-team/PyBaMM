#ifndef PYBAMM_BLOCK_PARTITION_BUILDER_HPP
#define PYBAMM_BLOCK_PARTITION_BUILDER_HPP

#include "BlockPartition.hpp"
#include "common.hpp"
#include <algorithm>
#include <casadi/casadi.hpp>
#include <numeric>
#include <vector>

/**
 * @brief Group the states of a square Jacobian into independently damped blocks.
 *
 * Runs CasADi's strongly-connected-component decomposition on the structural
 * sparsity, derives the block dependency DAG, and resolves `requested` against it.
 * Falls back to a single coupled block whenever no usable structure is found, so the
 * result is never worse than undamped-per-block behaviour.
 *
 * @param requested   Mode asked for; AUTO resolves to DECOUPLED or COUPLED.
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

  std::vector<casadi_int> colind(colptrs.begin(), colptrs.end());
  std::vector<casadi_int> row(rowvals.begin(), rowvals.end());
  for (int c = 0; c < n; c++)
    std::sort(row.begin() + colind[c], row.begin() + colind[c + 1]);

  std::vector<casadi_int> index, offset;
  casadi_int nb = casadi::Sparsity(n, n, colind, row).scc(index, offset);
  if (nb <= 1) return BlockPartition::coupled(n_vars, solve_idx);

  std::vector<int> block_id(n, 0);
  for (casadi_int b = 0; b < nb; b++)
    for (casadi_int k = offset[b]; k < offset[b + 1]; k++)
      block_id[index[k]] = static_cast<int>(b);

  // A structural entry (r, c) means equation r reads variable c, so the block
  // owning r must be solved after the block owning c.
  const int n_blocks = static_cast<int>(nb);
  std::vector<std::vector<int>> succ(n_blocks);
  for (int c = 0; c < n; c++) {
    for (casadi_int k = colind[c]; k < colind[c + 1]; k++) {
      int b_from = block_id[c], b_to = block_id[row[k]];
      if (b_from != b_to) succ[b_from].push_back(b_to);
    }
  }
  std::vector<int> indeg(n_blocks, 0);
  for (auto& s : succ) {
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
    for (int b : s) indeg[b]++;
  }

  std::vector<int> parent(n_blocks);
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&parent](int a) {
    while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
    return a;
  };
  for (int b = 0; b < n_blocks; b++)
    for (int s : succ[b]) parent[find(b)] = find(s);

  std::vector<int> component(n_blocks);
  int n_components = 0;
  {
    std::vector<int> label(n_blocks, -1);
    for (int b = 0; b < n_blocks; b++) {
      int root = find(b);
      if (label[root] < 0) label[root] = n_components++;
      component[b] = label[root];
    }
  }

  BlockMode mode = requested;
  if (mode == BlockMode::AUTO)
    mode = (n_components > 1) ? BlockMode::DECOUPLED : BlockMode::COUPLED;
  if (mode == BlockMode::COUPLED)
    return BlockPartition::coupled(n_vars, solve_idx);

  BlockPartition part;
  part.n_vars = n_vars;
  part.block_of.assign(n_vars, -1);

  if (mode == BlockMode::DECOUPLED) {
    part.blocks.resize(n_components);
    for (int i = 0; i < n; i++)
      part.blocks[component[block_id[i]]].push_back(solve_idx[i]);
    std::vector<int> all(n_components);
    std::iota(all.begin(), all.end(), 0);
    part.levels.push_back(std::move(all));
  } else {
    std::vector<int> level(n_blocks, 0), queue;
    for (int b = 0; b < n_blocks; b++)
      if (indeg[b] == 0) queue.push_back(b);
    for (size_t head = 0; head < queue.size(); head++) {
      int b = queue[head];
      for (int s : succ[b]) {
        level[s] = std::max(level[s], level[b] + 1);
        if (--indeg[s] == 0) queue.push_back(s);
      }
    }
    if (static_cast<int>(queue.size()) != n_blocks)  // cyclic DAG: cannot order
      return BlockPartition::coupled(n_vars, solve_idx);

    part.blocks.resize(n_blocks);
    for (int i = 0; i < n; i++)
      part.blocks[block_id[i]].push_back(solve_idx[i]);
    part.levels.resize(*std::max_element(level.begin(), level.end()) + 1);
    for (int b = 0; b < n_blocks; b++)
      part.levels[level[b]].push_back(b);
  }

  for (int b = 0; b < part.n_blocks(); b++) {
    std::sort(part.blocks[b].begin(), part.blocks[b].end());
    for (int i : part.blocks[b]) part.block_of[i] = b;
  }
  part.mode = (part.n_blocks() == 1 && part.n_levels() == 1)
    ? BlockMode::COUPLED : mode;
  return part;
}

#endif // PYBAMM_BLOCK_PARTITION_BUILDER_HPP
