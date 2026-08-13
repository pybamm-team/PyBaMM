#ifndef PYBAMM_BLOCK_PARTITION_HPP
#define PYBAMM_BLOCK_PARTITION_HPP

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief How NonlinearSolver damps and orders the Newton step.
 *
 * COUPLED    one block over every solved state, one damping factor (no structural analysis)
 * DECOUPLED  one block per independent subsystem, damped separately, all solved at once
 * STAGGERED  one block per strongly connected component, solved in dependency order
 * AUTO       request only; resolves to DECOUPLED, or COUPLED if there is a single subsystem
 */
enum class BlockMode { COUPLED, DECOUPLED, STAGGERED, AUTO };

inline BlockMode block_mode_from_string(const std::string& name) {
  if (name == "coupled") return BlockMode::COUPLED;
  if (name == "decoupled") return BlockMode::DECOUPLED;
  if (name == "staggered") return BlockMode::STAGGERED;
  if (name == "auto") return BlockMode::AUTO;
  throw std::invalid_argument("Unknown block mode '" + name + "'");
}

inline const char* block_mode_name(BlockMode mode) {
  switch (mode) {
    case BlockMode::COUPLED:   return "coupled";
    case BlockMode::DECOUPLED: return "decoupled";
    case BlockMode::STAGGERED: return "staggered";
    case BlockMode::AUTO:      return "auto";
  }
  return "unknown";
}

/**
 * @brief Which states move together, and in what order.
 *
 * Fully defines the damping structure: `blocks` holds the state indices of each
 * subsystem, one damping factor is carried per block, and `levels` gives the order
 * they are solved in. States absent from every block are frozen for the whole solve.
 */
struct BlockPartition {
  BlockMode mode = BlockMode::COUPLED;
  int n_vars = 0;
  std::vector<int> block_of;              // n_vars entries; -1 means frozen
  std::vector<std::vector<int>> blocks;   // state indices per block
  std::vector<std::vector<int>> levels;   // block ids per level, in solve order

  int n_blocks() const { return static_cast<int>(blocks.size()); }
  int n_levels() const { return static_cast<int>(levels.size()); }

  /** @brief Single block over `solve_idx`, solved in one level. */
  static BlockPartition coupled(int n_vars, const std::vector<int>& solve_idx) {
    BlockPartition p;
    p.mode = BlockMode::COUPLED;
    p.n_vars = n_vars;
    p.block_of.assign(n_vars, -1);
    p.blocks.push_back(solve_idx);
    p.levels.push_back({0});
    for (int i : solve_idx) p.block_of[i] = 0;
    return p;
  }

  /** @brief Every state in a single block. */
  static BlockPartition coupled_all(int n_vars) {
    std::vector<int> all(n_vars);
    for (int i = 0; i < n_vars; i++) all[i] = i;
    return coupled(n_vars, all);
  }
};

#endif // PYBAMM_BLOCK_PARTITION_HPP
