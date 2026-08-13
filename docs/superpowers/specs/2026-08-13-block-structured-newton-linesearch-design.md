# Block-structured Newton linesearch for the algebraic IC solver

Date: 2026-08-13
Component: `packages/pybammsolvers/src/pybammsolvers/idaklu_source`

## Problem

`NonlinearSolver` damps the whole Newton step with a single scalar `alpha`. When one
badly-scaled equation rejects the Armijo test, every state is damped with it, even
states whose residual was already decreasing.

This is not hypothetical. Running CasADi's `scc()` on `jac_algebraic_eval` (algebraic
columns, square) gives:

| Model | `n_alg` | SCC blocks | Structure |
| --- | --- | --- | --- |
| DFN | 100 | 1 | irreducible |
| DFN, 2-phase negative | 40 | 2 | 20 + 20, no coupling |
| DFN + SEI | 140 | 21 | 120 core, then 20 downstream scalars |
| DFN + SEI + plating | 140 | 21 | same |
| DFN + lumped thermal + SEI | 140 | 21 | same |

For the degradation models the 20 singletons are
`Total negative electrode interfacial current density variable [A.m-2]`, each an
exponential function of an overpotential and each strictly downstream of the
120-variable potential core. One of those scalars backtracking to `alpha = 2^-8`
currently forces the other 139 states to `2^-8` as well.

Plain DFN has no exploitable structure and must be unaffected.

## Goals

- Per-block damping, so an ill-scaled equation damps only its own block.
- Solve the block DAG in dependency order, fully converging each level before the next.
- Never worse than today: `COUPLED` mode is bit-identical to the current code, and a
  one-block partition degrades to `COUPLED` automatically.

## Non-goals

- IDA's time-stepping nonlinear solver. That needs a custom `SUNNonlinearSolver` and is
  a separate change.
- `StandaloneNewtonSolver`. It gets the capability for free once the partition lives in
  `NonlinearSolver`, but nothing is wired up here.
- Numerical (as opposed to structural) block detection.

## Design

### The setup object

One immutable struct fully defines all three modes. It is built once, at solver setup.

```cpp
enum class BlockMode { COUPLED, DECOUPLED, STAGGERED };

struct BlockPartition {
  BlockMode mode = BlockMode::COUPLED;
  std::vector<int> block_of;              // n_vars entries; -1 means frozen for all levels
  std::vector<std::vector<int>> blocks;   // state indices owned by each block
  std::vector<int> block_level;           // topological depth of each block
  std::vector<std::vector<int>> levels;   // block ids per level, in solve order
};
```

`blocks` is the vector of index arrays, the solver's `alpha_` array is the vector of
delay terms (one per block), and `levels` is the order. A block whose level is not the
active one has its delay term held at zero, which freezes those states.

### Building the partition

1. Obtain the square algebraic Jacobian sparsity as CSC.
   - `SUBBLOCK` mode: `PrecomputeSubBlockSparsity` already produces exactly this in
     `sb.colptrs` / `sb.rowvals`.
   - `FULL` mode: filter the algebraic rows and columns out of
     `jac_times_cjmass_colptrs` / `jac_times_cjmass_rowvals`. PyBaMM lays states out as
     `[rhs, alg]`, so the algebraic indices are the contiguous range
     `[len_rhs_, len_rhs_ + len_alg_)`.
2. `casadi::Sparsity(n, n, colptrs, rowvals).scc(index, offset)` gives the strongly
   connected components.
3. Build the block DAG from the off-diagonal structure and assign each block a level by
   longest path from a source. Do not assume a direction from `scc`'s output ordering:
   it returns block-upper-triangular ordering, which is the reverse of dependency order.
   Deriving levels from the DAG directly is convention-independent.
4. Choose the mode:
   - `COUPLED` — one block holding every non-frozen index, one level.
   - `DECOUPLED` — blocks are the weakly-connected components of the DAG, all at level 0.
     Only components are provably independent; two SCCs joined by a triangular edge are
     not, so per-block Armijo would have no descent guarantee for them.
   - `STAGGERED` — blocks are the SCCs, levels are DAG depth. On genuinely independent
     components it reproduces `DECOUPLED`: two single-SCC components both land at level 0
     and are solved simultaneously with separate delay terms. It differs only where a
     component splits into several triangularly coupled SCCs, which `DECOUPLED` keeps
     fused into one block.
   - `auto` — `STAGGERED` if more than one level, else `DECOUPLED` if more than one
     component, else `COUPLED`.
5. Fall back to `COUPLED` when the structure is unusable: structurally singular
   (`sprank < n`), a single SCC, or an unsupported mode.

Cost is `O(nnz)` once at setup.

### Freezing replaces `diff_idx`

In `FULL` mode the differential states are permanently frozen, which is exactly
`block_of[i] == -1`. `diff_idx_` and `ZeroDiffComponents` are removed in favour of the
partition, so both IC modes run the same loop.

### The Newton loop

```
for level in partition.levels:
  active = blocks in level
  for iter < max_iter:
    eval_residual                              # full vector
    zero res outside active                    # frozen + other levels
    solve_linear                               # the existing single full solve
    zero delta outside active
    per-block delnorm_b, res_norm_b
    if all blocks converged: break to next level
    save iterate
    per-block Armijo
```

Zeroing the residual rows of inactive blocks makes the existing full linear solve return
the exact block Gauss-Seidel step. The permuted Jacobian is block lower triangular, so
forward substitution over the zeroed rows of already-converged blocks yields `delta = 0`
for each of them; the active block then satisfies `delta_a = J_aa^-1 * res_a` exactly.
No submatrix extraction, no additional factorisation, no extra `SUNLinearSolver`.

Downstream blocks (later levels) receive a nonzero `delta` from the coupling term and are
masked explicitly.

This exactness argument holds in `SUBBLOCK` mode. In `FULL` mode the solve is over all
`n_states` with the differential rows zeroed, so `delta_d` is nonzero before masking and
the algebraic step is already an approximation in the current code. Block masking
inherits that same approximation; staggering is a heuristic there, not an identity. The
residual-improvement assertion below is what guards `FULL` mode.

### Per-block Armijo with shared residual evaluations

```
alpha_b = 1 for every active block
for ls < max_backtracks:
  x[i] = x_save[i] - alpha_[block_of[i]] * delta[i]      # frozen i untouched
  eval_residual                                          # one evaluation scores all blocks
  for each active b failing  ||res_b||_inf <= (1 - 0.5*alpha_b) * res_norm_b:
      alpha_b *= 0.5
  if nothing failed: break
```

Blocks within a level are mutually independent once lower levels are frozen, so a single
residual evaluation gives every block's trial norm. The evaluation count per iteration is
therefore identical to today's scalar Armijo. The existing `alpha * delnorm <= step_tol`
early exit becomes per-block.

### Convergence test

Per-block WRMS keeps the global `n_vars_` denominator, so the `COUPLED` formula is
unchanged. The per-block threshold is `epsNewt / sqrt(|active|)`, which gives

```
sum over active blocks of delnorm_b^2  <=  epsNewt^2
```

This is exactly today's global test when there is one active block, and is never looser
than it. A level is done when every one of its blocks passes.

Failure behaviour is unchanged: a level that exhausts `max_iter` returns the same
`NonlinearResult` the current solver returns, with no coupled retry.

### Iteration budget

`max_num_iterations_ic` applies per level. Worst case is `n_levels` times the current
work; `n_levels` is 2 on every model measured. This matches "fully solve each subsystem
before moving on".

## Verification

- **Bit-identical `COUPLED`.** A test asserts the full iterate trace under `COUPLED`
  matches the pre-change solver byte for byte, and that a one-block partition selects
  `COUPLED` automatically.
- **Residual improvement assertion.** After each accepted linesearch, every active
  block's residual inf-norm must be no larger than it was before the step. Checked in
  debug builds and recorded through `SolverLog`.
- **Synthetic structures.** A two-block independent system (per-block `alpha` diverge,
  same or fewer iterations than coupled) and a two-level triangular system (staggered
  reproduces the exact block Gauss-Seidel iterates).
- **Real models.** DFN 2-phase and DFN+SEI: Newton iterations and IC failures no worse
  than `COUPLED`, per-block residuals monotone.

## Interface

New `SolverOptions` field `newton_block_mode`, one of `"coupled"`, `"decoupled"`,
`"staggered"`, `"auto"`, defaulting to `"coupled"`. Plumbed from Python the same way
`newton_mode` is, defaulted in `idaklu_solver.py`.

## Files

| File | Change |
| --- | --- |
| `idaklu_source/BlockPartition.hpp` | New. CasADi `scc`, DAG levels, mode selection. Keeps `NonlinearSolver` free of CasADi. |
| `idaklu_source/NonlinearSolver.hpp` | Hold a `BlockPartition` and `alpha_`; drop `diff_idx_`. |
| `idaklu_source/NonlinearSolver.inl` | Masked residual and step, level loop, per-block Armijo and convergence test. |
| `idaklu_source/AlgebraicICBuilder.inl` | Build the square CSC for both IC modes, construct the partition, pass it in. |
| `idaklu_source/Options.hpp`, `Options.cpp` | `newton_block_mode`. |
| `pybamm/solvers/idaklu_solver.py` | Option default. |
| tests, `CHANGELOG.md` | As above. |

## Deferred

Levels made entirely of 1x1 blocks still pay for a full KLU solve per iteration when a
scalar division would do. For DFN+SEI that is roughly three extra 140x140 solves per IC
call, negligible against a DFN step. Marked with a `ponytail:` comment; revisit only if
profiling says otherwise.
