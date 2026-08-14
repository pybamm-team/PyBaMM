# Per-block Newton linesearch damping for the algebraic solver

Date: 2026-08-13
Component: `packages/pybammsolvers/src/pybammsolvers/idaklu_source`

## Problem

`NonlinearSolver` damps the whole Newton step with a single scalar `alpha`. When one
badly-scaled equation rejects the Armijo test, every state is damped with it, even
states whose residual was already decreasing.

## Goals

- Per-block damping, so an ill-scaled equation damps only its own subsystem.
- Never worse than today: `coupled` is bit-identical to the current code, and a
  single-subsystem partition degrades to `coupled` automatically.

## Non-goals

- Solving subsystems in dependency order. A staggered mode was built and measured; it
  reached the same residual everywhere but cost 15-50% more iterations and lost 12 of
  78 composite-electrode ESOH solves. Removed. See **What staggering cost** below.
- IDA's time-stepping nonlinear solver. That needs a custom `SUNNonlinearSolver`.
- Numerical (as opposed to structural) block detection.

## Design

### The setup object

One immutable struct, built once at solver setup.

```cpp
enum class BlockMode { COUPLED, DECOUPLED };

struct BlockPartition {
  BlockMode mode = BlockMode::COUPLED;
  int n_vars = 0;
  std::vector<int> block_of;              // n_vars entries; -1 means frozen
  std::vector<std::vector<int>> blocks;   // state indices per block
};
```

`blocks` is the vector of index arrays; the solver's `alpha_` array is the matching
vector of damping factors.

### Building the partition

1. Obtain the square algebraic Jacobian sparsity as CSC.
   - `SUBBLOCK` mode: `PrecomputeSubBlockSparsity` already produces exactly this.
   - `FULL` mode: filter the algebraic rows and columns out of
     `jac_times_cjmass_colptrs` / `jac_times_cjmass_rowvals`. PyBaMM lays states out as
     `[rhs, alg]`, so the algebraic indices are the contiguous range
     `[len_rhs_, len_rhs_ + len_alg_)`.
2. Union-find over the structural entries: two states share a block whenever some
   equation reads both. The result is the connected components of the coupling graph,
   the only partition for which per-block damping is exact — distinct blocks cannot
   influence each other, so each block's Armijo test is a genuine descent test for that
   block's residual.
3. Fall back to a single coupled block when there is nothing to split: `coupled`
   requested, one component, one state, or an empty Jacobian.

Cost is `O(nnz)` once at setup, and `coupled` skips it entirely.

### Freezing replaces `diff_idx`

In `FULL` mode the differential states are permanently frozen, which is exactly
`block_of[i] == -1`. `diff_idx_` and `ZeroDiffComponents` are gone, so both IC modes run
the same loop.

### The Newton loop

```
active = every block
for iter < max_iter:
  eval_residual                              # full vector, masked to active blocks
  solve_linear                               # the existing single full solve
  mask delta to active blocks
  per-block delnorm_b, res_norm_b
  classify each active block: finish on an undamped step, finish by rolling back a
    step that stopped reducing its residual, or keep searching
  per-block Armijo over the blocks still searching
```

Masking the residual and step of a retired block keeps it out of every subsequent
evaluation and linear solve, so it cannot be disturbed by the blocks still running.
Since blocks are independent components, the masked full solve returns exactly the
per-block Newton step.

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

Blocks are independent, so a single residual evaluation gives every block's trial norm.
The evaluation count per iteration is identical to today's scalar Armijo. The existing
`alpha * delnorm <= step_tol` early exit becomes per-block.

### Convergence test

Per-block WRMS keeps the global `n_vars_` denominator, so the `coupled` formula is
unchanged. The per-block threshold is `epsNewt / sqrt(n_blocks)`, which gives

```
sum over blocks of delnorm_b^2  <=  epsNewt^2
```

exactly today's global test when there is one block, and never looser than it.

**Blocks finish independently.** A block leaves the active set the moment it either takes
an undamped step within `step_tol` or rolls back a step that stopped reducing its
residual; the remaining blocks keep iterating. Treating either event as a whole-solve
exit — the obvious reading of the current single-block code — is wrong once there are
many blocks: with 20 blocks the chance that at least one has stalled on any given
iteration is high, and the solve returns "converged" with the other 19 still far from
their roots. That mistake showed up as 179/400 solved on the synthetic case below, and
per-block completion fixed it to 400/400.

**`CONVERGED_WRMS_AT_MAX_ITER` needs every still-running block.** The single-block code
latches a `converged` flag the first time the WRMS step test passes and reports success
on `max_iter` exhaustion if it was ever set. Latching that flag on *any* block is
unsound: on composite-electrode ESOH it returned success at ‖F‖ = 1.7e-2 because one
block had converged early while another ran to `max_iter` nowhere near its root. The
flag is per-block, and exhaustion only reports convergence when every block still in the
active set has latched. Retired blocks are converged by construction.

`initial_res_norm()` and `final_res_norm()` are whole-system inf-norms taken over every
solved state, not the masked residual of whichever blocks were last active, so the
accept/reject gate in `NonlinearSolverInitialConditions` compares like with like. That
costs one extra residual evaluation per solve.

Failure behaviour is unchanged: exhausting `max_iter` returns the same `NonlinearResult`
the current solver returns, with no retry.

## Verification

- **Bit-identical `coupled`.** A single-block partition reproduces the pre-change
  solver: same masking (a no-op), same WRMS formula, same Armijo decisions, same
  return values.
- **Residual improvement.** `residual_monotone()` reports whether every accepted
  linesearch lowered each active block's residual inf-norm. Exposed on
  `StandaloneNewtonSolver` and asserted over the sweeps below.
- **Converged residual.** The whole-system residual inf-norm at the returned solution,
  cross-checked in Python against `model.algebraic_eval`. This is the acceptance
  metric, not iteration count.
- **Pinned failures.** `TestElectrodeSOHCompositeKnownFailures` holds ten xfailed
  `ElectrodeSOHComposite` systems the Newton solver cannot solve, each anchored by a
  passing test showing the composite solver chain reaches a residual below 1e-8 on the
  same inputs. `xfail_strict` is on, so fixing any of them fails the suite until the
  marker is dropped.

## Measured results

All numbers from this worktree, macOS arm64.

### Structure actually present in PyBaMM models

CasADi sparsity of `jac_algebraic_eval`, algebraic columns, square:

| model | `n_alg` | components |
| --- | --- | --- |
| DFN | 100 | 1 |
| DFN + SEI | 140 | 1 |
| DFN, 2-phase negative | 40 | **2** |
| scalar ESOH | 2 | 1 |
| `ElectrodeSOHComposite`, voltage init | 9 | **2** |
| `ElectrodeSOHComposite`, SOC init | 9 | 1 |

Most models have nothing to split, and on those `decoupled` resolves to `coupled` and
costs only the `O(nnz)` analysis.

### Converged residual is unchanged by the mode

Scalar ESOH, 567 solves per mode (9x9 grid of `(x_100, x_0)` guesses spanning 0.001 to
0.999, crossed with seven `Q_Li` scalings from 0.15x to 1.15x nominal), `atol = 1e-6`:

| mode | resolved | blocks | max ‖F‖ | p90 ‖F‖ | median ‖F‖ | above atol | failures | mean iters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coupled | coupled | 1 | 8.9e-15 | 8.9e-15 | 8.9e-16 | 0 / 567 | 0 | 16.40 |
| decoupled | coupled | 1 | 8.9e-15 | 8.9e-15 | 8.9e-16 | 0 / 567 | 0 | 16.40 |

The solver-reported `final_res_norm()` agrees with the independent Python evaluation to
the last digit, which is what makes it usable as the acceptance metric.

### Composite-electrode ESOH — the case with real structure

`ElectrodeSOHComposite` (Chen2020_composite, `particle phases = ("2", "1")`,
current-sigmoid hysteresis on the negative secondary phase) is the only PyBaMM model
measured with more than one component, and it is the one with a known failure mode — the
`try_split_solve` rescue in `get_initial_stoichiometries_composite` exists because the
full solve fails. 78 solves per mode: 13 initial values (voltage spanning 2.5-4.2 V, or
SOC 0 to 1) crossed with six degradation states scaling `Q_n_1`, `Q_n_2`, `Q_p_1`,
`Q_Li` from 0.5x to 1.0x.

| init | mode | resolved | blocks | solved | failed | mean iters | max ‖F‖ | ‖F‖ > 1e-8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| voltage | coupled | coupled | 1 | 75 | 3 | 17.36 | 2.9e-14 | 0 |
| voltage | decoupled | decoupled | 2 | **76** | 2 | 16.93 | 2.2e-14 | 0 |
| SOC | coupled | coupled | 1 | 54 | 24 | 17.22 | 2.8e-14 | 0 |
| SOC | decoupled | coupled | 1 | 54 | 24 | 17.22 | 2.8e-14 | 0 |

Every returned solution is a true root: max ‖F‖ = 2.9e-14 over all solves, nothing above
1e-8, no mode ever returns a converged-but-wrong answer.

On voltage init `decoupled` gains one net solve — two gains and one loss, it fails on
`Q = (0.7, 1.0, 0.9, 0.75)` at 2.64 V where coupled succeeds — and takes 2.5% fewer
iterations. A net improvement, not a strict one. That loss is pinned as an xfail.

### Synthetic case where per-block damping wins

Twenty independent copies of a two-variable pair: a mildly nonlinear `a`, and a stiff
`exp(6b) - 1 - a` downstream of it. 400 random starts in `[-0.6, 0.6]^40`:

| mode | resolved | blocks | solved | mean iters | non-monotone |
| --- | --- | --- | --- | --- | --- |
| coupled | coupled | 1 | 400/400 | 9.31 | 0 |
| decoupled | decoupled | 20 | 400/400 | **8.51** | 0 |

9% cheaper when the components really do damp at different rates. This is the regime the
feature targets.

### Experiment initialisation, CC-CV with rests

Twenty steps (2 x discharge / rest / charge / CV hold / rest / 2C discharge / rest /
C-2 charge / CV hold / rest), so twenty Newton IC solves. `decoupled` matches `coupled`
exactly on every model tried, because none of them has more than one component. Worth
recording from the same run: the **unified experiment model has no block structure at
all**. It introduces one extra algebraic state, `Current variable [A]`, carrying the
step switch; every potential equation reads it and its own equation reads every
potential, so DFN+SEI goes from 21 strongly connected components under `legacy` to a
single irreducible block of 141 under `unified`. That cycle is structural — under
`legacy` the applied current is a parameter, under `unified` it must be an unknown so
the switch can select CC, CV or rest. Any structure-exploiting treatment of the
algebraic block is inert under `unified`. Worth tracking separately from this change.

### What staggering cost

A `staggered` mode was built and measured: strongly connected components via CasADi's
`Sparsity::scc`, ordered by DAG depth, each level solved to convergence before the next.
Zeroing the residual rows of frozen blocks makes the single full solve return the exact
block Gauss-Seidel step for a block-triangular Jacobian, so it was cheap to implement.
It reached the same residual as `coupled` everywhere, and:

| case | staggered vs coupled |
| --- | --- |
| scalar ESOH, 567 solves | same residual, +48% iters |
| composite ESOH, voltage init (3 blocks, 2 levels) | **-12 solves**, +55% iters |
| composite ESOH, SOC init (4 blocks, 4 levels) | **+7 solves**, +123% iters |
| DFN+SEI legacy experiment | -12% iters, +4% wall |
| synthetic 20-component | +45% iters |

The voltage-init collapse is instructive. Level 1 is the `x_0` block, whose root sits at
stoichiometries of 4e-4 and 1.3e-3 — the steep tail of the OCP — from a guess of 0.15.
Solved jointly with `x_100` it rides along on the well-conditioned part of the system;
solved alone it stalls at ‖F‖ = 2.9e-2 after 92 iterations. Isolating a block removes
the conditioning help it got from its neighbours, which rescues the four-stage chain and
wrecks the two-stage one. Problem-dependent in both directions, with no way to tell
which ahead of time, so it is not worth the level machinery it costs.

### Verdict

**Residual quality is never degraded by any mode.** Every solve that reports success
returns a true root: 8.9e-15 max over 567 scalar-ESOH solves, 2.9e-14 max over the
composite-electrode ESOH sweeps, 1e-11 or better on every IDAKLU IC.

That property was not free. Two defects had to be found and fixed to get it, both
invisible in iteration counts and both only detectable by checking the converged
residual: whole-solve exit on the first block to finish, and latching
`CONVERGED_WRMS_AT_MAX_ITER` on any one block. Each returned a confident wrong answer.

`decoupled` is net-positive everywhere measured and falls back to `coupled` on the
models that have nothing to split, which is most of them. Ship with the default at
`coupled`.

## Interface

New `SolverOptions` field `newton_block_mode`, `"coupled"` or `"decoupled"`, defaulting
to `"coupled"`. Plumbed from Python the same way `newton_mode` is, defaulted in
`idaklu_solver.py` and `nonlinear_solver.py`.

## Files

| File | Change |
| --- | --- |
| `idaklu_source/BlockPartition.hpp` | New. Partition struct plus the union-find builder. |
| `idaklu_source/NonlinearSolver.hpp` | Hold a `BlockPartition` and `alpha_`; drop `diff_idx_`. |
| `idaklu_source/NonlinearSolver.inl` | Masked residual and step, per-block Armijo and convergence test. |
| `idaklu_source/AlgebraicICBuilder.inl` | Build the square CSC for both IC modes, construct the partition, pass it in. |
| `idaklu_source/StandaloneNewtonSolver.{hpp,cpp}` | `block_mode` argument; expose the resolved mode, block count, iteration count, residual and monotonicity. |
| `idaklu_source/Options.{hpp,cpp}`, `idaklu.cpp` | `newton_block_mode`. |
| `pybamm/solvers/{idaklu_solver,nonlinear_solver}.py` | Option default. |
| tests, `CHANGELOG.md` | As above. |
