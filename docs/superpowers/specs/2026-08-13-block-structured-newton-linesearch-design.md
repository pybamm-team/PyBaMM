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
   - `auto` — `DECOUPLED` if there is more than one independent component, else
     `COUPLED`. `STAGGERED` is never selected automatically; it is opt-in.
     `COUPLED` is `DECOUPLED` with a single block, so `auto` only pays for the
     structural analysis and then falls back to today's path when there is nothing
     to exploit.
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
unchanged. The per-block threshold is `epsNewt / sqrt(|blocks in the level|)`, fixed for
the level, which gives

```
sum over the level's blocks of delnorm_b^2  <=  epsNewt^2
```

This is exactly today's global test when there is one block, and is never looser than it.

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
solved state, not the masked residual of whichever level ran last, so the accept/reject
gate in `NonlinearSolverInitialConditions` compares like with like. That costs one extra
residual evaluation per solve.

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
- **Residual improvement.** `residual_monotone()` reports whether every accepted
  linesearch lowered each active block's residual inf-norm. Exposed on
  `StandaloneNewtonSolver` and asserted over the sweeps below.
- **Converged residual.** The whole-system residual inf-norm at the returned solution,
  cross-checked in Python against `model.algebraic_eval`. This is the acceptance
  metric, not iteration count.

## Measured results

All numbers from this worktree, Chen2020 unless stated, macOS arm64.

### Converged residual is unchanged by the mode

Composite ESOH, 567 solves per mode (9x9 grid of `(x_100, x_0)` guesses spanning
0.001 to 0.999, crossed with seven `Q_Li` scalings from 0.15x to 1.15x nominal),
`atol = 1e-6`, backtrack budget 100:

| mode | resolved | blocks/levels | max ‖F‖ | p90 ‖F‖ | median ‖F‖ | above atol | failures | mean iters | wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coupled | coupled | 1 / 1 | 8.9e-15 | 8.9e-15 | 8.9e-16 | 0 / 567 | 0 | 16.40 | 343 ms |
| decoupled | coupled | 1 / 1 | 8.9e-15 | 8.9e-15 | 8.9e-16 | 0 / 567 | 0 | 16.40 | 343 ms |
| staggered | staggered | 2 / 2 | 8.9e-15 | 8.9e-15 | 8.9e-16 | 0 / 567 | 0 | 24.28 | 412 ms |

Every mode lands on the same machine-precision residual from every guess, and the bare
C++ Newton never fails, so the composite solver's `AlgebraicSolver` fallbacks are never
reached on this grid. Staggering buys nothing here and costs 48% more iterations.

The solver-reported `final_res_norm()` agrees with the independent Python evaluation to
the last digit (0.0 and 8.9e-16 respectively at the nominal point), which is what makes
it usable as the acceptance metric.

### Composite ESOH: setup, initial and warm solve

`ElectrodeSOHSolver` full model, `_set_up_solve` initial conditions, `atol = 1e-6`.
"Setup" is `NonlinearSolver.set_up`: CasADi function construction, the block analysis,
and the C++ solver build. Warm is the best of 30 repeat solves.

| parameter set | mode | resolved | blk/lvl | setup | initial | warm | iters | ‖F‖ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chen2020 | coupled | coupled | 1/1 | 9.99 ms | 9.27 ms | 176 us | 8 | 0.0 |
| Chen2020 | auto | coupled | 1/1 | 6.08 ms | 8.94 ms | 175 us | 8 | 0.0 |
| Chen2020 | decoupled | coupled | 1/1 | 5.77 ms | 9.07 ms | 171 us | 8 | 0.0 |
| Chen2020 | staggered | staggered | 2/2 | 6.56 ms | 8.90 ms | 241 us | 14 | 0.0 |
| Chen2020 cell-capacity | coupled | coupled | 1/1 | 5.37 ms | 8.83 ms | 294 us | 10 | 8.9e-16 |
| Chen2020 cell-capacity | staggered | coupled | 1/1 | 5.38 ms | 9.08 ms | 313 us | 10 | 8.9e-16 |
| OKane2022 | coupled | coupled | 1/1 | 6.13 ms | 8.49 ms | 184 us | 8 | 0.0 |
| OKane2022 | staggered | staggered | 2/2 | 5.97 ms | 8.80 ms | 253 us | 14 | 0.0 |
| Ecker2015 | coupled | coupled | 1/1 | 7.17 ms | 9.76 ms | 203 us | 9 | 0.0 |
| Ecker2015 | staggered | staggered | 2/2 | 6.69 ms | 10.22 ms | 273 us | 14 | 0.0 |

Setup cost is flat: the block analysis is inside the run-to-run noise of the CasADi work
that dominates `set_up`. Initial solve is unchanged. Warm solve is unchanged for
`coupled`/`auto`/`decoupled` and 35-40% slower for `staggered`, which is the cost of
converging `x_0` alone before starting `x_100`.

The full ESOH model is a 2x2 system whose blocks are 1x1: `x_100` reads `x_0`, so it is
one component and two levels. `auto` therefore resolves it to `coupled`, and the
cell-capacity variant is structurally irreducible and resolves to `coupled` under every
mode.

### IDAKLU algebraic IC

`sim.solve([0, 600])`, residual of the algebraic system at the solver's own `t = 0`
state:

| model | blocks/levels under staggered | ‖F_alg(0)‖ coupled | decoupled | staggered |
| --- | --- | --- | --- | --- |
| DFN | 1 / 1 | 1.21e-11 | 1.21e-11 | 1.21e-11 |
| DFN+SEI | 21 / 2 | 2.12e-11 | 2.12e-11 | 1.03e-11 |
| DFN 2-phase negative | 2 / 1 | 4.30e-11 | 4.35e-11 | 4.35e-11 |

Voltages agree to 1e-11 or better everywhere. Newton IC iteration counts: DFN 6/6/6,
DFN+SEI 7/7/8, DFN+SEI+plating 9/9/11, DFN 2-phase 7/7/7. Wall time is within noise in
every case. Staggering halves the DFN+SEI IC residual, at a cost of one extra iteration.

### Composite-electrode ESOH — the case with real block structure

`ElectrodeSOHComposite` (Chen2020_composite, `particle phases = ("2", "1")`, current-sigmoid
hysteresis on the negative secondary phase) is the only PyBaMM model measured with a
genuinely rich partition, and it is the one with a known failure mode — the
`try_split_solve` rescue in `get_initial_stoichiometries_composite` exists because the
full solve fails:

| `initialization_method` | n_alg | blocks | levels | components | chain |
| --- | --- | --- | --- | --- | --- |
| voltage | 9 | 3 (3+3+3) | 2 | **2** | {x_100} → {x_0}, {x_init} independent |
| SOC | 9 | 4 (3+3+2+1) | **4** | 1 | {x_100} → {x_0} → {x_init} → {y_init} |

78 solves per mode: 13 initial values (voltage spanning 2.5-4.2 V, or SOC 0 to 1) crossed
with six degradation states scaling `Q_n_1`, `Q_n_2`, `Q_p_1`, `Q_Li` from 0.5x to 1.0x.

| init | mode | resolved | blk/lvl | solved | failed | mean iters | max ‖F‖ | ‖F‖ > 1e-8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| voltage | coupled | coupled | 1/1 | 75 | 3 | 17.36 | 2.9e-14 | 0 |
| voltage | auto / decoupled | decoupled | 2/1 | **76** | 2 | 16.93 | 2.2e-14 | 0 |
| voltage | staggered | staggered | 3/2 | 63 | 15 | 26.86 | 2.7e-14 | 0 |
| SOC | coupled / auto / decoupled | coupled | 1/1 | 54 | 24 | 17.22 | 2.8e-14 | 0 |
| SOC | staggered | staggered | 4/4 | **61** | 17 | 38.38 | 2.7e-14 | 0 |

Every returned solution is a true root: max ‖F‖ = 2.9e-14 over all 312 solves, nothing
above 1e-8, no mode ever returns a converged-but-wrong answer.

- `decoupled` on voltage init: +1 net solve (two gains, one loss — it fails on
  `Q = (0.7, 1.0, 0.9, 0.75)` at 2.64 V where coupled succeeds) and 2.5% fewer
  iterations. Not a strict improvement, a net one.
- `staggered` on SOC init: **+7 solves, 54 to 61**, the largest robustness gain measured
  anywhere. The four-level chain lets each 1-3 variable stage converge on its own where
  the joint 9-variable solve stalls. It costs 2.2x the iterations.
- `staggered` on voltage init: 12 solves worse, failing all 13 nominal-degradation
  voltages. Level 1 is the `x_0` block, whose root sits at stoichiometries of 4e-4 and
  1.3e-3 — deep in the steep tail of the OCP — and its default guess is 0.15. Solved
  jointly with `x_100` it rides along on the well-scaled part of the system; solved alone
  it stalls at ‖F‖ = 2.9e-2 after 92 iterations. That is inherent to staggering, not a
  tuning problem: isolating a block removes the conditioning help it got from its
  neighbours.

This is the sharpest result in the set. `staggered` is not uniformly better or worse —
it is a different trade, and which way it goes depends on whether the hard block is
helped or hurt by isolation.

### Experiment initialisation, CC-CV with rests

Twenty steps (2 x discharge / rest / charge / CV hold / rest / 2C discharge / rest /
C-2 charge / CV hold / rest), so twenty Newton IC solves. Totals over the whole run:

| model | experiment mode | block mode | resolved | IC solves | Newton iters | IC failures | wall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DFN | unified | any | coupled 1/1 | 20 | 94 | 4 | 0.55 s |
| DFN | legacy | any | coupled 1/1 | 20 | 98 | 0 | 0.95 s |
| DFN+SEI | unified | any | coupled 1/1 | 20 | 101 | 4 | 0.64 s |
| DFN+SEI | legacy | coupled / auto / decoupled | coupled 1/1 | 20 | 162 | 0 | 1.11 s |
| DFN+SEI | legacy | staggered | **staggered 21/2** | 20 | **143** | 0 | 1.16 s |

All modes reach the same end state (`t_end`, `V_end` and discharge capacity agree to
1e-8). Two things come out of this:

**Staggering wins on legacy DFN+SEI**: 143 Newton iterations against 162, a 12%
reduction, the only real-model win measured. Wall time is 4% worse because the extra
iterations at level 1 each pay for a full 140x140 KLU solve to move 20 scalars — the
deferred 1x1 shortcut would turn this into a net win.

**The unified experiment model has no block structure to exploit.** It introduces one
extra algebraic state, `Current variable [A]`, carrying the step switch. Every potential
equation reads it and its own equation reads every potential, so the 21 strongly
connected components of legacy DFN+SEI collapse into a single irreducible block of 141:

| model | experiment mode | n_alg | SCC blocks |
| --- | --- | --- | --- |
| DFN | legacy | 100 | 1 |
| DFN | unified | 101 | 1 |
| DFN+SEI | legacy | 140 | 21 (120 + 20 singletons) |
| DFN+SEI | unified | 141 | **1** |

That cycle is structural, not incidental: under `legacy` the applied current is a
parameter, under `unified` it must be an unknown so the switch can select CC, CV or
rest. Any structure-exploiting treatment of the algebraic block — this partition, and
equally a block preconditioner — is inert under `unified` until the switch is
reformulated. Worth recording separately from this change.

Also visible: `unified` logs 4 Newton IC failures per run against 0 for `legacy`, on
both models and in every block mode, while still producing the same answer. That is
pre-existing, unrelated to this change, and not something the block modes can address
given the partition is a single block.

### Synthetic case where per-block damping does win

Twenty independent copies of a two-variable pair: a mildly nonlinear `a`, and a stiff
`exp(6b) - 1 - a` downstream of it. 400 random starts in `[-0.6, 0.6]^40`:

| mode | resolved | blocks/levels | solved | mean iters | non-monotone |
| --- | --- | --- | --- | --- | --- |
| coupled | coupled | 1 / 1 | 400/400 | 9.31 | 0 |
| decoupled | decoupled | 20 / 1 | 400/400 | 8.51 | 0 |
| staggered | staggered | 40 / 2 | 400/400 | 13.51 | 0 |

`DECOUPLED` is 9% cheaper than `COUPLED` when the components really do damp at different
rates. This is the regime the feature targets.

### Verdict

**Residual quality is never degraded by any mode.** Every solve that reports success
returns a true root: 8.9e-15 max over 567 hard scalar-ESOH solves, 2.9e-14 max over 312
composite-electrode ESOH solves, 1e-11 or better on every IDAKLU IC. Nothing above
tolerance, anywhere.

That property was not free. Two defects had to be found and fixed to get it, both
invisible in iteration counts and both only detectable by checking the converged
residual: whole-solve exit on the first block to finish, and latching
`CONVERGED_WRMS_AT_MAX_ITER` on any one block. Each returned a confident wrong answer.

Convergence *rate* is a genuinely mixed picture, and the claim that block structure can
only help does not survive measurement:

| case | decoupled vs coupled | staggered vs coupled |
| --- | --- | --- |
| scalar ESOH, 567 solves | identical (resolves to coupled) | same residual, +48% iters |
| composite ESOH, voltage init | +1 solve, -2.5% iters | **-12 solves**, +55% iters |
| composite ESOH, SOC init | identical (resolves to coupled) | **+7 solves**, +123% iters |
| DFN+SEI legacy experiment | identical | -12% iters, +4% wall |
| synthetic 20-component | -9% iters | +45% iters |

`STAGGERED` is a different trade, not a strictly better one: isolating a block removes
the conditioning help it got from its neighbours. That rescues the four-stage
composite-ESOH chain and wrecks the two-stage one.

Ship it as: default `coupled`; `auto` (which picks `decoupled` or falls back to
`coupled`) costs only the structural analysis and is net-positive everywhere measured;
`STAGGERED` opt-in and documented as problem-dependent. Do not promote `staggered` to
`auto`.

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
