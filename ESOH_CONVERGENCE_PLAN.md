# Electrode SOH convergence: the minimal fix

Branch `claude/esoh-brent-convergence`, off `claude/brent-rootfinder`.

The electrode SOH solve is a Newton iteration on a system that is **two independent
scalar equations on brackets the code already computes**. Solving them with `pybamm.Brent`
instead makes convergence unconditional and deletes the fallback machinery.

---

## 1. The problem, measured

On `origin/main`, over 17,712 points (12 shipped parameter sets × 36 capacity states ×
41 `Q_Li`):

| | |
|---|---|
| solves returning **success** with residual > 1e-6 | **1564** |
| worst residual | **5.601197** |
| stoichiometries outside [0, 1] | 2220 |

Nothing raises. `get_esoh_default_solver()` composes a Newton solver plus three least
squares fallbacks, and **nothing checks the residual after one of them reports success**.
The native Newton's criterion is a weighted RMS norm of the *step*, not the residual, so a
poorly scaled system satisfies it while an equation is far off.

The worst case, reproduced on this branch:

```
Ai2020, Q_n=3.8030 Q_p=3.2194 Q_Li=3.7669
  current solve : x_100 = 0.990708   |F_100| = 5.6012      <- "success"
  bracket from _get_lims : [0.143972, 0.990529]  F(lo) = -2.11, F(hi) = +2.34
  brentq, same equation  : x_100 = 0.629672   |F_100| = 8.9e-16
```

The Newton iterate pinned itself at the top of the feasible range. A bracketed method
cannot do that, and the bracket it needs is already in the code.

---

## 2. Why this is a small change

**The default system is diagonal.** With `Q_Li` known,
`y_100 = (Q_Li − Q_n x_100)/Q_p` and `y_0 = (Q_Li − Q_n x_0)/Q_p` — the `x_100` terms in
`y_0` cancel identically. Verified: the analytic Jacobian is exactly

```
[[1.99e+00, 0.00e+00],
 [0.00e+00, 2.45e+01]]        J[1,0] == 0.0, bit-true, at every point
```

So `electrode_soh.py` already builds two scalar residuals and hands them to a solver that
does not know they are scalar:

```python
self.algebraic[x_100] = Up_100 - Un_100 - V_max     # depends only on x_100
self.algebraic[x_0]   = Up_0   - Un_0   - V_min     # depends only on x_0
```

**The brackets already exist.** `ElectrodeSOHSolver._get_lims` returns
`(x0_min, x100_max, y100_min, y0_max)`, already tightened by lithium conservation, and
already raises when the request is infeasible. That is exactly the bracket each scalar
equation needs.

So the fix is to stop solving a 2×2 system and start evaluating two `pybamm.Brent` nodes.

---

## 3. The edit

**E1 — `_ElectrodeSOH.__init__`.** Build the residual against a local unknown, then wrap
it, instead of registering an algebraic equation:

```python
unknown = pybamm.Variable("x_100")
# ... Un_100 / Up_100 built from `unknown` exactly as today ...
x_100 = pybamm.Brent(
    Up_100 - Un_100 - V_max,
    unknown,
    (pybamm.InputParameter("x_100_min"), pybamm.InputParameter("x_100_max")),
)
```

and the same for `x_0`. Everything downstream already consumes `x_100`/`x_0` as
expressions, so nothing else in the model changes.

**E2 — `ElectrodeSOHSolver`.** `_get_lims` already computes the bounds; add them to
`inputs`. The model now has no algebraic equations, so `solve()` evaluates the requested
variables directly — `_solve_full`, `_solve_split`, the `SolverError` fallback chain and
`get_esoh_default_solver` all become unreachable and go.

**E3 — `_ElectrodeSOHMSMR`.** Same shape, unknowns `Un_100`/`Un_0`.

**E4 — `known_value = "cell capacity"`.** The only genuinely coupled case: 2×2 and
structurally irreducible. It still reduces to one scalar equation — the first row gives
`y_100 = Up⁻¹(V_max + Un(x_100))`, itself a `Brent`, after which the second row is scalar
in `x_100`. Two nested `Brent` nodes. Do this **last**, behind its own test; if it proves
awkward, leave this path on the existing solver and ship E1–E3.

Estimated net: **~40 lines changed, mostly deletions.**

---

## 4. Tests

Acceptance is the **residual**, not "did not raise". That is precisely what the 1564
silent failures slipped past.

**T1 — non-composite, all shipped parameter sets.** For each of the 12 lithium-ion
parameter sets, sweep capacity states and `Q_Li` across the feasible range from
`_get_lims`, ~1200 points total. For every point assert: no exception, every
stoichiometry in [0, 1], and `max |F| < 1e-10` against the model's own equations.

**T2 — composite, all configurations.** The existing shape: 200 deterministic samples ×
{`both`, `negative`, `positive`} × {SOC, voltage} = 1200 targets. Same three assertions.

**T3 — the known-bad points, pinned.** `Ai2020` at `Q_n=3.8030, Q_p=3.2194, Q_Li=3.7669`
and the `Ramadass2004` cases, asserted to machine precision. These fail loudly on `main`
and must stay fixed.

**T4 — infeasible input still raises.** `_get_lims` raises before any solve when `Q_Li` is
out of range; and where the bracket has no sign change, `Brent` fails rather than
returning an endpoint. Assert both, so "rock solid" does not mean "silently returns
something".

Run T1 and T2 as integration tests; keep a small subset in the unit suite.

---

## 5. What not to do

- **Do not add a residual check to the existing solver.** It would convert 1564 silent
  wrong answers into 1564 loud failures. Correct, but not a fix.
- **Do not touch the composite univariate solve yet.** It is a separate reduction with its
  own bugs (`_BracketedInverse` saturates on endpoint values rather than the true range,
  wrong for `Ai2020`/`Ramadass2004`-shaped OCPs). Land E1–E3 first.
- **Do not keep the Newton fallback "just in case".** If a bracket with a sign change
  exists, Brent converges; if it does not, no solver should be returning an answer. A
  fallback here would restore the failure mode being removed.

---

## 6. Order

1. T1 + T3 against current `main` behaviour, so the baseline is recorded and red.
2. E1, E2 — the default path. Expect T1/T3 green.
3. E3 — MSMR.
4. T2 + composite.
5. E4 — cell capacity, or explicitly defer it.
