# `U_inverse` as the single source of truth

Plan to finish `PhaseParameters.U_inverse` and route every OCP inversion through it.

---

## 1. What is wrong with the draft

| # | Issue |
|---|---|
| B1 | `_is_monotonic` builds a **symbolic** expression (`pybamm.linspace/diff/all`). `if monotonic:` on a `Symbol` is always truthy and is forbidden by AGENTS.md. |
| B2 | **`PhaseParameters` has no `ParameterValues`**, so monotonicity cannot be evaluated here at all — the OCP is still a `FunctionParameter`. This is structural, not a bug to patch. |
| B3 | `residual` uses `pybamm.Variable(name)` but `unknown` is `pybamm.BrentUnknown(name)`. Different objects, so `Brent.__init__` raises *"does not appear in"*. |
| B4 | `pybamm.Brent(..., bracket=…, xtol=…)`; the node's signature is `bounds` (positional) and `abstol` (keyword-only). |
| B5 | `_U_inverse_bounded` ignores the caller's `sto_bounds` and hardcodes ±0.1. |
| B6 | `_U_inverse_guess` is empty, so the module does not import. |
| B7 | `_U_inverse_bounded`'s docstring says "fallback when the OCP is non-monotonic" — it is the monotonic path. |
| B8 | `num_points_monotonic`'s `None` check is dead given the signature default. |

**B5 is the dangerous one.** Slack of ±0.1 puts the search outside [0, 1], where a
tabulated OCP is undefined and CasADi's bspline returns 0 — exactly the failure that
produced 961 defective Ai2020 results. Warning afterwards does not help: the *solve*
lands on the discontinuity. Bounds must stay inside the OCP's domain.

---

## 2. The design decision

Monotonicity can't be tested where `U_inverse` lives. Three ways out:

- **(a) caller passes `monotonic=`** — pushes the problem onto every caller and breaks
  "single source of truth".
- **(b) decide at solve time, in the plugin** — it already has the residual and the
  bracket, so it can scan.
- **(c) Newton from a guess** — needs a guess the caller may not have.

**Take (b) as the default, keep (c) for branch selection.** A scan subsumes most of
what (c) exists for, needs no guess, and — critically — fixes the pole problem in the
same stroke. Today's sweep still has **115 defective composite results**, all from poles
and from nobody checking the residual.

The scan is cheap and exact in what it decides:

```
sample the bracket at N points
  -> collect sub-intervals with a sign change
  -> discard any whose endpoints are both large (a pole, not a root)
  -> Brent on the survivor
0 survivors  -> no root: fail
1 survivor   -> the answer, no guess needed
2+ survivors -> non-monotonic: use sto_guess to pick, else fail naming the count
```

This makes `U_inverse` total: it either returns a root or says why not, for monotone
and non-monotone OCPs alike.

---

## 3. Edits

**E1 — `brent_impl.hpp` / `brent.cpp`: bracket scanning.** Add a `scan` option (0 =
off, default ~32). Before the iteration, sample `scan` points, pick the first
sub-interval with a sign change whose endpoint residuals are both below a pole
threshold, and bracket on that. Returns the existing status codes plus one for
"multiple roots". Lives in the shared source so codegen and the interpreted path stay
identical.

**E2 — `pybamm.Brent`: a `method` option.** `"brent"` (default) and `"newton"`, the
latter emitting `casadi.rootfinder(..., "newton", …)` with `sto_guess` as the initial
guess. No new C++ — CasADi ships it. Rename `bounds`→`bracket` and `abstol`→`xtol` to
match the draft's naming, which reads better and matches scipy.

**E3 — `PhaseParameters.U_inverse`.** Becomes a thin builder, with no monotonicity
test:

```python
def U_inverse(self, U_target, T, lithiation=None, sto_guess=None,
              sto_bounds=None, xtol=None):
    tol = pybamm.settings.tolerances["U__c_s"]
    bracket = sto_bounds or (tol, 1 - tol)
    unknown = pybamm.BrentUnknown(f"{self.domain} sto")
    residual = self.U(unknown, T, lithiation) - U_target
    if sto_guess is None:
        return pybamm.Brent(residual, unknown, bracket, xtol=xtol)
    return pybamm.Brent(residual, unknown, bracket, method="newton",
                        guess=sto_guess, xtol=xtol)
```

Fixes B1, B3, B4, B5, B6, B7, B8 and sidesteps B2. Delete `_U_inverse_bounded`,
`_U_inverse_guess`, `_is_monotonic` and `num_points_monotonic`.

**E4 — the OCP domain belongs to the OCP.** Move the data-range logic currently in
`ElectrodeSOHSolver._ocp_domain` onto `PhaseParameters`, so `U_inverse` defaults its
own bracket to where the OCP is defined rather than [0,1]. That is the one piece of
"exactly known a priori" the eSOH solver should not be carrying.

**E5 — eSOH calls it.** `electrode_soh.py`'s `solve_for_limit` becomes a difference of
two `U_inverse` results rather than building its own `Brent`; `electrode_soh_composite.py`'s
phase inversions call it too, replacing `_BracketedInverse` entirely.

---

## 4. Tests

- **T1** `U_inverse` round-trips: for every shipped parameter set and both electrodes,
  `U_inverse(U(s)) == s` across the OCP's domain, monotone or not.
- **T2** non-monotonic with multiple roots: without a guess it fails naming the count;
  with a guess it returns the nearest root.
- **T3** pole rejection: Ramadass2004's negative OCP at `x ≈ 0.0891` must fail, not
  return the pole.
- **T4** the 14,400-solve composite sweep goes to **0 raised / 0 out-of-range / 0
  residual > 1e-7**, from today's 38 / 9 / 115.
- **T5** the existing 1200-point non-composite sweep stays green.

---

## 5. Order

1. E3 + E4 so the module imports and `U_inverse` works for monotone OCPs; T1.
2. E1 scanning; T3, then re-run T4 to see how much of the 115 it clears.
3. E2 Newton; T2.
4. E5 wiring; T4, T5.

Step 2 is the one that matters — it decides whether the remaining composite failures
are poles (fixed by scanning) or something else still unfound.
