# Monotonicity-aware `U_inverse` and electrode SOH

Certify monotonicity exactly, then exploit it: a bracket derived from the data, a
Brent that treats its bounds as a guess, and a uniqueness guarantee that removes the
pole failure mode. Fall back to today's path otherwise.

---

## 1. The exact test — **landed**

`Interpolant.monotonicity(region=None) -> +1 | -1 | 0`
([interpolant.py](packages/pybamm/src/pybamm/expression_tree/interpolant.py)).

An interpolant is a piecewise polynomial, so its derivative is one too. A polynomial
attains its extrema on a closed interval only at the ends or at roots of its own
derivative — both finite sets, both closed form. So the derivative's true range on each
piece is read off the coefficients. Nothing is sampled; an overshoot between knots
cannot be missed.

- Strict: a derivative vanishing at isolated points keeps monotonicity, one vanishing
  over an interval does not. Exactly decidable — a flat stretch ⟺ a piece with all-zero
  coefficients, since a nonzero polynomial has finitely many roots. That is precisely
  the invertibility condition.
- `region` may reach outside the data; the end pieces are continued, matching `PPoly`.
- One tolerance, `1e-12 ×` the derivative's own scale, and it is **not** a sampling
  tolerance: a true stationary point evaluates to `-2.4e-18` rather than `0` from
  rounding in the spline coefficients. Real overshoots sit ~`8e-4` relative.

Verified: 4000 random cases × 3 interpolators × random sub-regions against
2,000,001-point sweeps, **0 disagreements**.

### Whole-domain verdicts, shipped sets

Only 4 of 22 electrodes are interpolants; the other 18 are analytic.

| set / electrode | kind | data | as built | wrong-way |
|---|---|---|---|---|
| Ai2020 U_n | cubic | monotone | **NOT monotone** | 98.8 mV |
| Ai2020 U_p | cubic | monotone (11 ties) | **NOT monotone** | 0.17 mV |
| OKane2022 U_n | cubic | not monotone | NOT monotone | 56.2 mV |
| Chayambuka2022 U_n | linear | monotone | decreasing | 0 |
| Chayambuka2022 U_p | linear | not monotone | NOT monotone | 3.5 mV |

`Ai2020` U_n is monotone as data and not monotone as the spline PyBaMM builds from it.
Ringing, 98.8 mV of wrong-way travel on a 3.575 V span.

---

## 1b. `U_asymptotes` — the region is known in closed form

The whole-domain verdict above is the wrong question, because `PhaseParameters.U` never
evaluates a bare interpolant. It evaluates

```
U(sto, T) = U_ref(clip(sto)) + dUdT(clip(sto))·(T − T_ref) + U_asymptotes(sto)
```

with `clip` to `[tol, 1−tol]`, `tol = settings.tolerances["U__c_s"] = 1e-10`. The
interpolant argument is clipped; the barrier's is **not**.

`U_asymptotes` is a **softplus**, `a·log(1 + exp(−b(sto − c)))` mirrored about
`sto = 1/2`, with an exact linear continuation below `sto_limit = c − 53·ln2/b`. Not
`1/sto + 1/(sto−1)` — the comment at
[lithium_ion_parameters.py:726](packages/pybamm/src/pybamm/parameters/lithium_ion_parameters.py#L726)
says that, but it is stale and should be fixed. It matters, because a softplus is finite
and continuous everywhere while `1/sto` has a pole. Measured:

| sto | barrier | d barrier/d sto |
|---|---|---|
| `−2` | `2.83e6` V | `−1.417e6` |
| `−0.01` | `1.31e4` V | `−1.417e6` |
| `c = −7.7e-4` | `142.1` V | — |
| `0` | `1.000` V | `−6893` |
| `tol = 1e-10` | `0.999999` V | `−6893` |
| `0.001` | `1.00e-3` V | `−6.91` |
| `0.0045` | `4.6e-14` V | `−1.6e-10` |
| `0.5` | **exactly `0.0`** | **exactly `−0.0`** |

The barrier is **exactly `0.0` on `(0.00454632, 0.99545368)`, 99.09% of `[0, 1]`** — the
closed form is `c + 53·ln2/b`, the point where `1 + exp(·)` rounds to `1.0`.

Three consequences, and they set the whole design:

1. **Inside the window the barrier does nothing.** Its derivative is exactly `−0.0`, so
   `U' = U_ref'` there and the barrier **cannot rescue non-monotone data**. Certification
   inside the window is purely the interpolant's job — which is what §1 does.
2. **Outside the clip the interpolant does nothing.** `U_ref` is frozen at its endpoint,
   so `U' = barrier' < 0` strictly. Certified for free, no analysis needed.
3. **In the two ramps the barrier dominates.** `|barrier'|` runs from `1.6e-10` up to
   `6893`, orders above any real OCP slope.

So: **`U` is strictly decreasing on all of ℝ if the interpolant is strictly decreasing on
the barrier-free window.** That is the composition statement, and it needs no analysis of
the barrier beyond the closed-form window.

### Why this is the right region to test

| set / electrode | full domain | barrier-free window |
|---|---|---|
| **Ai2020 U_n** | NOT monotone | **decreasing** |
| Ai2020 U_p | NOT monotone | NOT monotone |
| OKane2022 U_n | NOT monotone | NOT monotone |
| Chayambuka2022 U_n | decreasing | decreasing |
| Chayambuka2022 U_p | NOT monotone | NOT monotone |

`Ai2020` U_n is certified on the window, under `cubic`, with no parameter-set change —
its 98.8 mV of ringing lives entirely in the slivers the barrier owns.

---

## 2. Per-region verdicts

Maximal monotone stretches over the full data range, from the roots of each derivative:

| set / electrode | stretches | widest stretch | as % of domain |
|---|---|---|---|
| **Ai2020 U_n** | 8 | `[0.001587, 0.998192]` | **99.66%** |
| Ai2020 U_p | 23 | `[0.400000, 0.787812]` | 64.75% |
| OKane2022 U_n | 117 | `[0.883121, 1.000000]` | 11.69% |
| Chayambuka2022 U_p | 5 | `[0.655863, 0.999940]` | 43.56% |

So restricting the region rescues `Ai2020` U_n outright — the reversals are slivers at
the two edges, 5 of them narrower than 0.1% of the domain. It does not rescue
`OKane2022` U_n, whose data is genuinely non-monotone; no interpolator fixes that.

This needs no new API: `monotonicity(region)` already takes the region, and the region
the eSOH search actually uses is known exactly (§4). Test the region you will search,
not the whole domain.

### The cheaper fix for Ai2020: pchip

`pchip` preserves the monotonicity of its data by construction.

| | cubic | pchip |
|---|---|---|
| Ai2020 U_n | NOT monotone, 98.8 mV | **decreasing over the whole domain** |
| Ai2020 U_p | NOT monotone, 0.17 mV | **decreasing over the whole domain** |
| OKane2022 U_n | NOT monotone (data isn't) | NOT monotone (data isn't) |

`max |cubic − pchip|` is **190 mV** on Ai2020 U_n. Both fit the same data and disagree
by 190 mV between the points — larger than anything the solver work is chasing, and a
one-word change. Separate decision, since it moves shipped numbers; raise on its own.

---

## 3. Soft bounds — the barrier argues *against* them for the monotone path

I had this backwards. The barrier does guarantee that expansion terminates, but it also
guarantees that **the solve can never fail**, and that is worse.

Over `[tol, 1−tol]` the barrier spans only `+1 V` to `−1 V`, so a target within ~1 V of
the OCP's endpoint already produces a sign change inside the clip. Expand past it and the
barrier absorbs `142 V`, then `13 kV`, then anything — a numerically valid, uniquely
determined root at a physically meaningless `sto`. Soft bounds walk into that ramp *by
design*. This is the mechanism behind the existing known-bad points (residual `5.6` and
`3.4e12` returned as successes).

**So for a certified-monotone interpolant, do not soften the bounds. Harden them to the
barrier-free window** `[0.00454632, 0.99545368]`, intersected with the OCP's data range:

- Inside it the barrier is exactly `0.0`, so the residual **is** the plain interpolant —
  a piecewise polynomial. §1 certifies it exactly and the knot table brackets it directly.
- A sign change there ⇒ the root is physical, and unique by monotonicity.
- No sign change there ⇒ the request is infeasible in the physical range. Say so, rather
  than letting the barrier absorb it.

That is strictly better than expanding: same guarantee, and it distinguishes "no answer"
from "an answer in the barrier", which expansion cannot.

**What monotonicity still buys**, and it is the valuable part:

1. **At most one root** — no branch to select, so no guess is needed to disambiguate.
2. **A sign change is a root.** `Ramadass2004` U_n jumps `−7.09e14 → +3.65e14` at
   `x ≈ 0.08914525` — a sign change with no root. That cannot happen inside the window
   under certification, because there the residual is a piecewise polynomial: finite and
   continuous. The failure mode is removed, not detected. (The barrier itself introduces
   no pole either — softplus, max `2.83e6` at `sto = −2`.)
3. **Every evaluation is a half-line.** `f(c) > 0` on an increasing `f` puts the root
   below `c`. One evaluation orients the search.

**So what is the guess actually for?** Not existence, not uniqueness, not bracketing —
the window gives all three. It is for warm solves (§4), where the previous answer puts
Brent inside one knot spacing immediately.

### Expansion is still worth building — for the analytic OCPs

18 of 22 electrodes have no table, so there is no window to compute and no exact
certificate. There, soft bounds plus expansion are the best available, and the barrier is
what makes them terminate. Same pre-loop, gated on `direction != 0`, and the answer is
labelled approximate.

```
a  = clamp(guess, lo, hi);  fa = f(a)
s  = -direction * sign(fa)          # (3) says which way the root is
step = (hi - lo) / 2
repeat up to max_expansions:
    b = a + s*step;  fb = f(b)
    if !isfinite(fb):  step /= 2; continue      # extrapolation overflow
    if sign(fb) != sign(fa):  return brent(f, a, b)   # true bracket, unchanged Brent
    a, fa = b, fb;  step *= 2                   # monotonicity says keep going
return NO_ROOT
```

- Expands in **one** direction only, where a general `zbrac` must try both.
- Each step keeps the previous point, so the final bracket is tight (step ratio 2) and
  no evaluation is discarded.
- Terminates whenever `f` changes sign along the ray, which the barrier's linear
  continuation guarantees: slope `−1.417e6` V per unit `sto`, so the residual spans ℝ.
- Exhausting `max_expansions` means no root on the ray. Under a *sampled* certificate
  that is a strong hint, not a proof.
- **It will walk into the barrier ramp, and any root it finds there is not physical.**
  Whatever it returns must be checked against the barrier-free window before use — which
  is exactly why the certified path takes the window directly instead.

**Where it goes.** `brent_impl.hpp`, the single source shared by the compiled and
codegen paths — roughly 25 lines ahead of the existing iteration. New options
`direction` (default `0` = today's behaviour, require a valid bracket) and
`max_expansions`. Inputs 1 and 2 stop being a hard bracket and become soft bounds when
`direction != 0`. The rootfinder's `x0` slot already exists and is currently thrown away
([brent.py:179-181](packages/pybamm/src/pybamm/expression_tree/brent.py#L179)) — it
becomes the guess, so no interface change.

**Two guards still missing** and needed here more than before, since expansion
deliberately walks into extrapolation: `isfinite` on every residual, and `q != 0` before
`d = p / q`.

---

## 4. Wiring into the electrode SOH solve

### The composition rule is exact and needs no new analysis

`solve_for_limit` drives `G(x) = U_p(y(x)) − U_n(x) − V` with
`y(x) = (Q_Li − x·Q_n)/Q_p`, so `dy/dx = −Q_n/Q_p < 0` and

```
G'(x) = −(Q_n/Q_p)·U_p'(y) − U_n'(x)
```

If `U_p' < 0` on the searched y-range and `U_n' < 0` on the searched x-range, both terms
are strictly positive, so **`G` is strictly increasing** — certified from the two OCPs
separately, with no analysis of `G` itself. Monotonicity composes; that is what makes
the per-interpolant test in §1 sufficient for the whole solve.

Both searched ranges sit inside the barrier-free window, so the barrier contributes
exactly `−0.0` to both derivatives and drops out of the argument entirely.

### The bracket is exact, not a guess

Restrict `x` and `y` to the barrier-free window first. There `U_p` is monotone, so its
range is exactly its two endpoint values on that window — no search needed. Then

```
U_n(x) = U_p(y) − V  ∈  [min U_p − V,  max U_p − V]     ⇒     x ∈ U_n^{-1}([...])
```

and `U_n^{-1}` at those two values is a table inversion of a monotone interpolant —
`searchsorted` on the knot values, exact. Intersect with the window and the data range.
Symmetrically for `y`. This is the "known exactly a priori" bracket, derived rather than
estimated, and it is also the region to hand `monotonicity()`.

If that bracket carries no sign change, the request has **no physical answer**. Today the
barrier absorbs it and a root comes back anyway; this is the check that turns that into
an error.

### Per interpolator

- **linear**: the swapped table *is* the inverse. No solve at all.
- **cubic / pchip**: `searchsorted` gives the two knots straddling the target — a bracket
  one knot-spacing wide, so Brent takes a handful of iterations, deterministically.
- **analytic**: no table; soft bounds plus §3's expansion, with monotonicity only
  sampled, so the answer is labelled approximate.

### Call sites

- `electrode_soh.py::solve_for_limit` — classify `U_n`, `U_p` over the §4 bracket; if
  both certified, build the `Brent` in monotone mode with that bracket. Otherwise
  unchanged.
- `electrode_soh_composite.py::_BracketedInverse` — this one inverts a **single** OCP per
  phase, so it is the direct hit: closed form for linear, one-knot bracket otherwise.
  Replace it with `U_inverse`.

### Warm solves

The inverse table depends on `(phase, lithiation, T)` and on none of `Q_n`, `Q_p`,
`Q_Li`, `V` — build once, reuse for every solve after. Cold cost is zero for
interpolants; the table is the data.

---

## 5. Where `U_inverse` lives

Knots exist only after `ParameterValues.process_symbol`, and `PhaseParameters` has no
`ParameterValues`, so `U_inverse` cannot classify at build time. It stays the single
entry point and always returns a `Brent`; the classification runs in
`Brent._to_casadi`, where the residual is processed and the knots are reachable, behind
an explicit flag. Callers classifying and passing the result in would spread the logic
across every call site and break "single source of truth".

---

## 6. Fallback

Not certified → today's general path, unchanged, correct-over-fast, and it says when it
is less accurate. Today that is every analytic OCP, `Ai2020` U_p, `OKane2022` U_n,
`Chayambuka2022` U_p, and `Ramadass2004` U_p (the pole). `Ai2020` U_n and
`Chayambuka2022` U_n are certified on the window and take the fast path.

---

## 7. Order

1. ~~`Interpolant.monotonicity`~~ — **landed**, 37 tests, `-nauto` clean.
2. The barrier-free window as a named constant derived from `_U_ASYMPTOTE_PARAMETERS`
   (`c + 53·ln2/b`), not a literal, plus a test that the barrier is bit-exactly `0.0`
   across it. Pure addition, and it is what makes everything below exact.
3. Knot-table inverse (`searchsorted`) + the §4 bracket derivation on that window.
   Assert it round-trips the forward evaluation, and that a bracket with no sign change
   raises instead of returning a barrier root.
4. `direction` / `max_expansions` in `brent_impl.hpp` for the analytic OCPs only, plus
   the two missing guards. Test against `Ramadass2004`'s pole.
5. Wire `solve_for_limit` and `_BracketedInverse`. Re-run the 1200-point and
   14,400-point sweeps — must not regress from today's 38 raised / 9 out-of-range /
   115 bad.
6. Warm-solve cache, with a before/after number rather than an assertion.

Steps 2–3 are pure addition and can land alone; they are also where most of the value
is. Step 4 changes `pybammsolvers`, so it needs a rebuild and has never been built on
Linux.

---

## 8. Underneath this

**13 failing unit tests**, uncommitted: `_solve_split`, the MSMR paths and
`get_initial_ocps` still route through a solver with nothing left to solve. Independent
of this plan, but it sits on top of them.

Also open, and both now matter more:

- `extrapolate` is honoured by the NumPy path and ignored by the CasADi one, so any root
  outside the knots is evaluator-dependent. The certified path no longer goes there;
  §3's expansion, for the analytic OCPs, still does.
- The comment at
  [lithium_ion_parameters.py:726](packages/pybamm/src/pybamm/parameters/lithium_ion_parameters.py#L726)
  describes a `1/sto + 1/(sto-1)` barrier. The code implements a softplus. The difference
  is a pole versus no pole, so the stale comment is worth fixing on its own.
