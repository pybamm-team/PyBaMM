## Benchmarks

This directory contains the benchmark suites of the PyBaMM project. These benchmarks can be run using [airspeed velocity](https://asv.readthedocs.io/en/stable/) (`asv`).

### Running the benchmarks

First of all, you'll need `asv` installed:

```shell
pip install asv
```

To run the benchmarks for the latest commit on the `main` branch, simply enter the following command:

```shell
asv run
```

If it is the first time you run `asv`, you will be prompted for information about your machine (e.g. its name, operating system, architecture...).

Running the benchmarks can take a while, as all benchmarks are repeated several times to ensure statistically significant results. If accuracy isn't an issue, use the `--quick` option to avoid repeating each benchmark multiple times.

```shell
asv run --quick
```

Benchmarks can also be run over a range of commits. For instance, the following command runs the benchmark suite over every commit between version `0.3` and the tip of the `main` branch:

```shell
asv run v0.3..main
```

Further information on how to run benchmarks with `asv` can be found in the documentation at [Using airspeed velocity](https://asv.readthedocs.io/en/stable/using.html).

`asv` is configured using a file `asv.conf.json` located at the root of the PyBaMM repository. See the [asv reference](https://asv.readthedocs.io/en/stable/user_reference.html) for details on available settings and options.

Benchmark results are stored in a directory `results/` at the location of the configuration file. There is one result file per commit, per machine.

### Visualising benchmark results

`asv` is able to generate a static website with a visualisation of the benchmarks results, i.e. the benchmark's duration as a function of the commit hash.
To generate the website, use

```shell
asv publish
```

then, to view the website:

```shell
asv preview
```

Current benchmarks over PyBaMM's history can be viewed at https://pybamm-team.github.io/pybamm-bench/

### Adding benchmarks

To contribute benchmarks to PyBaMM, add a new benchmark function in one of the files in the `benchmarks/` directory.
Benchmarks are distributed across multiple files, grouped by theme. You're welcome to add a new file if none of your benchmarks fit into one of the already existing files.
Inside a benchmark file (e.g. `benchmarks/benchmarks.py`) benchmarks functions are grouped within classes.

Note that benchmark functions _must_ start with the prefix `time_`, for instance

```python3
def time_solve_SPM_ScipySolver(self):
    solver = pb.ScipySolver()
    solver.solve(self.model, [0, 3600])
```

In the case where some setup is necessary, but should not be timed, a `setup` function
can be defined as a method of the relevant class. For example:

```python3
class TimeSPM:
    def setup(self):
        model = pb.lithium_ion.SPM()
        geometry = model.default_geometry

        # ...

        self.model = model

    def time_solve_SPM_ScipySolver(self):
        solver = pb.ScipySolver()
        solver.solve(self.model, [0, 3600])
```

Similarly, a `teardown` method will be run after the benchmark. Note that, unless the `--quick` option is used, benchmarks are executed several times for accuracy, and both the `setup` and `teardown` function are executed before/after each repetition.

Running benchmarks can take a while, and by default encountered exceptions will not be shown. When developing benchmarks, it is often convenient to use the following command instead of `asv run`:

```shell
asv dev
```

`asv dev` implies options `--quick`, `--show-stderr`, and `--dry-run` (to avoid updating the `results` directory).

### Rust observability suite

For branch-local Rust vs CasADi observability, use the lightweight harness instead of
adding another standalone script:

```shell
uv run python benchmarks/run_rust_observability.py --lane artifact
uv run python benchmarks/run_rust_observability.py --lane solver
```

Useful options:

```shell
uv run python benchmarks/run_rust_observability.py --lane artifact --artifact-scenarios toy_expr --repeats 1 --warmup 0
uv run python benchmarks/run_rust_observability.py --lane solver --models SPM DFN --json /tmp/rust-observability.json
uv run python benchmarks/run_rust_observability.py --lane solver --models SPM --output-points 1000
uv run python benchmarks/run_rust_observability.py --lane solver --protocols drive_cycle pulse_train --aot none
uv run python benchmarks/run_rust_observability.py --lane inference --models DFN --inference-seed 7
uv run python benchmarks/run_rust_observability.py --lane solver --reference-tolerance 0
```

### The converged reference

Every row in the solver, sensitivity and inference lanes — **the CasADi baseline
included** — is scored against one converged `casadi_idaklu` solve of the same scenario,
run at `--reference-tolerance` (1e-10 by default) instead of the scenario's 1e-6. The `Δ`
columns are that error. `Base Δ` is the raw difference from `casadi_idaklu` at the
scenario tolerance, reported beside it and gated on nothing: one cell holding whichever
of that row's comparisons came closest to its own tolerance, so on a gradient run it can
be a gradient difference rather than a value one. The per-block breakdown is in the JSON.

Both numbers are needed because they answer different questions. Two backends at the same
tolerance differ by their *mutual* error, which is not either one's accuracy; when they
are the same integrator taking the same steps the shared error cancels outright, so
`rust_idaklu` reads ~1e-13 against `casadi_idaklu` while both sit a full tolerance unit
from the answer. That makes `Base Δ` the most sensitive port-regression detector in the
suite and a useless accuracy check, and the reference the reverse. Measured on SPM
`cc_discharge`: `casadi_idaklu` and `rust_idaklu` are each 1.0e-06 from the reference and
2.3e-13 from each other; `rust_diffsol` is 5.1e-06 from the reference. Before the
reference existed, that last number was read as diffsol being wrong, when most of it is
the baseline's own error.

The gate carries two decades of headroom over the scenario tolerance. A tolerance bounds
the local error of one step and the global error accumulates it over thousands, so a
correct solve lands several tolerance units out: measured across the matrix,
`casadi_idaklu` itself reaches ~26 units on states, `rust_diffsol` ~104. A gate at one
tolerance unit would flag the reference integrator on its own scenarios.

Gradients keep the looser `10 * sqrt(tol)` allowance for the reason in **Inference lane**
below — forward sensitivities are not error-controlled to the state tolerance. The
reference is correspondingly less converged in gradients than in values: SPMe `cc_charge`
improves only 3.5e-03 → 1.0e-04 over four decades of tolerance (the `sqrt(tol)` rate),
and DFN does not improve at all, holding 6.9e-03 at `t = 0` from 1e-6 to 1e-10 while its
value at the same point converges to 2e-10 (see R5 in the issue tracker). A gradient
reference is therefore worth 3-4 decades on SPM and SPMe and about half the gate on DFN,
where a regression below ~1e-02 would not be visible. Values are clear at every model.

A converged solve is not always reachable. DFN under a ramping current fails IDA's error
test at `t = 0` below 1e-9, so the reference tolerance is loosened a decade at a time
until it converges, never past two decades clear of the scenario. DFN `drive_cycle`
therefore reports against a 1e-09 reference and DFN `pulse_train`, which converges at no
usable tolerance, falls back to the baseline comparison with a logged warning. The
tolerance each row was actually judged against is on the row (`reference_tolerance` in
JSON, listed in the caption above each table).

`--reference-tolerance 0` drops the reference entirely and restores the older behaviour:
every candidate compared against `casadi_idaklu` at the scenario tolerance, with the
baseline row itself ungated (`baseline` rather than `pass`/`warn`).

The reference costs one extra untimed solve per scenario in the solver and sensitivity
lanes, and one per draw in the inference lane (~26/160/818 ms for SPM/SPMe/DFN with
gradients). It is solved outside every timed region, so no timing column moves.

Note the reference shares an integrator family with the baseline, so it cannot by itself
catch a systematic IDA error. What would catch one is `rust_diffsol` — an independent
integrator — failing to converge on the same answer as the tolerance tightens.

### Protocols

A scenario is a model crossed with an operating protocol. `--protocols` selects the
protocol axis and defaults to `cc_discharge` alone.

Every protocol layers its current law on one shared base parameter set, `Chen2020`
(`registry.BASE_PARAMETER_SET`), so a row's timings depend only on the model and the
protocol. Chen2020 rather than each model's own defaults because it gives both particle
diffusivities as scalars, which the inference lane needs to swap for inputs like for
like. Note this differs from the pre-protocol suite, which used `model.default_parameter_values`
— solver-lane numbers recorded before that change are not comparable.

| Protocol | What it runs | What it exercises beyond a plain discharge |
| --- | --- | --- |
| `cc_discharge` | 1C discharge for 3600 s | the baseline |
| `cc_charge` | 2C charge from 0% SOC over 1800 s | event termination (~730-1150 s), so the trajectory comparison sees a non-`final time` stop |
| `drive_cycle` | triangle-wave current at 50% SOC | an `Interpolant` node inside the rhs graph, in one continuous solve |
| `pulse_train` | ramped pulse/rest train at 50% SOC | the same graph shape with sharp transitions: step-size rejection and recovery |
| `experiment` | discharge / rest / charge / rest | the `solver.step` path: per-step restarts and per-step initial conditions |

`initial_soc` is applied once at build, never per solve, because passing it to
`Simulation.solve` re-runs `set_initial_state` (and an ElectrodeSOH solve) on every call,
which would swamp the warm timings. The `experiment` protocol takes its output grid from
its step period rather than `--output-points`, so its `Pts` column reads `-` in the solver
and sensitivity lanes; it also builds lazily on first solve, so its `Build` column reads 0
by design.

Only `experiment` restarts the integrator: it solves step by step through
`solver.step`, with fresh initial conditions per step. `pulse_train` is one continuous
solve — a linear `Interpolant` raises no discontinuity events (those come only from
`Heaviside`/`Modulo` nodes in `t`), so its breakpoints are `t_eval` stops that the
stepper adapts through rather than restarts at.

`SENSITIVITY_INPUTS` includes `Current function [A]`, which `drive_cycle` and
`pulse_train` replace with an `Interpolant` and `experiment` supersedes with its own
control. Under those protocols the sensitivity lane differentiates the remaining
parameters only. Which parameters a row actually differentiated is recorded on the row
itself (the `Params` column, and `sensitivity_parameters` in JSON) rather than inferred
from the run's configuration.

### Inference lane

`--lane inference` models a parameter-estimation loop rather than a single solve. Four
parameters — both particle diffusivities and both active-material volume fractions —
stay symbolic as `InputParameter`s, and
**every timed repeat solves with a different input vector**, drawn log-uniformly from
`--inference-seed` within a per-parameter half-width (`registry.INFERENCE_SPREADS`). The
same vectors are shared across backends, so repeat *i* is compared like for like.

The width is per parameter because one figure cannot suit all of them: 20% is routine for
a diffusivity, but the same 20% on an active-material fraction swings porosity between
0.165 and 0.39 about a nominal 0.25, which stalls DFN with `IDA_CONV_FAIL` and drives
SPMe's 2C charge into its voltage ceiling within a couple of output points. Volume
fractions therefore get 5%, holding negative porosity to 0.213-0.285. A parameter added
without a width raises rather than inheriting someone else's.

The inputs are layered *on top of* the selected protocol, so `--lane inference
--protocols drive_cycle` fits against a drive cycle rather than silently reverting to a
constant current. Because active material and pore volume sum to 1 in the base set, each
electrode's porosity is bound to `1 - eps` (`registry.INFERENCE_COMPLEMENTS`) so sampled
geometries stay feasible.

A fitted parameter must also leave the initial state valid. The base set gives the
initial concentration as an absolute value, so a fitted *maximum* concentration moves the
initial stoichiometry rather than the capacity: from a nominal 0.90, a -20% draw puts it
past 1.0 and the cell starts above its voltage cutoff. Maximum concentrations are
therefore excluded — they are not usually fitted quantities either.

Two things differ from the solver lane on purpose. Observation goes through the
interpolating call interface (`solution[var](t)`) at midpoints between solver nodes,
which is the door a fitting loop actually uses, whereas the solver lane deliberately
reads `.data` on the solver's own stored grid. Between them both doors are covered. And
the table splits one-time `Build`/`Setup`/`ColdObs` from the per-evaluation `Eval p50`
with its `p10-p90` spread — under changing inputs that spread is real, not timer noise.

`ColdObs` is the first, forced materialisation of the observed variable, taken before the
warmup loop. Lazy variable compilation costs several milliseconds and would otherwise
either vanish into a discarded warmup repeat or land inside the first timed one; charging
it explicitly keeps the per-evaluation samples steady-state at any `--warmup`.

Both full-state and `output_variables` rows run. They are not duplicates: restricting the
solver to the observed variable changes state storage, the observation path and the
per-evaluation cost substantially (often ~2x). Not every backend supports every
combination — diffsol cannot currently stitch an `output_variables` solution across
experiment steps — and those degrade to an in-row `unsupported`.

Parity is worst-case across every repeat, not just the last, and every backend is read on
the same observation grid. Repeats are ranked by *tolerance-normalised* error, not by
absolute difference: the admissible difference scales with the reference value, so a
repeat that breached tolerance low on the discharge curve must not be masked by one
further up whose larger difference is still permitted. The reference is solved once per
draw and, like every backend, is built from the first input vector: the lane holds `y0`
fixed there, and a reference resolved from a later draw would start the cell at a
different state of charge and move a `cc_charge` event by ~80 s.

Agreement in value is not sufficient. Each repeat also records its termination reason and
final time, and `Cover` reports the fraction of the shared grid the candidate actually
reached, so a backend that stops early cannot pass on a matching prefix. Where the two
terminations disagree, observation points inside that measured window are excluded from
the value comparison — both trajectories are racing to a cutoff at different moments
there, so those points report the endpoint gap rather than trajectory agreement. The
window is measured, not a fixed point count, so it is zero when the terminations coincide
and does not change meaning with `--output-points`.

`--inference-sensitivities` adds forward sensitivities for the gradient-based case, and
the gradient is materialised inside the timed observation, so the chain rule is costed
rather than only the sensitivity integration. Gradients are compared as `p . dV/dp`: the
fitted parameters span eighteen orders of magnitude (a diffusivity at 1e-14 beside a
concentration at 1e4), so raw `d/dp` columns are not comparable under one tolerance, while
the scaled form is dimensionless and is what a log-space fitting loop consumes. They are
judged at the square root of the state tolerance, because a solver error-controls the
state to `(atol, rtol)` but not the sensitivities integrated alongside it; that still
catches the order-of-magnitude miss a broken chain rule produces.

Against the converged reference this lane inverts a conclusion the cross-backend gate used
to reach. On SPMe `cc_charge` with gradients, `casadi_idaklu` and `rust_idaklu` are both
1.8e-01 from the reference while `rust_diffsol` is 4.2e-03 — diffsol's gradients are the
*more* accurate by some forty times, and what the old gate scored as diffsol's error was
mostly the baseline's own. diffsol puts sensitivities under error control with
`sens_atol_factor` tightening the differential-state floor, where IDAS runs them at the
state tolerance.

The initial state is resolved once, from the first input vector, and held fixed while the
fitted parameters vary. That measures compiled-model reuse under a fixed `y0`, which is
the intended workload, and it is why the fitted set is restricted to parameters that do
not define `y0`. A fit that did vary one would have to re-derive `y0` — and pay an
ElectrodeSOH solve — every evaluation, which is a different workload.

The lane never profiles AOT in an isolated cache; the `AOT` column instead reports what
the compiler actually did (`miss`, `disk`, `memory`) so a cheap warm `Setup` is never
misread as a fresh compile. Its `Pts` column counts observation timestamps, not the
`--output-points` request.

### AOT rows

`--aot` controls which lanes run the CasADi ahead-of-time rows: `solver` (the default),
`all`, or `none`. The sensitivity and inference lanes are off by default because their AOT
rows roughly double the run time; turn them on with `--aot all` when the AOT comparison on
those paths is what you are after.

The artifact lane times only each backend's native kernel call (format conversion
is excluded) and uses auto-calibrated inner batching, so `--repeats` counts timed
batches rather than individual calls. All solver backends run at the per-scenario
tolerance for an iso-accuracy comparison.

The suite is intentionally small. It reports parity and timing breakdowns for the
prep-artifact API and representative solver paths. The solver lane always attempts
CasADi IDAKLU, Rust IDAKLU, and Rust diffsol rows, plus separate `output_only` rows
for the observed variable path. Unsupported diffsol configurations are reported in-row
instead of aborting the whole suite. This harness is not a CI gate or a replacement
for ASV.

Console and JSON both carry `reference_tolerance` per row and in the run metadata, so a
saved run records what its Δ columns were measured against.

Solver and sensitivity lanes request 1000 output points by default; use
`--output-points` to measure trajectory scaling explicitly. `Build` covers model
processing and discretisation. `Prep` combines first-solve setup/compilation with
the first forced output materialisation, including lazy observation compilation.
`Cold` is the measured wall clock from the start of model build through that first
materialised result. `Solve` is the backend's internal timer, `Wall` covers the
complete warm `Simulation.solve` call, and `E2E` is a paired warm wall-clock sample
through forced output materialisation. JSON output retains the raw samples and
runtime metadata. Rust rows also report parent colours and dense-row count, entries,
and sweeps. Backend order is randomized reproducibly with `--backend-order-seed`,
including the CasADi baseline: pinning the reference first measures it on a
systematically colder machine than everything it is compared against. The shuffle key
covers model *and* protocol, so protocols of one model no longer share an order. Every
comparison is computed after execution, leaving the numbers independent of the order
things ran in. Repeats within a case are still contiguous, so this reduces
between-backend drift rather than eliminating it; interleaving individual repeats would
need every backend's simulation resident at once. Console and JSON results are reordered
deterministically for comparison: full-state rows then output-only rows, each ordered as
CasADi IDAKLU, CasADi AOT, Rust IDAKLU, and Rust diffsol.

JSON metadata records the commit, whether the tree was dirty, and a digest of the
combined staged and unstaged diff, so two runs from different uncommitted work are
distinguishable. Untracked files fall outside the digest, as they fall outside the diff.

Solver and sensitivity CasADi AOT rows use an isolated empty cache, so their main
`Prep` and `Cold` measurements always include genuine code generation, native
compilation, and library loading. The separate AOT profile verifies every fresh
cache miss, repeats the same case in a new Python process, and reports the
corresponding persistent-cache disk hits, phase timings, disk-cached `Prep`/`Cold`,
and generated library size. A failed compile or unexpected cache state aborts the
row rather than silently reporting the CasADi VM as AOT. This makes solver and
sensitivity runs slower by design, especially for DFN; the warm timing columns
remain the steady-state comparison.

Console output uses the detected terminal width. The solver and sensitivity lanes
keep one full table when it fits, split timing and validation into compact tables on
laptop-sized terminals, and fall back to one wrapped block per backend below 101
columns. JSON output retains every field regardless of the console layout.
