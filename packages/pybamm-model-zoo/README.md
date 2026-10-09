# PyBaMM model zoo

Community- and partner-contributed PyBaMM models. One folder per model, holding
its code, tests, examples, citation, and a declarative `model.toml` manifest.

The manifest is the only boilerplate you write. The registry, the contract test
suite, the docs page, the CI routing, and the status badge are all derived from
it. Manifests are parsed, never imported, so a model whose dependencies are
missing still appears in the registry and the docs and reports a clean failure
rather than taking the zoo down.

## Installation

The zoo is released to PyPI independently of PyBaMM, so `pip install pybamm`
does not provide it:

```bash
pip install pybamm-model-zoo
```

A model with third-party dependencies of its own declares them as an extra named
`zoo-<slug>`, and the `zoo-all` extra installs every model's:

```bash
pip install "pybamm-model-zoo[zoo-all]"
```

Every model supports the PyBaMM releases the zoo's own `pybamm` requirement
admits, so `pip` only installs a PyBaMM they all work with.

In a PyBaMM checkout the zoo is a `uv` workspace member, so
`uv sync --extra all --group dev` from the repository root installs it editable
alongside `pybamm`.

## Usage

```python
import pybamm
import pybamm_model_zoo as zoo

zoo.list_models()
entry = zoo.info("LinearisedSPM")
entry.tier, entry.maintainers

model = zoo.load("LinearisedSPM")()
solution = pybamm.Simulation(model).solve([0, 300])
print(solution["Voltage [V]"](150))
```

Using a model registers its citation, so `pybamm.print_citations()` credits its
authors.

## Adding a model

> [!IMPORTANT]
> Before adding a model, please read the [model zoo policy](https://github.com/pybamm-team/PyBaMM/blob/main/packages/pybamm-model-zoo/POLICY.md). By opening a pull request that adds or changes a model in the zoo, you agree to the policy.

```bash
nox -s zoo-new -- --slug my_model --name MyModel --author "A. Author" --github ahandle
```

That renders [`template/`](https://github.com/pybamm-team/PyBaMM/tree/main/packages/pybamm-model-zoo/template) into
`src/pybamm_model_zoo/my_model/` and appends your `.github/CODEOWNERS` line, so
changes to your folder request you as reviewer. The rendered skeleton passes the
whole contract suite as generated — the zoo's own test suite renders the template
and runs every check against the result, so that is a tested claim rather than a
hope.

Then:

1. Replace the TODOs in `model.toml`, `README.md`, and `CITATION.bib`.
2. Put your physics in `model.py`.
3. Write at least one test that pins a **physical** result — a known limit, an
   analytic solution, a conservation check, or a published figure. That the model
   merely runs is already covered by the contract suite, so a test that only
   checks it runs adds nothing.
4. Regenerate the docs: `nox -s zoo-docs`.
5. Run it: `nox -s zoo -- --zoo-model=my_model`.
6. Add a bullet to this package's `CHANGELOG.md`. Zoo pull requests never touch
   PyBaMM's changelog.

[`linearised_spm/`](https://github.com/pybamm-team/PyBaMM/tree/main/packages/pybamm-model-zoo/src/pybamm_model_zoo/linearised_spm)
is the reference entry: the smallest thing that is still a real model. Copy that
folder rather than reading prose.

### Dependencies

A model's dependencies are declared in one place: a per-model **optional** extra
named `zoo-<slug>` (dashes, not underscores) in this package's `pyproject.toml`,
listed in the `zoo-all` aggregate — never in the base dependencies. Workspace
members share one lockfile and one dev venv, so a base dependency here lands in
every contributor's environment.

For a PyBaMM feature with optional dependencies of its own, depend on PyBaMM's
extra rather than the packages behind it, so their versions stay PyBaMM's to pin:

```toml
zoo-my-model = ["pybamm[fem]"]
```

The registry reads the extra from the installed package metadata. `zoo.load()`
checks it before importing your model, and the template's
`pybamm_model_zoo.require(SLUG)` checks it when the model is created, so a
missing extra fails straight away with the `pip install` that fixes it, however
the model was imported. Keep that call first in your `__init__`.

The aggregate is deliberately not called `all`: `uv sync --extra all` applies the
extra to every workspace member that defines one, so an `all` extra here would
drag every model's heavy dependencies into the core dev environment.

### PyBaMM versions

Every model shares the zoo's one PyBaMM floor, its `pybamm>=` requirement in
`pyproject.toml`. If your model needs a newer PyBaMM, raise the floor in the same
pull request. The release workflow builds and solves every model against the
floor before publishing.

## Tiers

| Tier | Who sets it | CI |
| --- | --- | --- |
| `community` | the default for a new model | Runs on every pull request, **advisory**: reports red without blocking a merge. |
| `core` | PyBaMM maintainers, by adopting a model | Runs in PyBaMM's merge gate. A failure blocks a merge, in core and in the zoo. |

Advisory-ness lives in the CI job, never in an `xfail` marker: a community model
that fails, fails, and that is exactly the signal its badge reports. Being
advisory is what lets the zoo test contributed models on every pull request
without a contributed model ever blocking a PyBaMM release. The manifest tier is
what puts a model's tests behind the `gating` marker, and what `--zoo-tier=core`
keeps out of the merge gate's collection entirely.

## Testing

Three layers; only the middle one is yours.

| Layer | What | Where |
| --- | --- | --- |
| A | Contract suite — every check below, against every registered model, automatically | `tests/test_contract.py`, checks in `pybamm_model_zoo.testing.contract` |
| B | Your physics tests | `src/pybamm_model_zoo/<slug>/tests/` |
| C | Your example scripts, executed | `tests/test_examples.py` over `<slug>/examples/*.py` |

### The contract

`pybamm_model_zoo.testing.contract.CHECKS` is the one definition of the contract:
the test suite, the manifest's `skip_contract` validation, and this table all
derive from it. Each check has a **scope**, which is what lets the same module
hold an in-tree model and a third-party collection to the right rules.

| Check | Scope | What it asserts |
| --- | --- | --- |
| `manifest` | model | Schema valid; `slug` matches the folder name; `class` is parseable; at least one maintainer. |
| `layout` | model | `README.md`, `CITATION.bib`, `examples/`, and `tests/` present, and the README has `Summary`, `Usage`, `Validation`, and `Citation` sections. |
| `import` | model | The declared class imports and subclasses `pybamm.BaseModel`. Skipped with a reason when your `zoo-<slug>` extra is not installed. |
| `citation` | model | The manifest's citation key resolves in your `CITATION.bib`, and instantiating the model registers it, so `pybamm.print_citations()` credits you. |
| `well_posed` | model | `model.check_well_posedness()` passes. |
| `build` | model | `pybamm.Simulation(model).build()` succeeds. |
| `solve` | model | The model solves for the manifest's `solve_time`, and every `key_variables` entry is finite read through the interpolating call interface. |
| `packaging` | packaging | An in-tree model is importable as `pybamm_model_zoo.<slug>`, and `zoo-all` includes its extra if it has one. |
| `license` | packaging | An in-tree model's manifest says `BSD-3-Clause`, the zoo package's license. Cannot be waived. |
| `docs` | repo | The generated docs page is present *and current*, so docs cannot drift from code. |
| `codeowners` | repo | `.github/CODEOWNERS` names an owner for your folder, so ownership cannot be dropped silently. |

A third-party collection is held only to the `model` scope; it is not wired into
this package and does not live in this repository.

`skip_contract` waives an individual check for a genuinely unusual model — one
with no meaningful standalone solve, say. It is per-check, reviewed, and visible
in the manifest diff. The `manifest` and `license` checks cannot be waived.

```bash
nox -s zoo                          # the whole zoo, both tiers, plus examples
nox -s zoo -- --zoo-model=my_model  # one model
nox -s zoo-gating                   # only core-tier models (the merge gate)
nox -s zoo-community                # only what the gate does not cover (advisory CI)
nox -s zoo-examples                 # every model's example scripts
nox -s zoo-docs                     # regenerate docs pages and badges
```

`--zoo-model` and `--zoo-tier` drop the folders they exclude *before* pytest
imports anything in them, so a model that fails to import can only fail its own
run. That is what keeps the merge gate independent of the community tier: marker
deselection alone would not, since it happens after every test module is loaded.

Aim for contract checks under 60 s per model and a model's own tests under
5 minutes; `solve_time` in the manifest is the lever.

## External model collections

A third-party package can register its own directory of manifests and be
discovered by the same registry:

```toml
[project.entry-points."pybamm_zoo_models"]
my_lab_models = "my_lab_models"
```

It can also hold itself to the `model`-scope contract in its own CI, since the
checks ship in `pybamm_model_zoo.testing.contract`. Its models' extras follow the
same `zoo-<slug>` naming in the collection's own distribution, and its own
`pybamm` requirement is its PyBaMM floor. In-tree models win a name
collision, so a third-party package cannot shadow one.

## Review checklist for a zoo pull request

Short by design — the contract suite does the rest.

- [ ] You have read the [model zoo policy](https://github.com/pybamm-team/PyBaMM/blob/main/packages/pybamm-model-zoo/POLICY.md).
- [ ] License is BSD-3-Clause.
- [ ] No edits to `packages/pybamm/src/` in the same pull request; core changes
      are split out.
- [ ] Dependencies declared only as a `zoo-<slug>` extra, using PyBaMM's own
      extras where they exist.
- [ ] At least one test pins a physical result.
- [ ] The example script runs.
- [ ] The citation resolves and is registered on instantiation.
- [ ] `.github/CODEOWNERS` line added.
- [ ] `CHANGELOG.md` entry in this package.
