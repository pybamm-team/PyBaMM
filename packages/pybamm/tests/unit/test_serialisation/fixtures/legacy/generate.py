"""Regenerate the legacy fixtures.

These fixtures pin the PRE-REFACTOR on-disk format, so this script must run
against a pre-kernel checkout (e.g. commit f80dcec9a) -- the legacy writer no
longer exists on main, and running this there writes canonical-format files
that silently defeat the legacy-regression tests. Recipe (from the main repo
root; this copy of the script writes into the main tree's fixture dir):

    git worktree add ../PyBaMM-legacy-fixture f80dcec9a
    echo 'version = "26.5.1.dev3+g50075c64c"' > ../PyBaMM-legacy-fixture/src/pybamm/_version.py
    PYTHONPATH=../PyBaMM-legacy-fixture/src uv run python tests/unit/test_serialisation/fixtures/legacy/generate.py
    git worktree remove --force ../PyBaMM-legacy-fixture

discretised_model.json is ~19 MB and excluded from git; the other files ARE
tracked. compact_symbols.json and submesh_types.json reproduce byte-identically
(hence the pinned version string above, which is embedded in the fixtures). The
discretised fixtures do NOT: the pre-refactor Symbol.set_id hashes the class
object (a per-process memory address), so every run embeds fresh "id" values.
That is harmless -- ids are stripped on load -- but regenerating
discretised_model_coarse.json.gz produces an equally-valid file with a fully
noisy diff, so only regenerate it when the fixture itself must change.
"""

import gzip
import json
import pathlib

import numpy as np

import pybamm
from pybamm.expression_tree.operations.serialise import (
    Serialise,
    convert_symbol_to_json,
)

# Fail fast on a post-refactor checkout: the kernel module IS the refactor that
# removed the legacy writer (a byte-level check below re-validates the output).
try:
    import pybamm.expression_tree.operations.serialise_kernel
except ImportError:
    pass  # pre-kernel checkout: the legacy writer exists, carry on
else:
    raise SystemExit(
        "running against post-refactor pybamm: this would write canonical-format "
        "fixtures that defeat the legacy-regression tests (see module docstring)"
    )

out = pathlib.Path(__file__).parent
out.mkdir(parents=True, exist_ok=True)

# 1. Compact-symbol format
neg = -pybamm.Variable("u", domains={"primary": ["negative electrode"]})
interp = pybamm.Interpolant(
    np.linspace(0.0, 1.0, 4), np.array([0.0, 1.0, 0.5, 0.2]), pybamm.Variable("a")
)
var = pybamm.Variable("v", bounds=(pybamm.Scalar(0.0), pybamm.Scalar(5.0)))
fp = pybamm.FunctionParameter("k", {"T": pybamm.Variable("T")})
compact_cases = {
    "addition_broadcast_time": pybamm.PrimaryBroadcast(
        pybamm.Scalar(2.0), "negative electrode"
    )
    + pybamm.Time(),
    "negate_generic_fallback": neg,
    "interpolant_entries_string": interp,
    "variable_with_bounds": var,
    "function_parameter": fp,
}
(out / "compact_symbols.json").write_text(
    json.dumps(
        {k: convert_symbol_to_json(v) for k, v in compact_cases.items()}, indent=2
    )
)
print("wrote compact_symbols.json")

# 2. Discretised-model format (~19 MB, git-ignored)
model = pybamm.lithium_ion.SPM()
sim = pybamm.Simulation(model)
sim.build()
model_json = Serialise().serialise_model(sim.built_model, mesh=sim.mesh)
serialised = json.dumps(model_json, indent=2)
# Validate the bytes are the pre-refactor format the fixtures exist to pin.
if "py/object" not in serialised or '"$type"' in serialised:
    raise SystemExit("wrote canonical-format output, not the legacy format")
(out / "discretised_model.json").write_text(serialised)
print("wrote discretised_model.json")

# 2b. Coarse-mesh variant: same format, small enough to track in git, so CI
# always exercises this path. gzip mtime=0 keeps the header timestamp-free
# (the payload still varies per run -- see module docstring).
coarse = pybamm.Simulation(
    pybamm.lithium_ion.SPM(),
    var_pts=dict.fromkeys(model.default_var_pts, 5),
)
coarse.build()
coarse_json = Serialise().serialise_model(coarse.built_model, mesh=coarse.mesh)
(out / "discretised_model_coarse.json.gz").write_bytes(
    gzip.compress(json.dumps(coarse_json).encode(), mtime=0)
)
print("wrote discretised_model_coarse.json.gz")

# 3. Submesh-types format
(out / "submesh_types.json").write_text(
    json.dumps(Serialise.serialise_submesh_types(sim.submesh_types), indent=2)
)
print("wrote submesh_types.json")
print("done")
