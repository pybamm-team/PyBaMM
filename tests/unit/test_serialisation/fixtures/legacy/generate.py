"""Regenerate the legacy fixtures (run from repo root with uv run python <this file>).

discretised_model.json is ~19 MB and excluded from git; the other two files ARE
tracked. Re-run this whenever the serialisation format changes so the pinned
fixtures stay in sync.
"""

import json
import pathlib

import numpy as np

import pybamm
from pybamm.expression_tree.operations.serialise import (
    Serialise,
    convert_symbol_to_json,
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
(out / "discretised_model.json").write_text(json.dumps(model_json, indent=2))
print("wrote discretised_model.json")

# 3. Submesh-types format
(out / "submesh_types.json").write_text(
    json.dumps(Serialise.serialise_submesh_types(sim.submesh_types), indent=2)
)
print("wrote submesh_types.json")
print("done")
