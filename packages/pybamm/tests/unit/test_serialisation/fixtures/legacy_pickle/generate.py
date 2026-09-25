"""Regenerate the legacy pickle fixtures; run against a pre-slot checkout (e034ac50e):

    git worktree add ../PyBaMM-legacy-pickle e034ac50e
    PYTHONPATH=../PyBaMM-legacy-pickle/packages/pybamm/src uv run python \
        packages/pybamm/tests/unit/test_serialisation/fixtures/legacy_pickle/generate.py
    git worktree remove --force ../PyBaMM-legacy-pickle
"""

import gzip
import json
import pathlib
import pickle

import numpy as np

import pybamm
from pybamm.expression_tree.operations.serialise import convert_symbol_to_json

HERE = pathlib.Path(__file__).parent
TIMES = np.linspace(0, 3600, 13)


def symbol_cases():
    cases = {}
    for model in (
        pybamm.lithium_ion.SPM(),
        pybamm.lithium_ion.DFN(),
        pybamm.lead_acid.Full(),
    ):
        sim = pybamm.Simulation(model)
        sim.build()
        for source, discretised in ((model, False), (sim.built_model, True)):
            expressions = [
                *source.rhs.values(),
                *source.algebraic.values(),
                *source.initial_conditions.values(),
                *source.variables.values(),
            ]
            for expression in expressions:
                for symbol in expression.pre_order():
                    key = (type(symbol).__qualname__, discretised)
                    if key not in cases:
                        cases[key] = {
                            "symbol": symbol,
                            # machine-independent: floats are written exactly
                            "json": json.dumps(convert_symbol_to_json(symbol)),
                            "str": str(symbol),
                            "domains": {k: list(v) for k, v in symbol.domains.items()},
                        }
    return cases


if __name__ == "__main__":
    with gzip.open(HERE / "symbols.pkl.gz", "wb") as f:
        pickle.dump(symbol_cases(), f)
    sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
    solution = sim.solve([0, 3600])
    with gzip.open(HERE / "spm_simulation.pkl.gz", "wb") as f:
        pickle.dump(sim, f)
    np.save(HERE / "spm_voltage.npy", solution["Voltage [V]"](TIMES))
