"""Regenerate the model graphs that the pybamm-core benchmarks and tests load.

Usage::

    uv run python scripts/snapshot_model_dags.py [OUTPUT_DIR]

Writes ``spm.bin``, ``spme.bin`` and ``dfn.bin`` to ``OUTPUT_DIR``, by default
``packages/pybamm-rust/pybamm-core/benches/fixtures``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pybamm
from pybamm.rust import ExprGraph

FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "pybamm-rust"
    / "pybamm-core"
    / "benches"
    / "fixtures"
)
MODELS = {
    "spm": pybamm.lithium_ion.SPM,
    "spme": pybamm.lithium_ion.SPMe,
    "dfn": pybamm.lithium_ion.DFN,
}


def discretise(model: pybamm.BaseModel) -> pybamm.BaseModel:
    """Parameterise ``model`` with Chen2020 and discretise it in place, without events."""
    model.events = []
    parameter_values = pybamm.ParameterValues("Chen2020")
    geometry = model.default_geometry
    parameter_values.process_model(model)
    parameter_values.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    pybamm.Discretisation(mesh, model.default_spatial_methods).process_model(model)
    return model


def snapshot(model: pybamm.BaseModel, name: str, path: Path) -> None:
    """Write the residual of the discretised ``model`` to ``path``."""
    residual = model.concatenated_rhs
    if model.len_alg > 0:
        residual = pybamm.numpy_concatenation(residual, model.concatenated_algebraic)
    graph = ExprGraph()
    graph.dump_dag(residual.to_rust(graph), path, name, n_states=model.len_rhs_and_alg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", nargs="?", type=Path, default=FIXTURES)
    output_dir = parser.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pybamm.set_logging_level("INFO")
    for name, model_class in MODELS.items():
        path = output_dir / f"{name}.bin"
        snapshot(discretise(model_class()), name, path)
        pybamm.logger.info(f"{name}: {path.stat().st_size / 1024:.1f} KB -> {path}")


if __name__ == "__main__":
    main()
