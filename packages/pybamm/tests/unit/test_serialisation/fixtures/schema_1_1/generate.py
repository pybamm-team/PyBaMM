"""Regenerate the schema 1.1 fixtures.

These pin files written by the last PyBaMM release before symbolic models moved
to the flat node table (schema 2.0), so they must be written by that release
and not by this checkout. Recipe (from any directory; needs no checkout):

    uv run --no-project --with pybamm==26.10.1.1 python \
        packages/pybamm/tests/unit/test_serialisation/fixtures/schema_1_1/generate.py

Each file is the bytes produced by the release's own writer. The model files
are 12 MB and 41 MB as written, so they are stored xz-compressed.
"""

import lzma
import pathlib
import tempfile

import pybamm
from pybamm.expression_tree.operations.serialise import Serialise

RELEASE = "26.10.1.1"
if pybamm.__version__ != RELEASE:
    raise SystemExit(
        f"running against pybamm {pybamm.__version__}: these fixtures must be "
        f"written by the {RELEASE} release (see module docstring)"
    )

out = pathlib.Path(__file__).parent
model = pybamm.lithium_ion.SPM()
writers = {
    "spm": pybamm.lithium_ion.SPM().to_json,
    "dfn": pybamm.lithium_ion.DFN().to_json,
    "geometry": lambda f: Serialise.save_custom_geometry(model.default_geometry, f),
    "spatial_methods": lambda f: Serialise.save_spatial_methods(
        model.default_spatial_methods, f
    ),
    "var_pts": lambda f: Serialise.save_var_pts(model.default_var_pts, f),
    "submesh_types": lambda f: Serialise.save_submesh_types(
        model.default_submesh_types, f
    ),
}
with tempfile.TemporaryDirectory() as tmp:
    for name, write in writers.items():
        written = pathlib.Path(tmp) / f"{name}.json"
        write(str(written))
        content = written.read_bytes()
        if name in ("spm", "dfn"):
            target = out / f"{name}.json.xz"
            target.write_bytes(lzma.compress(content, preset=9 | lzma.PRESET_EXTREME))
        else:
            target = out / f"{name}.json"
            target.write_bytes(content)
        print(f"wrote {target.name} ({target.stat().st_size} bytes)")
