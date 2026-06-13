# TODO

## Mesh / spatial methods

### Extend `UserSuppliedUnstructuredMesh` for arbitrary domain names
**Why:** current `_domain_name_from_lims` (`unstructured_submesh.py:671-688`) only
recognises hardcoded prefixes (`x_n`, `x_s`, `x_p`) and maps to fixed battery
domains. For user-defined domain names (`body`, `tab_0`, ...), it returns `None`
and falls through to "use all cells" — every region's submesh ends up holding
the entire mesh. Users currently work around this with a custom
`MeshGenerator` subclass (see
`scripts/mesh/run_thermal_3d_multi_domain.py::TaggedSubMeshGenerator`).

**Change:** add an explicit `tag_id` (or `domain_tag`) kwarg that bypasses the
name-prefix heuristic.

```python
gen = pybamm.UserSuppliedUnstructuredMesh(
    filepath="cell.msh",
    tag_id=1,  # filter cells by gmsh:physical == 1
    coord_sys="cartesian",
)
```

Implementation sketch (`__call__`):
```python
if self.tag_id is not None:
    cell_mask = self._get_cell_mask(mesh, cell_type, self.tag_id)
    elements = cells[cell_mask]
elif domain_name and domain_name in self.subdomain_mapping:
    # existing path
    ...
```

**Bonus:** module-level LRU cache keyed by `filepath` so multiple instances
reading the same `.msh` don't re-parse it (currently each instance has its own
`_cached_mesh` field, so once you have N region generators you do N reads).

**Effort:** ~15 lines + a unit test that loads a multi-tag mesh into separate
domains and checks each submesh has only its tag's cells.
