# PyBaMM Serialisation Schema Versions

Tracks schema versions used by `Serialise` (`serialise.py`). On load, `schema_version` is read and migrations applied automatically.

---

## Version 1.2 (February 2026 – current)

**coord_sys moved to geometry dict:**
```json
{
  "negative particle": {
    "symbol_r_n": {"type": "SpatialVariable", "name": "r_n", "domains": {...}},
    "r_n": {"min": {...}, "max": {...}},
    "coord_sys": "spherical polar"
  }
}
```

Also: `ExpressionFunctionParameter` support, `Constant` type in symbol decoder, removal of `mass_matrix_inv`.

---

## Version 1.1 (November 2025 – February 2026)

Major serialisation refactor (PR #5235, #5236, #5244): improved symbol mapping, geometry serialisation, event serialisation, `InputParameter` fixes, optional zlib compression, custom variable observability.

**coord_sys format:** stored as an attribute on `SpatialVariable`:
```json
{
  "negative particle": {
    "symbol_r_n": {"type": "SpatialVariable", "name": "r_n", "coord_sys": "spherical polar", "domains": {...}},
    "r_n": {"min": {...}, "max": {...}}
  }
}
```

---

## Version 1.0 (August 2023 – October 2025)

Initial serialisation format. Files from this era may omit `schema_version`; a missing key is treated as `"1.1"` for migration purposes.

**coord_sys format:** same as 1.1 — stored as an attribute on `SpatialVariable`.

---

## Migration

| File's `schema_version` | Behaviour |
|---|---|
| absent | Treated as `"1.1"` (legacy) |
| `< 1.2` (semver) | `coord_sys` extracted from `SpatialVariable` JSON and injected into geometry dict |
| `"1.2"` | Loaded as-is |
| `> 1.2` | `ValueError` |

The helper `_is_legacy_schema(version)` performs the comparison (`packaging.version.Version(version) < Version("1.2")`).

---

## Adding a new schema version

1. Bump `SUPPORTED_SCHEMA_VERSION` in `serialise.py`.
2. Add a migration block in every `load_*` method.
3. Add a section to this file.
4. Add/update tests in `test_serialisation.py`.
