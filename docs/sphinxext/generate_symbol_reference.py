"""Sphinx extension: regenerate the parameter symbol-reference page on build.

For every parameter class in :mod:`pybamm.parameters` this walks the live
object graph and writes ``docs/source/api/parameters/symbol_reference.rst``,
which maps each Python-side shorthand (e.g. ``param.kappa_e``) to the
underlying string used in :class:`pybamm.ParameterValues`
(e.g. ``"Electrolyte conductivity [S.m-1]"``).

The output file is gitignored — it is rebuilt from the current source on
every Sphinx run, so adding a new shorthand to ``src/pybamm/parameters/``
automatically updates the reference page.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from pathlib import Path

import pybamm
from pybamm.parameters.base_parameters import BaseParameters

# Roots to introspect. Each entry is (label, root, factory) where ``factory``
# returns a freshly constructed instance and ``root`` is the suggested
# variable name users would bind it to.
ROOTS: list[tuple[str, str, callable]] = [
    ("LithiumIonParameters", "param", lambda: pybamm.LithiumIonParameters()),
    ("LeadAcidParameters", "param", lambda: pybamm.LeadAcidParameters()),
    ("GeometricParameters", "geo", lambda: pybamm.GeometricParameters()),
    ("ElectricalParameters", "elec", lambda: pybamm.electrical_parameters),
    ("ThermalParameters", "therm", lambda: pybamm.thermal_parameters),
    ("EcmParameters", "param", lambda: pybamm.EcmParameters()),
]

# Wiring/back-references that don't carry useful info for end users.
SKIP_ATTRS = {
    "options",
    "_options",
    "main_param",
    "domain_params",
    "phase_params",
    "domain",
    "_domain",
    "_Domain",
    "domain_Domain",
    "phase",
    "phase_name",
    "phase_prefactor",
    "geo",
    "therm",
    "elec",
}

# Placeholders for method parameters when we need to call a method to discover
# its underlying FunctionParameter name. Anything we don't have a placeholder
# for (and which has no default) is skipped.
PLACEHOLDERS = {
    "c_e": pybamm.Variable("c_e"),
    "c_s": pybamm.Variable("c_s"),
    "c_s_surf": pybamm.Variable("c_s_surf"),
    "c_Li": pybamm.Variable("c_Li"),
    "c_ox": pybamm.Variable("c_ox"),
    "c_hy": pybamm.Variable("c_hy"),
    "T": pybamm.Variable("T"),
    "T_cell": pybamm.Variable("T_cell"),
    "y": pybamm.Variable("y"),
    "z": pybamm.Variable("z"),
    "t": pybamm.Variable("t"),
    "R": pybamm.Variable("R"),
    "sto": pybamm.Variable("sto"),
    "soc": pybamm.Variable("soc"),
    "U": pybamm.Variable("U"),
    "ocv": pybamm.Variable("ocv"),
    "L_sei": pybamm.Variable("L_sei"),
    "i": pybamm.Variable("i"),
    "current": pybamm.Variable("current"),
    "index": 0,
    "element_number": 0,
    "lithiation": None,
    "direction": None,
    "name": "Element-0 resistance [Ohm]",
}


def _call_method(method):
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return None
    args = []
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if name in PLACEHOLDERS:
            args.append(PLACEHOLDERS[name])
        elif p.default is not inspect.Parameter.empty:
            args.append(p.default)
        else:
            return None
    try:
        return method(*args)
    except Exception:
        return None


def _collect_param_names(expr):
    if isinstance(expr, pybamm.FunctionParameter):
        return [("FunctionParameter", expr.name)]
    if isinstance(expr, pybamm.Parameter):
        return [("Parameter", expr.name)]
    if not isinstance(expr, pybamm.Symbol):
        return []
    seen = set()
    out = []
    for sub in expr.pre_order():
        if isinstance(sub, pybamm.FunctionParameter):
            key = ("FunctionParameter", sub.name)
        elif isinstance(sub, pybamm.Parameter):
            key = ("Parameter", sub.name)
        else:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _walk(obj, path, results, seen):
    if id(obj) in seen:
        return
    seen.add(id(obj))

    for name in sorted(dir(obj)):
        if name.startswith("_") or name in SKIP_ATTRS:
            continue
        cls_attr = inspect.getattr_static(type(obj), name, None)
        if isinstance(cls_attr, (classmethod, staticmethod)):
            continue
        try:
            v = getattr(obj, name)
        except Exception:
            continue
        full = f"{path}.{name}"

        if isinstance(v, pybamm.FunctionParameter):
            results.append((path, full, "FunctionParameter", v.name))
            continue
        if isinstance(v, pybamm.Parameter):
            results.append((path, full, "Parameter", v.name))
            continue
        if isinstance(v, pybamm.Symbol):
            for kind, pname in _collect_param_names(v):
                results.append((path, full, f"derived ({kind})", pname))
            continue
        if isinstance(v, BaseParameters):
            _walk(v, full, results, seen)
            continue
        if inspect.ismethod(v) or (callable(v) and not inspect.isclass(v)):
            result = _call_method(v)
            if result is None:
                continue
            param_names = _collect_param_names(result)
            if not param_names:
                continue
            try:
                sig = inspect.signature(v)
                arg_names = [n for n in sig.parameters if n != "self"]
                call = f"{full}({', '.join(arg_names)})"
            except (TypeError, ValueError):
                call = f"{full}(...)"
            for _, pname in param_names:
                results.append((path, call, "FunctionParameter", pname))


def _emit_table(rows: Iterable[tuple[str, str, str]]) -> str:
    out = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 35 25 40",
        "",
        "   * - Shorthand",
        "     - Kind",
        "     - Parameter string",
    ]
    for shorthand, kind, pname_cell in rows:
        out.append(f"   * - ``{shorthand}``")
        out.append(f"     - {kind}")
        out.append(f"     - {pname_cell}")
    out.append("")
    return "\n".join(out)


def _render_class(label: str, root: str, results) -> str:
    # Bucket rows by their group (i.e. the access path of the closest
    # enclosing container). Within a group, collapse multiple constituent
    # parameter strings for a single derived shorthand into one row so the
    # tables don't get repetitive.
    grouped: dict[str, dict[tuple[str, str], list[str]]] = {}
    for group, shorthand, kind, pname in results:
        bucket = grouped.setdefault(group, {})
        key = (shorthand, kind)
        if pname not in bucket.setdefault(key, []):
            bucket[key].append(pname)

    collapsed: dict[str, list[tuple[str, str, str]]] = {}
    for group, bucket in grouped.items():
        rows = []
        for (shorthand, kind), pnames in bucket.items():
            rows.append((shorthand, kind, ", ".join(f"``{p}``" for p in pnames)))
        collapsed[group] = sorted(rows, key=lambda r: r[0])
    grouped = collapsed

    out: list[str] = []
    out.append(label)
    out.append("-" * len(label))
    out.append("")
    out.append(
        f"Instantiate as ``{root} = pybamm.{label}()``. "
        f"The tables below list each Python-side shorthand and the "
        f"underlying string used in :class:`pybamm.ParameterValues`."
    )
    out.append("")

    if root in grouped:
        out.append(f"Top-level — ``{root}.<name>``")
        out.append("^" * 60)
        out.append("")
        out.append(_emit_table(grouped[root]))

    for group in sorted(g for g in grouped if g != root):
        out.append(f"``{group}``")
        out.append("^" * 60)
        out.append("")
        out.append(_emit_table(grouped[group]))

    return "\n".join(out)


def _build_page() -> str:
    parts: list[str] = []
    parts.append(".. This file is generated by")
    parts.append(".. docs/sphinxext/generate_symbol_reference.py — do not edit by hand.")
    parts.append("")
    parts.append("Symbol reference")
    parts.append("================")
    parts.append("")
    parts.append(
        "This page maps each Python-side parameter shorthand (e.g. "
        "``param.kappa_e``) onto the string used in "
        ":class:`pybamm.ParameterValues` and in parameter-set files "
        '(e.g. ``"Electrolyte conductivity [S.m-1]"``). It is regenerated '
        "from the parameter classes in ``src/pybamm/parameters/`` on every "
        "documentation build."
    )
    parts.append("")
    parts.append(
        "``Kind`` is ``Parameter`` for scalars set directly via the parameter "
        "values dictionary, ``FunctionParameter`` for entries that may be set "
        "to a Python callable (e.g. a function of concentration and "
        "temperature), and ``derived (...)`` for shorthands defined as "
        "expressions over other parameters (only their constituent parameter "
        "strings are listed)."
    )
    parts.append("")

    for label, root, factory in ROOTS:
        try:
            instance = factory()
        except Exception as exc:  # pragma: no cover - reported in build log
            parts.append(f".. note:: Skipped ``{label}``: {exc}")
            parts.append("")
            continue
        results: list[tuple[str, str, str, str]] = []
        _walk(instance, root, results, set())
        parts.append(_render_class(label, root, results))
        parts.append("")

    return "\n".join(parts) + "\n"


def _output_path(app) -> Path:
    return Path(app.srcdir) / "api" / "parameters" / "symbol_reference.rst"


def _on_builder_inited(app) -> None:
    out_path = _output_path(app)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_build_page())


def setup(app):
    app.connect("builder-inited", _on_builder_inited)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
