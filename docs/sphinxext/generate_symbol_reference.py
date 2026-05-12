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
import re
from collections.abc import Iterable
from pathlib import Path

import pybamm
from pybamm.parameters.base_parameters import BaseParameters

_FLOAT_RE = re.compile(r"\d+\.\d+(?:[eE][+-]?\d+)?")


def _prettify_floats(text: str) -> str:
    """Rewrite float literals that are exact reciprocals of small integers
    (e.g. ``0.0002777777777777778`` → ``(1/3600)``).

    PyBaMM simplifies expressions like ``F / 3600`` into ``0.000277… * F`` at
    construction time, which is ugly in docs. Most of the offenders are
    unit-conversion factors (``1/60``, ``1/3600``, ``1/86400``), which are
    well-handled by this pass.
    """

    def _sub(match: re.Match[str]) -> str:
        try:
            f = float(match.group(0))
        except ValueError:
            return match.group(0)
        if not (0 < f < 1):
            return match.group(0)
        n = round(1 / f)
        if n > 1 and abs(1 / n - f) / f < 1e-12:
            return f"(1/{n})"
        return match.group(0)

    return _FLOAT_RE.sub(_sub, text)

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

def _call_method(method):
    """Call ``method`` with placeholder arguments to discover what it returns.

    First tries ``pybamm.Variable(arg_name)`` for every positional parameter
    (works for almost everything, since methods on parameter classes accept
    pybamm expressions). If that fails — e.g. the method indexes by an integer
    or treats an arg as a string — retries with each argument's default value
    where available, or ``0`` for the rest.
    """
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return None
    params = [(n, p) for n, p in sig.parameters.items() if n != "self"]

    def _try(args):
        try:
            return method(*args)
        except Exception:
            return None

    result = _try([pybamm.Variable(n) for n, _ in params])
    if result is not None:
        return result
    fallback = [
        p.default if p.default is not inspect.Parameter.empty else 0
        for _, p in params
    ]
    return _try(fallback)


def _classify(expr):
    """Return ``(kind, detail)`` for a pybamm expression discovered on a
    parameter class.

    - ``("Parameter", name)`` for a bare :class:`pybamm.Parameter`.
    - ``("FunctionParameter", name)`` for a bare :class:`pybamm.FunctionParameter`.
    - ``("derived", str(expr))`` for any composite expression.
    - ``None`` if it's a non-Symbol or carries no parameters worth listing.
    """
    if isinstance(expr, pybamm.FunctionParameter):
        return ("FunctionParameter", expr.name)
    if isinstance(expr, pybamm.Parameter):
        return ("Parameter", expr.name)
    if not isinstance(expr, pybamm.Symbol):
        return None
    # Composite expression — only worth showing if it actually contains a
    # parameter somewhere in the tree (otherwise it's a numerical constant).
    has_param = any(
        isinstance(sub, (pybamm.Parameter, pybamm.FunctionParameter))
        for sub in expr.pre_order()
    )
    if not has_param:
        return None
    return ("derived", _prettify_floats(str(expr)))


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

        if isinstance(v, BaseParameters):
            _walk(v, full, results, seen)
            continue

        if isinstance(v, pybamm.Symbol):
            classified = _classify(v)
            if classified is not None:
                kind, detail = classified
                results.append((path, full, kind, detail))
            continue

        if inspect.ismethod(v) or (callable(v) and not inspect.isclass(v)):
            result = _call_method(v)
            if result is None:
                continue
            classified = _classify(result)
            if classified is None:
                continue
            try:
                sig = inspect.signature(v)
                arg_names = [n for n in sig.parameters if n != "self"]
                call = f"{full}({', '.join(arg_names)})"
            except (TypeError, ValueError):
                call = f"{full}(...)"
            kind, detail = classified
            results.append((path, call, kind, detail))


def _emit_table(rows: Iterable[tuple[str, str, str]]) -> str:
    out = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 30 20 50",
        "",
        "   * - Shorthand",
        "     - Kind",
        "     - Parameter string / expression",
    ]
    for shorthand, kind, detail in rows:
        out.append(f"   * - ``{shorthand}``")
        out.append(f"     - {kind}")
        # Wrap the detail in inline code so the parameter strings (which
        # contain spaces, brackets and units) render cleanly. Replace any
        # backticks in the expression to avoid breaking inline-code parsing.
        safe = detail.replace("``", "'")
        out.append(f"     - ``{safe}``")
    out.append("")
    return "\n".join(out)


def _render_class(label: str, root: str, results) -> str:
    # Bucket rows by their group (the access path of the closest enclosing
    # container) and dedupe.
    grouped: dict[str, list[tuple[str, str, str]]] = {}
    for group, shorthand, kind, detail in results:
        rows = grouped.setdefault(group, [])
        row = (shorthand, kind, detail)
        if row not in rows:
            rows.append(row)
    for g in grouped:
        grouped[g] = sorted(grouped[g], key=lambda r: r[0])

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
    parts.append(
        ".. docs/sphinxext/generate_symbol_reference.py — do not edit by hand."
    )
    parts.append("")
    parts.append("Symbol reference")
    parts.append("================")
    parts.append("")
    parts.append(
        "This page maps each Python-side parameter shorthand (e.g. "
        "``param.kappa_e``) onto the string used in "
        ":class:`pybamm.ParameterValues` and in parameter-set files "
        '(e.g. ``"Electrolyte conductivity [S.m-1]"``).'
    )
    parts.append("")
    parts.append(
        "``Kind`` is ``Parameter`` for scalars set directly via the parameter "
        "values dictionary, ``FunctionParameter`` for entries that may be set "
        "to a Python callable (e.g. a function of concentration and "
        "temperature), and ``derived`` for shorthands defined as expressions "
        "over other parameters — the third column then shows the full "
        "expression."
    )
    parts.append("")
    parts.append("Domain and phase sub-objects")
    parts.append("----------------------------")
    parts.append("")
    parts.append(
        "Each electrochemical parameter class exposes domain sub-objects "
        "``.n`` (negative electrode), ``.s`` (separator) and ``.p`` "
        "(positive electrode). The electrode domains additionally carry "
        "particle-phase sub-objects ``.prim`` (primary phase) and ``.sec`` "
        "(secondary phase). So ``param.n.prim.D(c_s, T)`` is the primary-"
        "phase particle diffusivity in the negative electrode, and "
        "``param.p.sec.U(sto, T)`` is the secondary-phase open-circuit "
        "potential in the positive electrode."
    )
    parts.append("")
    parts.append(
        "The underlying parameter string depends on the model options. For "
        "the **default single-phase** electrode (``\"particle phases\": "
        '"1"``), the ``Primary:`` / ``Secondary:`` prefix is dropped — so '
        "``param.n.prim.D(c_s, T)`` is the string "
        '``"Negative particle diffusivity [m2.s-1]"``. For a **composite '
        "electrode** (``\"particle phases\": \"2\"``), the same shorthand "
        'becomes ``"Primary: Negative particle diffusivity [m2.s-1]"`` and '
        "``param.n.sec.D(c_s, T)`` becomes "
        '``"Secondary: Negative particle diffusivity [m2.s-1]"``. The tables '
        "below are generated with default options, so they show the "
        "unprefixed strings; mentally prepend ``Primary:`` / ``Secondary:`` "
        "when working with a composite electrode."
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
