"""Workarounds for PyBaMM behaviour the zoo cannot fix from here.

The zoo changes nothing under ``packages/pybamm/src`` by design, so the upstream
quirks it has to work around are collected here instead of being copied into each
model. Every function names the upstream change that would retire it, so the list
doubles as the inventory of core fixes the zoo is waiting on.
"""

from __future__ import annotations

from typing import Any


def spm_default_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """Options an ``SPM`` subclass needs but does not inherit.

    ``pybamm.lithium_ion.SPM.__init__`` defaults ``"x-average side reactions"``
    with ``self.__class__ in [SPM, MPM]``, so a subclass is left with ``"false"``
    and then rejected by the option validator. Retire this once SPM carries the
    default as a class attribute its subclasses inherit.
    """
    return {"x-average side reactions": "true", **(options or {})}


def cited_keys() -> set[str]:
    """The citation keys PyBaMM would credit if asked to print right now.

    Reads ``pybamm.citations`` private state: keys registered by key, plus those
    registered as raw BibTeX, which PyBaMM leaves unparsed until print time.
    Retire this once ``pybamm.Citations`` exposes its registered keys publicly.
    """
    import pybamm
    from pybamm_model_zoo._citations import parse_bibtex

    keys = set(pybamm.citations._papers_to_cite)
    for citation in pybamm.citations._unknown_citations:
        keys.update(parse_bibtex(citation))
    return keys


def reset_citations() -> None:
    """Clear PyBaMM's citation registry so a check can observe one model's own.

    Retire alongside :func:`cited_keys`.
    """
    import pybamm

    pybamm.citations._reset()
