Tree utilities
==============

Tree rewriting accepts a single ``pybamm.Symbol`` root. To rewrite several
expressions with the same mapping, iterate explicitly and share a cache::

    cache = {}
    equations = {
        variable: pybamm.replace(equation, mapping, cache=cache)
        for variable, equation in model.rhs.items()
    }

The cache belongs to this mapping; use a fresh cache for a different mapping.
Model parameter processing remains available through
``parameter_values.process_model(model)``.

.. autoclass:: pybamm.TreeDef
  :members:

.. autofunction:: pybamm.tree_flatten

.. autofunction:: pybamm.tree_unflatten

.. autofunction:: pybamm.rebuild

.. autofunction:: pybamm.tree_map

.. autofunction:: pybamm.replace

Migrating in-place updates
--------------------------

Public symbol setters and domain/list edits remain available but emit
``SymbolMutationDeprecationWarning``. Prefer constructing replacements::

    updated = variable.create_copy(scale=2, reference=0, bounds=(0, 1))
    expression = pybamm.replace(expression, {variable: updated})
    symbol = symbol.with_domains({"primary": ["negative electrode"]})
    symbol = symbol.without_domains()
    symbol = symbol.with_mesh(mesh)

For names, scalar values and function inputs, construct a new symbol of the
appropriate class. IDs update automatically; callers need not call ``set_id``.

Deprecated updates invalidate cached IDs and derived values across live symbols.
This deliberately slow compatibility path adds no per-symbol tracking to normal
replacement. Public domain and child-list views support deprecated edits while
internal storage remains shared. Mutating symbols already used as dictionary keys
cannot repair those dictionaries: use a fresh replacement memo and rebuild any
parameter-processed or discretised model after legacy edits. Changes to a child's
domains do not propagate into constructor-derived parent domains; reconstruct
those expressions as well. Legacy mutation is rejected while ``gc.freeze()`` is
active because frozen ancestors cannot be discovered for cache invalidation;
use out-of-place replacement or unfreeze the collector first.

The test suite forbids deprecated mutation even with warnings ignored. Only isolated compatibility subprocesses exercise the legacy path;
notebook kernels inherit the same guard as other tests.
