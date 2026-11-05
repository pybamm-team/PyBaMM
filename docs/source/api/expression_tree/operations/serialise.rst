Serialise
=========

PyBaMM serialises expression trees, discretised models, meshes, solvers and
related objects through a single encode/decode kernel
(``pybamm.expression_tree.operations.serialise_kernel``). The kernel is
**safe-or-loud**: a value either round-trips, or encoding raises
``SerialisationError`` naming the offending class and field. It never silently
drops data. There is one canonical wire format (each node carries a ``$type``
tag holding its dotted ``module.ClassName`` path), and a read-only compatibility
layer keeps files written by older PyBaMM versions loading.

Wire format
-----------

``encode`` produces plain JSON. Native values (``None``, booleans, integers,
strings, finite floats) pass through unchanged, lists and dicts stay JSON
arrays/objects, and every other object becomes a tagged node: its class-specific
fields plus a ``"$type"`` key holding the dotted ``module.ClassName`` path. Child
symbols (and other serialisable sub-objects) are recursively encoded under
``"children"``. For example, ``pybamm.Index(pybamm.StateVector(slice(0, 2)), 0)``
encodes as:

.. code-block:: json

    {
      "name": "Index[0]",
      "index": {"start": 0, "stop": 1, "step": null},
      "check_size": false,
      "children": [
        {
          "name": "y[0:2]",
          "domains": {"primary": [], "secondary": [], "tertiary": [], "quaternary": []},
          "y_slice": [{"start": 0, "stop": 2, "step": null}],
          "evaluation_array": [true, true],
          "$type": "pybamm.expression_tree.state_vector.StateVector"
        }
      ],
      "$type": "pybamm.expression_tree.unary_operators.Index"
    }

On decode the ``"$type"`` path is resolved with ``importlib``, so classes defined
outside the ``pybamm`` package round-trip too. Apart from ``"$type"`` and
``"children"``, the keys of a node belong to the class (its constructor
parameters, or whatever its ``to_json`` hook emits).

A few non-class values the kernel encodes itself use reserved ``"$type"`` tags:

.. list-table::
   :header-rows: 1

   * - Value
     - Encoding
   * - ``nan``, ``inf``, ``-inf``
     - ``{"$type": "builtins.float", "value": "NaN"}`` (or ``"Infinity"`` / ``"-Infinity"``)
   * - ``numpy.ndarray``
     - ``{"$type": "numpy.ndarray", "data": [...], "dtype": "float64"}``
   * - ``slice``
     - ``{"$type": "builtins.slice", "start": 0, "stop": 1, "step": null}``
   * - ``tuple``
     - ``{"$type": "builtins.tuple", "items": [...]}`` (so it does not decode back as a list)
   * - a class itself (not an instance)
     - ``{"$type": "type", "class": "module.ClassName"}``

Legacy formats (read-only)
--------------------------

Files written by older PyBaMM versions used three different shapes. All three
still load: ``decode`` passes every dict through ``normalise_legacy``, which maps
them onto the canonical tag. New files are always written in the canonical
format; the legacy shapes are never written.

.. list-table::
   :header-rows: 1

   * - Legacy shape
     - Previously written by
     - Normalisation
   * - ``{"py/object": "dotted.path", "py/id": 1, ...}``
     - discretised-model files
     - ``py/object`` becomes ``$type``; ``py/id`` is dropped
   * - ``{"type": "ClassName", ...}``
     - symbol, parameter-values and experiment files
     - bare class names resolve as ``pybamm.ClassName``
   * - ``{"class": "ClassName", "module": "some.module"}``
     - submesh and spatial-method type references
     - becomes a ``{"$type": "type"}`` class reference

Making a class serialisable
---------------------------

**Usually you do nothing.** A new ``Symbol`` (or other serialisable) subclass is
encoded automatically by the introspection ``DefaultCodec``: it reads each
``__init__`` parameter back off the instance (by the same name, or the
``_``-prefixed private attribute) and emits it. If your class stores its
constructor arguments as same-named attributes, it round-trips with no extra
code.

**The safe-or-loud contract tells you when you must act.** If a stored or
required ``__init__`` parameter is *not* captured, ``encode`` raises
``SerialisationError`` naming the class and the parameter the first time the
class is serialised. The strategy-coverage meta-tests
(``tests/unit/test_serialisation/test_symbol_strategy_coverage.py`` and
``test_base_strategy_coverage.py``) turn a new concrete subclass with no
round-trip coverage into a CI failure. So a gap is a loud failure, never a silent
one.

**Add a hook when reconstruction is irregular.** Define ``to_json`` /
``_from_json`` on the class when it cannot be rebuilt by simply replaying its
``__init__`` parameters, for example when a constructor argument is itself a
``Symbol`` or another serialisable object. The kernel only recurses the
``children`` list, so **any ``Symbol``-valued argument must travel through
``children``**; every other field a hook emits must already be JSON-native.

.. code-block:: python

    class MyEvent(pybamm.Symbol):
        def __init__(self, name, expression):
            super().__init__(name, children=[expression])

        def to_json(self):
            # ``expression`` is a Symbol, so it rides in children; the kernel
            # recurses children and reconstructs it for us.
            return {"name": self.name, "children": [self.children[0]]}

        @classmethod
        def _from_json(cls, snippet):
            return cls(snippet["name"], snippet["children"][0])

**Opt out genuinely derived fields explicitly.** If a constructor parameter is
re-derived on construction (so it need not be stored) or is a non-serialisable
transient, list its name in the class's ``_serialise_derived_params`` frozenset.
This is a reviewed, in-diff decision that the coverage guard honours; it is not a
runtime skip. See ``Mesh``, ``SubMesh`` and ``DomainConcatenation`` for existing
examples (each waives construction inputs it rebuilds from other serialised
fields on decode).

.. code-block:: python

    class MySubMesh(pybamm.SubMesh):
        # ``builder`` is only used to compute ``edges`` in __init__; the mesh is
        # reconstructed from ``edges``, so ``builder`` is never serialised.
        _serialise_derived_params = frozenset({"builder"})

**Register round-trip coverage.** Add (or extend) a strategy under
``tests/strategies/`` so your class is exercised by the round-trip property
tests, or -- if it is intentionally not independently round-trippable -- add it to
the relevant base's exemption set in the coverage meta-test with a one-line
reason. The meta-test will tell you exactly which classes are uncovered.

Class references
----------------

A bare class (rather than an instance) is serialised through the kernel's
class-reference codec as ``{"$type": "type", "class": "module.ClassName"}`` (used
for submesh types and spatial methods). The legacy ``{"class": ..., "module":
...}`` shape is still read on decode.

API
---

.. autoclass:: pybamm.expression_tree.operations.serialise.Serialise
  :members:
