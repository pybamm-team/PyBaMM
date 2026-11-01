Citations
=========

Install the optional ``pybamm[cite]`` extra to load and export citations.
Downstream packages can load a BibTeX database without citing all its entries:

.. code-block:: python

   pybamm.citations.read_citations("path/to/CITATIONS.bib")
   pybamm.citations.register("Downstream2026", reason="Downstream model")
   pybamm.citations.register("Downstream2026", reason="Parameter values")

``read_citations()`` with no argument loads PyBaMM's bundled database. Existing
entries remain available; replacing a key with different content warns. A key
registered before loading its database is resolved when that database is loaded.

Reasons are plain text. Distinct reasons accumulate, and repeated reasons are
included only once. Without an explicit reason, the calling class name is used
when available. Parameter metadata can mix keys and two-string ``(key, reason)``
tuples or lists:

.. code-block:: python

   parameters = pybamm.ParameterValues(
       {"citations": ["Chen2020", ("Nyman2008", "Electrolyte transport")]}
   )

For named parameter sets, plain keys use the set name as their reason; dictionary
construction retains the ``ParameterValues`` fallback. Named-set reasons survive
copies and JSON round-trips. Invalid metadata entries raise ``ValueError`` when
the parameter values are constructed.

Reasons appear only with ``verbose=True``; default output remains unannotated.
Text output appends ``key: reason; other reason`` annotations. BibTeX output
appends reasons to each entry's ``note`` field, preserving existing notes and
escaping special characters. Both formats support terminal and file output:

.. code-block:: python

   pybamm.print_citations(verbose=True)
   pybamm.print_citations("citations.txt", verbose=True)
   pybamm.print_citations("citations.bib", output_format="bibtex", verbose=True)

Exports leave the stored bibliography unchanged. Reasons are sorted for stable
output, and later registrations appear in subsequent exports.

.. autoclass:: pybamm.Citations
  :members:

.. autofunction:: pybamm.print_citations
