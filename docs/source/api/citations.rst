Citations
=========

Install the optional ``pybamm[cite]`` extra to load and export citations.
Downstream packages can load their own BibTeX databases and register citations
from their models and parameter sets:

.. code-block:: python

   pybamm.citations.read_citations("path/to/CITATIONS.bib")
   pybamm.citations.register("Downstream2026", reason="Downstream model")

Each citation can be given a reason. Without an explicit reason, the calling
class name is used when available. Parameter sets can provide citations as keys
or two-string ``(key, reason)`` pairs:

.. code-block:: python

   pybamm.ParameterValues(
       {"citations": ["Chen2020", ("Nyman2008", "Electrolyte transport")]}
   )

Use ``verbose=True`` to include reasons in the text output:

.. code-block:: python

   pybamm.print_citations(verbose=True)

.. autoclass:: pybamm.Citations
  :members:

.. autofunction:: pybamm.print_citations
