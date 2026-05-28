AOT compilation
===============

.. automodule:: pybamm.codegen.compilation
   :members: aot_compile

Environment variables
---------------------

``PYBAMM_CASADI_AOT_CACHE``
   On-disk cache directory. Defaults to ``$TMPDIR/pybamm_casadi_aot``.

``PYBAMM_CASADI_AOT_KEEP_C``
   If set, retain the generated ``.c`` source next to each compiled library.
   Useful for debugging codegen output.
