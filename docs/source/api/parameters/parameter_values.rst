Parameter Values
================

The ``ParameterValues`` class is the main interface for managing battery simulation
parameters. It provides methods for storing, updating, and processing parameters,
as well as introspection features for exploring parameter sets.

Features
------------

The ``ParameterValues`` class now includes several helpful features:

- **Parameter Introspection**: Use :meth:`get_info` to get metadata about parameters
  including units and category.
- **Category Browsing**: Use :meth:`list_by_category` to find parameters by domain
  (e.g., "negative electrode", "thermal").
- **Parameter Comparison**: Use :meth:`diff` to compare two parameter sets and see
  what's added, removed, or changed.
- **Unified Update API**: The ``check_already_exists`` argument in :meth:`update` has
  been deprecated. Use :meth:`set` instead

Example usage::

    import pybamm

    # Load a parameter set
    params = pybamm.ParameterValues("Chen2020")

    # Get info about a parameter
    info = params.get_info("Maximum concentration in negative electrode [mol.m-3]")
    print(f"Value: {info.value}, Units: {info.units}, Category: {info.category}")

    # List parameters by category
    neg_electrode_params = params.list_by_category("negative electrode")
    print(f"Found {len(neg_electrode_params)} negative electrode parameters")

    # Compare with another parameter set
    marquis = pybamm.ParameterValues("Marquis2019")
    diff = params.diff(marquis)
    print(f"Changed parameters: {list(diff.changed.keys())[:5]}")

API Reference
-------------

.. autoclass:: pybamm.ParameterValues
  :members:
