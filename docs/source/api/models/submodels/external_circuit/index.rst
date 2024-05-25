External circuit
================

Models to enforce different boundary conditions (as imposed by an imaginary external
circuit) such as constant current, constant voltage, constant power, or any other
relationship between the current and voltage. "Current control" enforces these directly
through boundary conditions, while "Function control"
submodels add an algebraic equation (for the current) and hence can be used to set any
variable to be constant.

.. toctree::
  :maxdepth: 1

  discharge_throughput
  explicit_control_external_circuit
  function_control_external_circuit
