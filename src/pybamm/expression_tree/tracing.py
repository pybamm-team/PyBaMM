"""Symbolic tracing context for faithful expression tree capture.

When tracing a Python function to capture its computation as a PyBaMM
expression tree (for serialization), algebraic simplifications like
distribution (a*(b+c) -> a*b+a*c) and reassociation (a+(b+c) -> (a+b)+c)
must be suppressed. These transformations are algebraically correct but
change floating-point evaluation order, causing the serialized expression
to produce different numerical results than the original callable.

This follows the standard approach used by all major tracing frameworks:
  - JAX (jax.make_jaxpr): records primitives without optimization
  - PyTorch FX (torch.fx.symbolic_trace): captures the program as-written
  - TensorFlow (tf.function): records ops without Grappler optimizations
The principle: tracing records what the function does; optimization is a
separate, explicit step.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager

_tracing: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "pybamm_tracing", default=False
)


def is_tracing() -> bool:
    """Return True if we are inside a tracing context."""
    return _tracing.get()


@contextmanager
def tracing():
    """Disable FP-reordering simplifications while building an expression tree.

    Use this when tracing a callable to capture its computation graph
    faithfully, without algebraic rewrites that change evaluation order.
    """
    token = _tracing.set(True)
    try:
        yield
    finally:
        _tracing.reset(token)
