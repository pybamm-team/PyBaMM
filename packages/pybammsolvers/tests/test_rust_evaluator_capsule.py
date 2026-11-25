"""pybammsolvers claims a Rust evaluator only from a capsule whose ABI it can read."""

from __future__ import annotations

import ctypes

import pytest
import pybammsolvers.idaklu as idaklu

# Kept alive at module level: a capsule borrows its name for as long as it exists.
_EVALUATOR_CAPSULE = b"pybamm.rust.evaluator"


class _TableHead(ctypes.Structure):
    """The two fields every version's function table leads with."""

    _fields_ = [("abi_version", ctypes.c_uint32), ("size", ctypes.c_uint32)]


class _Handle(ctypes.Structure):
    _fields_ = [("ffi", ctypes.POINTER(_TableHead)), ("model", ctypes.c_void_p)]


def _capsule(handle):
    new_capsule = ctypes.pythonapi.PyCapsule_New
    new_capsule.restype = ctypes.py_object
    new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return new_capsule(ctypes.addressof(handle), _EVALUATOR_CAPSULE, None)


def _newton_solver(rust_evaluator):
    return idaklu.StandaloneNewtonSolver(
        rust_evaluator=rust_evaluator,
        n_rhs=0,
        n_alg=1,
        jac_rows=[0],
        jac_cols=[0],
        atol=[1e-8],
        rtol=1e-8,
        step_tol=1e-12,
        max_iter=10,
        max_backtracks=5,
        eps_newt=1e-3,
        use_sparse=False,
    )


class TestRustEvaluatorCapsule:
    def test_refuses_another_abi_version(self):
        table = _TableHead(abi_version=0, size=ctypes.sizeof(_TableHead))
        handle = _Handle(ctypes.pointer(table), None)
        capsule = _capsule(handle)
        with pytest.raises(
            RuntimeError,
            match=rf"has ABI version 0, but pybammsolvers was built for version "
            rf"{idaklu.rust_abi_version}\.",
        ):
            _newton_solver(capsule)
        # Refused before the claim, so the capsule still owns what it holds
        assert repr(capsule).startswith('<capsule object "pybamm.rust.evaluator"')

    def test_refuses_a_table_shorter_than_it_reads(self):
        # This build's version, but only the table's leading fields
        table = _TableHead(
            abi_version=idaklu.rust_abi_version, size=ctypes.sizeof(_TableHead)
        )
        handle = _Handle(ctypes.pointer(table), None)
        capsule = _capsule(handle)
        with pytest.raises(
            RuntimeError, match=r"function table is 8 bytes, older than the \d+ bytes"
        ):
            _newton_solver(capsule)

    def test_refuses_anything_but_an_evaluator_capsule(self):
        with pytest.raises(
            ValueError, match=r"expected a 'pybamm\.rust\.evaluator' capsule, got int"
        ):
            _newton_solver(0xDEAD0)
