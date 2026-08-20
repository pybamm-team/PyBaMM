"""How ``pybamm.rust`` loads the compiled extension.

pybammsolvers resolves the Rust FFI entry points with ``dlsym(RTLD_DEFAULT)``,
which only finds them when the extension is in the process-global symbol scope.
CPython opens extension modules ``RTLD_LOCAL`` on Linux, so the facade has to ask
for global visibility there; macOS already exports them globally.
"""

import os
import sys

import pytest

import pybamm.rust

# The Linux cases stub sys.getdlopenflags/setdlopenflags and compare against
# os.RTLD_*, none of which exist off POSIX.
requires_posix_dlopen = pytest.mark.skipif(
    not hasattr(sys, "getdlopenflags"), reason="POSIX dlopen flags only"
)


class TestGlobalSymbolVisibility:
    @requires_posix_dlopen
    def test_sets_rtld_global_on_linux(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(sys, "getdlopenflags", lambda: os.RTLD_NOW)
        applied = []
        monkeypatch.setattr(sys, "setdlopenflags", applied.append)

        with pybamm.rust._global_symbol_visibility():
            pass

        assert applied, "no dlopen flags were set on Linux"
        assert applied[0] & os.RTLD_GLOBAL

    @requires_posix_dlopen
    def test_restores_the_previous_flags_on_linux(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(sys, "getdlopenflags", lambda: os.RTLD_NOW)
        applied = []
        monkeypatch.setattr(sys, "setdlopenflags", applied.append)

        with pybamm.rust._global_symbol_visibility():
            pass

        assert applied[-1] == os.RTLD_NOW

    @requires_posix_dlopen
    def test_restores_the_previous_flags_when_the_import_raises(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(sys, "getdlopenflags", lambda: os.RTLD_NOW)
        applied = []
        monkeypatch.setattr(sys, "setdlopenflags", applied.append)

        with (
            pytest.raises(ImportError),
            pybamm.rust._global_symbol_visibility(),
        ):
            raise ImportError("extension missing")

        assert applied[-1] == os.RTLD_NOW

    def test_leaves_flags_untouched_off_linux(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        # raising=False: Windows has no setdlopenflags, and is itself off-Linux.
        monkeypatch.setattr(
            sys,
            "setdlopenflags",
            lambda _: pytest.fail("must not touch dlopen flags"),
            raising=False,
        )

        with pybamm.rust._global_symbol_visibility():
            pass

    @requires_posix_dlopen
    def test_importing_the_facade_does_not_leak_flags(self):
        """The real import already ran at collection; flags must be back to normal."""
        assert not sys.getdlopenflags() & os.RTLD_GLOBAL


class TestFacadeExports:
    def test_core_is_importable_through_the_facade(self):
        assert pybamm.rust.CompiledModel is not None
        assert pybamm.rust.ExprGraph is not None
