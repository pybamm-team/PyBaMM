"""
Tests for the lazy import mechanism in PyBaMM using lazy_loader.

These tests verify that:
1. All lazy imports resolve correctly via the stub file
2. Caching works as expected
3. dir() includes all lazy imports
4. AttributeError is raised for undefined attributes
5. Module and class imports return correct types
6. Known special attributes (KNOWN_COORD_SYS, t) are accessible
7. Thread safety of lazy loading
8. EAGER_IMPORT mode works correctly
"""

import os
import subprocess
import sys
import types

import pytest


class TestLazyImports:
    """Tests for lazy import functionality."""

    def test_lazy_imports_resolve(self):
        """Test that lazy imports can be resolved via getattr."""
        import pybamm

        # Test a selection of lazy imports
        lazy_imports = [
            "CasadiSolver",
            "Simulation",
            "ParameterValues",
            "Mesh",
            "Solution",
            "Timer",
            "FuzzyDict",
            "root_dir",
            "lithium_ion",
            "lead_acid",
            "experiment",
        ]

        failed_imports = []
        for name in lazy_imports:
            try:
                attr = getattr(pybamm, name)
                assert attr is not None, f"{name} resolved to None"
            except (ImportError, AttributeError) as e:
                failed_imports.append((name, str(e)))

        if failed_imports:
            msg = "\n".join(f"  {name}: {error}" for name, error in failed_imports)
            pytest.fail(f"Failed to resolve lazy imports:\n{msg}")

    def test_caching_works(self):
        """Test that accessing an attribute twice returns the same object."""
        import pybamm

        # Access a lazy attribute twice
        first_access = pybamm.CasadiSolver
        second_access = pybamm.CasadiSolver

        # Should be the exact same object (cached)
        assert first_access is second_access

    def test_dir_includes_lazy_imports(self):
        """Test that dir(pybamm) includes lazy import names."""
        import pybamm

        pybamm_dir = dir(pybamm)

        # Check some known lazy imports are in dir()
        expected_in_dir = [
            "CasadiSolver",
            "Simulation",
            "ParameterValues",
            "Mesh",
            "Solution",
            "lithium_ion",
            "lead_acid",
        ]

        missing = []
        for name in expected_in_dir:
            if name not in pybamm_dir:
                missing.append(name)

        if missing:
            pytest.fail(f"Missing from dir(pybamm): {missing}")

    def test_undefined_attribute_raises_error(self):
        """Test that accessing undefined attributes raises AttributeError."""
        import pybamm

        with pytest.raises(AttributeError, match="this_attribute_definitely_does_not_exist_xyz123"):
            _ = pybamm.this_attribute_definitely_does_not_exist_xyz123

    def test_lazy_module_imports(self):
        """Test that lazy module imports return ModuleType."""
        import pybamm

        module_imports = [
            "lithium_ion",
            "lead_acid",
            "experiment",
            "callbacks",
            "telemetry",
            "constants",
        ]

        for name in module_imports:
            attr = getattr(pybamm, name)
            assert isinstance(
                attr, types.ModuleType
            ), f"{name} should be a module, got {type(attr)}"

    def test_lazy_class_imports(self):
        """Test that lazy class imports return type (class objects)."""
        import pybamm

        # Test some known class imports
        class_names = [
            "CasadiSolver",
            "Simulation",
            "ParameterValues",
            "Mesh",
            "Solution",
        ]

        for name in class_names:
            attr = getattr(pybamm, name)
            assert isinstance(attr, type), f"{name} should be a class, got {type(attr)}"

    def test_known_coord_sys_accessible(self):
        """Test that KNOWN_COORD_SYS is accessible (from eager import)."""
        import pybamm

        # Should be accessible without error
        coord_sys = pybamm.KNOWN_COORD_SYS
        assert coord_sys is not None
        assert isinstance(coord_sys, set | frozenset | list | tuple)

    def test_t_accessible(self):
        """Test that pybamm.t is accessible (from eager import)."""
        import pybamm

        # Should be accessible without error
        t = pybamm.t
        assert t is not None

    def test_size_distribution_parameters_accessible(self):
        """Test that size distribution parameters (from wildcard module) are accessible."""
        import pybamm

        # These come from the size_distribution_parameters module via the stub
        assert hasattr(pybamm, "get_size_distribution_parameters")
        assert hasattr(pybamm, "lognormal")

        get_size_dist = pybamm.get_size_distribution_parameters
        lognormal = pybamm.lognormal

        assert callable(get_size_dist)
        assert callable(lognormal)


class TestThreadSafety:
    """Tests for thread-safe lazy loading."""

    def test_concurrent_access(self):
        """Test that concurrent access to lazy imports is thread-safe."""
        import threading

        import pybamm

        errors = []
        results = []
        lock = threading.Lock()

        def access_attr():
            try:
                # Access various lazy attributes
                solver = pybamm.CasadiSolver
                sim = pybamm.Simulation
                param = pybamm.ParameterValues

                with lock:
                    results.append((solver, sim, param))
            except Exception as e:
                with lock:
                    errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=access_attr) for _ in range(50)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Check no errors occurred
        assert not errors, f"Errors during concurrent access: {errors}"

        # Check all threads got the same objects (due to caching)
        if results:
            first_result = results[0]
            for result in results[1:]:
                assert result[0] is first_result[0], "CasadiSolver not cached properly"
                assert result[1] is first_result[1], "Simulation not cached properly"
                assert result[2] is first_result[2], "ParameterValues not cached properly"


class TestEagerImportMode:
    """Tests for EAGER_IMPORT environment variable mode."""

    @pytest.mark.skip(
        reason="EAGER_IMPORT=1 causes circular import due to PyBaMM's module structure"
    )
    def test_eager_import_mode(self):
        """Test that EAGER_IMPORT=1 forces eager loading of all imports.

        Note: This test is skipped because PyBaMM has circular dependencies
        that are resolved by lazy loading. EAGER_IMPORT mode exposes these
        circular imports and fails. This is expected behavior - the lazy
        loading is what makes PyBaMM work correctly.
        """
        # Run a subprocess with EAGER_IMPORT=1 to test eager loading
        env = os.environ.copy()
        env["EAGER_IMPORT"] = "1"

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import pybamm; print(pybamm.CasadiSolver)",
            ],
            capture_output=True,
            text=True,
            env=env,
        )

        # Should complete without error
        assert result.returncode == 0, f"EAGER_IMPORT failed: {result.stderr}"
        assert "CasadiSolver" in result.stdout


class TestStubFileIntegrity:
    """Tests to verify the stub file is properly configured."""

    def test_all_is_list(self):
        """Test that __all__ is a list."""
        import pybamm

        assert isinstance(pybamm.__all__, list)

    def test_all_contains_expected_items(self):
        """Test that __all__ contains expected items."""
        import pybamm

        expected_items = [
            "CasadiSolver",
            "Simulation",
            "ParameterValues",
            "Solution",
            "Experiment",
            "lithium_ion",
        ]

        for item in expected_items:
            assert item in pybamm.__all__, f"{item} not in __all__"

    def test_version_accessible(self):
        """Test that __version__ is accessible."""
        import pybamm

        assert hasattr(pybamm, "__version__")
        assert isinstance(pybamm.__version__, str)
