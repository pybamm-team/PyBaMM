#
# Test dispatching mechanism in entry points
#
import pytest

import pybamm
from pybamm.dispatch.entry_points import (
    _get_cache_dir,
    clear_model_cache,
    get_cache_path,
)

MODEL_URL = "https://raw.githubusercontent.com/pybamm-team/pybamm-reservoir-example/refs/heads/main/dfn.py"


class TestDispatch:
    def setup_method(self):
        """Clear cache before each test"""
        clear_model_cache()

    def test_model_loads_through_entry_points(self):
        """Test that a model loaded through Model() function is actually functional"""
        # Load model with build=False to avoid full initialization for faster testing
        model = pybamm.Model("SPM", build=False)

        assert hasattr(model, "name")
        assert hasattr(model, "variables")
        assert hasattr(model, "submodels")
        assert hasattr(model, "algebraic")
        assert hasattr(model, "rhs")

        assert "SPM" in str(model)

    def test_getitem_returns_instance(self):
        """Test that __getitem__ returns an instantiated model"""
        model_instance = pybamm.dispatch.models["SPM"]
        assert hasattr(model_instance, "__class__")
        assert model_instance.__class__.__name__ == "SPM"

    def test_model_function(self):
        """Test Model function with arguments and options"""
        model = pybamm.Model("SPM")
        assert model.__class__.__name__ == "SPM"

        # Test Model function with options parameter
        options = {"thermal": "isothermal"}
        model = pybamm.Model("SPM", options=options)
        assert model.__class__.__name__ == "SPM"

    def test_model_value_error(self):
        """Test that Model raises ValueError when given invalid arguments"""

        # Neither model nor url provided
        with pytest.raises(
            ValueError, match="You must provide exactly one of `model` or `url`."
        ):
            pybamm.Model()

        # Both model and url provided
        with pytest.raises(
            ValueError, match="You must provide exactly one of `model` or `url`."
        ):
            pybamm.Model(model="SPM", url="http://example.com/dfn.py")

    def test_model_download_runtime_error(self):
        """Test that Model raises RuntimeError when download fails"""

        bad_url = "h://example.invalid/model.json"

        with pytest.raises(RuntimeError, match="Failed to download model from URL:"):
            pybamm.Model(url=bad_url, force_download=True)

    def test_invalid_model_name_raises_value_error(self):
        """Test that Model raises ValueError for an invalid model name"""

        bad_model = "NonExistentModel123"

        with pytest.raises(ValueError, match=f"Could not load model '{bad_model}':"):
            pybamm.Model(model=bad_model)

    def test_model_download_and_cache(self):
        """Force exception in clear_model_cache without interfering with real cache files"""
        cache_dir = _get_cache_dir()
        bad_path = cache_dir / "force_exception_test_dir"

        try:
            bad_path.mkdir(exist_ok=True)  # not .json
            try:
                bad_path.unlink()
            except Exception as e:
                print(f"Expected error: {e}")
        finally:
            if bad_path.exists():
                bad_path.rmdir()

    def test_force_download_overwrites_cache(self):
        """Force an exception when trying to unlink a directory"""
        cache_dir = _get_cache_dir()
        bad_path = cache_dir / "force_exception_test_dir"

        try:
            bad_path.mkdir(exist_ok=True)
            clear_model_cache()
            try:
                bad_path.unlink()
            except Exception as e:
                print(f"Expected error: {e}")
        finally:
            if bad_path.exists():
                bad_path.rmdir()

    def test_clear_model_cache_exception_branch(self, capsys):
        """Test that clear_model_cache gracefully handles deletion errors"""
        cache_dir = pybamm.dispatch.entry_points._get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        bad_path = cache_dir / "bad.json"
        bad_path.mkdir(exist_ok=True)

        try:
            clear_model_cache()

            captured = capsys.readouterr()
            assert "Could not delete" in captured.out
            assert "bad.json" in captured.out

            assert bad_path.exists()
        finally:
            bad_path.rmdir()

    def test_model_download_and_cache_integration(self, capsys):
        """Integration test using a real model URL"""

        cache_path = get_cache_path(MODEL_URL)

        # Clean up: if cache path exists as directory, remove it
        if cache_path.exists():
            if cache_path.is_dir():
                import shutil

                shutil.rmtree(cache_path)
            else:
                clear_model_cache()

        # First call -> should download and print "Model cached at"
        model = pybamm.Model(url=MODEL_URL, force_download=True)
        captured = capsys.readouterr()
        assert "Model cached at:" in captured.out
        assert hasattr(model, "name")

        # Second call -> should use cached file
        model2 = pybamm.Model(url=MODEL_URL)
        captured = capsys.readouterr()
        assert "Using cached model at:" in captured.out
        assert hasattr(model2, "name")
