#
# Test dispatching mechanism in entry points
#
import pybamm
from pybamm.dispatch.entry_points import clear_model_cache, get_cache_path

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

    def test_model_download_and_cache(self):
        """Test that models are downloaded once and reused from cache"""
        clear_model_cache()
        cache_path = get_cache_path(MODEL_URL)

        model = pybamm.Model(url=MODEL_URL, force_download=True)
        assert model is not None
        assert cache_path.exists()

        model2 = pybamm.Model(url=MODEL_URL)
        assert model2 is not None

    def test_force_download_overwrites_cache(self):
        """Test that force_download replaces stale cached content"""
        cache_path = get_cache_path(MODEL_URL)

        cache_path.write_text("stale content")

        model = pybamm.Model(url=MODEL_URL, force_download=True)
        assert model is not None
        assert "stale content" not in cache_path.read_text()
