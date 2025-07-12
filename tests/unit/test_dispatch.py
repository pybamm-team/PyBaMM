#
# Test dispatching mechanism in entry points
#
import pybamm


class TestDispatch:
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
        model = pybamm.Model("SPM", options)
        assert model.__class__.__name__ == "SPM"
