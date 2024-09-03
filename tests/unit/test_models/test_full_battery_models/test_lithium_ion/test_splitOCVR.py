#
# Test for the ecm split-OCV model
#
import pybamm


class TestSplitOCVR:
    def test_ecmsplitocv_well_posed(self):
        model = pybamm.lithium_ion.SplitOCVR()
        model.check_well_posedness()

    def test_get_default_quick_plot_variables(self):
        model = pybamm.lithium_ion.SplitOCVR()
        variables = model.default_quick_plot_variables
        assert "Current [A]" in variables
