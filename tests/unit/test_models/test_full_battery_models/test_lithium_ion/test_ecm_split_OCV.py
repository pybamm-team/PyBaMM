#
# Test for the ecm split-OCV model
#
import pybamm


class TestECMSplitOCV:
    def test_ecmsplitocv_well_posed(self):
        model = pybamm.lithium_ion.ECMsplitOCV()
        model.check_well_posedness()

    def test_get_default_quick_plot_variables(self):
        model = pybamm.lithium_ion.ECMsplitOCV()
        variables = model.default_quick_plot_variables
        assert "Current [A]" in variables
