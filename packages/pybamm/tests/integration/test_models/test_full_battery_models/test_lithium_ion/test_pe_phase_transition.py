#
# Integration tests for the PE phase-transition (core-shell) degradation model
#
import pytest

import pybamm
import tests


class TestPEPhaseTransition:
    """End-to-end smoke tests: build, discretise and solve SPM and DFN with the
    Zhuo2023 parameter set under the PE phase-transition degradation option.
    Exercises the PE-degradation submodels, the PE-shell branch in the kinetics
    submodels, and the core/shell-domain extensions to averages and broadcasts.
    """

    @pytest.mark.parametrize(
        "model_cls", [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN]
    )
    def test_basic_processing(self, model_cls):
        model = model_cls({"PE degradation": "phase transition"})
        param = pybamm.ParameterValues("Zhuo2023")
        # Zhuo2023's default initial state is fully discharged (NE near empty,
        # PE near full). Apply 1C charge.
        param["Current function [A]"] = -param["Nominal cell capacity [A.h]"]
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        # skip_output_tests because the standard output checks (e.g. capacity
        # comparison) are calibrated for non-degrading models
        modeltest.test_all(skip_output_tests=True)
