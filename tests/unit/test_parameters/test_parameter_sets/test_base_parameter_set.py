from pybamm.input.parameters.base_parameter_set import AbstractBaseParameters


def test_unavailability():
    class TestEmptyParamSet(AbstractBaseParameters):
        def __init__(self):
            super().__init__()

    assert not TestEmptyParamSet().degradation_available()
    assert not TestEmptyParamSet().thermal_available()
    assert not TestEmptyParamSet().plating_available()


def test_degradation_available():
    class TestFilledParamSet(AbstractBaseParameters):
        def __init__(self):
            super().__init__()
            self._sei = {"A param": 0}

    assert TestFilledParamSet().degradation_available()


def test_thermal_available():
    class TestFilledParamSet(AbstractBaseParameters):
        def __init__(self):
            super().__init__()
            self._thermal = {"A param": 0}

    assert TestFilledParamSet().thermal_available()


def test_plating_available():
    class TestFilledParamSet(AbstractBaseParameters):
        def __init__(self):
            super().__init__()
            self._plating = {"A param": 0}

    assert TestFilledParamSet().plating_available()
