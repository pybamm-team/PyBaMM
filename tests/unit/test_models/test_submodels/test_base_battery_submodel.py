import pytest

import pybamm


class DummyParam:
    def __init__(self):
        self.domain_params = {"Negative": {}, "Positive": {}}


def test_valid_domain():
    submodel = pybamm.BaseBatterySubModel(DummyParam(), "Negative")
    assert submodel.domain == "Negative"


def test_invalid_domain():
    with pytest.raises(pybamm.DomainError):
        pybamm.BaseBatterySubModel(DummyParam(), "InvalidDomain")
