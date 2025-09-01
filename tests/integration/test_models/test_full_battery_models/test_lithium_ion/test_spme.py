#
# Tests for the lithium-ion SPMe model
#
import pytest

import pybamm
from tests import BaseIntegrationTestLithiumIon


class TestSPMe(BaseIntegrationTestLithiumIon):
    @pytest.fixture(autouse=True)
    def setup(self):
        pybamm.set_logging_level("DEBUG")
        self.model = pybamm.lithium_ion.SPMe
