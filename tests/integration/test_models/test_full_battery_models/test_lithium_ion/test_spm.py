#
# Tests for the lithium-ion SPM model
#
from tests import TestCase
import pybamm
from tests import BaseIntegrationTestLithiumIon
import pytest


class TestSPM(BaseIntegrationTestLithiumIon):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.model = pybamm.lithium_ion.SPM
