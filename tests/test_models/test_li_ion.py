#
# Tests for the li-ion models
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import unittest


class TestLiIonSPM(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.li_ion.SPM()
        modeltest = tests.StandardModelTest(model)

        modeltest.test_all()
