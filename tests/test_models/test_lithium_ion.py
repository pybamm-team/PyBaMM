#
# Tests for the lithium-ion models
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import numpy as np
import unittest


class TestSPM(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lithium_ion.SPM()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all(t_eval=np.linspace(0, 1))


class TestSPMe(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lithium_ion.SPMe()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()


# class TestDFN(unittest.TestCase):
#     def test_basic_processing(self):
#         model = pybamm.lithium_ion.DFN()
#         modeltest = tests.StandardModelTest(model)
#         modeltest.test_all()
