from pybat_lead_acid.solutions import *
import unittest

class TestSolutions(unittest.TestCase):

    def test_square(self):
        self.assertEqual(square(3), 9)

if __name__ == '__main__':
    unittest.main()
