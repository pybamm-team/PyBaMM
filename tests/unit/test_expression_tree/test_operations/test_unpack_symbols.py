#
# Tests for the symbol unpacker
#
from tests import TestCase
import pybamm
import unittest


class TestSymbolUnpacker(TestCase):
    def test_basic_symbols(self):
        a = pybamm.Scalar(1)
        unpacker = pybamm.SymbolUnpacker(pybamm.Scalar)

        unpacked = unpacker.unpack_symbol(a)
        self.assertEqual(unpacked, set([a]))

        b = pybamm.Parameter("b")
        unpacker_param = pybamm.SymbolUnpacker(pybamm.Parameter)

        unpacked = unpacker_param.unpack_symbol(a)
        self.assertEqual(unpacked, set())

        unpacked = unpacker_param.unpack_symbol(b)
        self.assertEqual(unpacked, set([b]))

    def test_binary(self):
        a = pybamm.Scalar(1)
        b = pybamm.Parameter("b")

        unpacker = pybamm.SymbolUnpacker(pybamm.Scalar)
        unpacked = unpacker.unpack_symbol(a + b)
        self.assertEqual(unpacked, set([a]))

        unpacker_param = pybamm.SymbolUnpacker(pybamm.Parameter)
        unpacked = unpacker_param.unpack_symbol(a + b)
        self.assertEqual(unpacked, set([b]))

    def test_unpack_list_of_symbols(self):
        a = pybamm.Scalar(1)
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")

        unpacker = pybamm.SymbolUnpacker(pybamm.Parameter)
        unpacked = unpacker.unpack_list_of_symbols([a + b, a - c, b + c])
        self.assertEqual(unpacked, set([b, c]))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
