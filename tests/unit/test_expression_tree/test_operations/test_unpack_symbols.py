#
# Tests for the jacobian methods
#
import pybamm
import unittest


class TestSymbolUnpacker(unittest.TestCase):
    def test_basic_symbols(self):
        a = pybamm.Scalar(1)
        unpacker = pybamm.SymbolUnpacker(pybamm.Scalar)

        unpacked = unpacker.unpack_symbol(a)
        self.assertEqual(unpacked, {a.id: a})

        b = pybamm.Parameter("b")
        unpacker_param = pybamm.SymbolUnpacker(pybamm.Parameter)

        unpacked = unpacker_param.unpack_symbol(a)
        self.assertEqual(unpacked, {})

        unpacked = unpacker_param.unpack_symbol(b)
        self.assertEqual(unpacked, {b.id: b})

    def test_binary(self):
        a = pybamm.Scalar(1)
        b = pybamm.Parameter("b")

        unpacker = pybamm.SymbolUnpacker(pybamm.Scalar)
        unpacked = unpacker.unpack_symbol(a + b)
        # Can't check dictionary directly so check ids
        self.assertEqual(unpacked.keys(), {a.id: a}.keys())
        self.assertEqual(unpacked[a.id].id, a.id)

        unpacker_param = pybamm.SymbolUnpacker(pybamm.Parameter)
        unpacked = unpacker_param.unpack_symbol(a + b)
        # Can't check dictionary directly so check ids
        self.assertEqual(unpacked.keys(), {b.id: b}.keys())
        self.assertEqual(unpacked[b.id].id, b.id)

    def test_unpack_list_of_symbols(self):
        a = pybamm.Scalar(1)
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")

        unpacker = pybamm.SymbolUnpacker(pybamm.Parameter)
        unpacked = unpacker.unpack_list_of_symbols([a + b, a - c, b + c])
        # Can't check dictionary directly so check ids
        self.assertEqual(unpacked.keys(), {b.id: b, c.id: c}.keys())
        self.assertEqual(unpacked[b.id].id, b.id)
        self.assertEqual(unpacked[c.id].id, c.id)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
