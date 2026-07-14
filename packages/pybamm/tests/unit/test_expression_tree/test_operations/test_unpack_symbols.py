#
# Tests for the symbol unpacker
#
import pybamm


class TestSymbolUnpacker:
    def test_basic_symbols(self):
        a = pybamm.Scalar(1)
        unpacker = pybamm.SymbolUnpacker(pybamm.Scalar)

        unpacked = unpacker.unpack_symbol(a)
        assert unpacked == set([a])

        b = pybamm.Parameter("b")
        unpacker_param = pybamm.SymbolUnpacker(pybamm.Parameter)

        unpacked = unpacker_param.unpack_symbol(a)
        assert unpacked == set()

        unpacked = unpacker_param.unpack_symbol(b)
        assert unpacked == set([b])

    def test_binary(self):
        a = pybamm.Scalar(1)
        b = pybamm.Parameter("b")

        unpacker = pybamm.SymbolUnpacker(pybamm.Scalar)
        unpacked = unpacker.unpack_symbol(a + b)
        assert unpacked == set([a])

        unpacker_param = pybamm.SymbolUnpacker(pybamm.Parameter)
        unpacked = unpacker_param.unpack_symbol(a + b)
        assert unpacked == set([b])

    def test_unpack_list_of_symbols(self):
        a = pybamm.Scalar(1)
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")

        unpacker = pybamm.SymbolUnpacker(pybamm.Parameter)
        unpacked = unpacker.unpack_list_of_symbols([a + b, a - c, b + c])
        assert unpacked == set([b, c])

    def test_unpack_parameter_values(self):
        a = pybamm.Scalar(1)
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")

        param_vals = {"x": a + b, "y": 0.5, "z": a - c}
        unpacker = pybamm.SymbolUnpacker(pybamm.Parameter)
        unpacked = unpacker.unpack_parameter_values(param_vals)
        assert unpacked == {b, c}
