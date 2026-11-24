#
# Tests for the flat node-table serialisation
#
import json

import numpy as np
import pytest
from hypothesis import assume, given, settings

import pybamm
from pybamm.expression_tree.operations.serialise import (
    Serialise,
    convert_symbol_from_json,
)
from pybamm.expression_tree.operations.serialise_flat import (
    FlatEncoder,
    decode_nodes,
    to_nested,
)
from pybamm.expression_tree.operations.serialise_kernel import (
    SerialisationError,
    encode,
)
from tests.strategies import serialisation_settings
from tests.strategies.serialise_values import model_options_kwargs
from tests.strategies.symbols import symbols


def as_json(document):
    return json.loads(json.dumps(document, default=Serialise._json_encoder))


def decode_root(document, root):
    return convert_symbol_from_json(root, decode_nodes(document))


class _Fields(pybamm.Symbol):
    """A symbol whose fields hold a tuple, a dict and an array."""

    def __init__(self, child, pair, mapping, weights):
        super().__init__("fields", children=[child])
        self.pair = pair
        self.mapping = mapping
        self.weights = weights


class TestFlatEncoder:
    def test_shared_term_stored_once(self):
        a = pybamm.Parameter("a")
        shared = a * 2
        expression = shared + pybamm.exp(a * 2)
        encoder = FlatEncoder()
        reference = encoder.encode(expression)
        tables = encoder.tables()
        # a, 2, a * 2, exp(a * 2), sum
        assert len(tables["nodes"]) == 5
        assert reference == {"$ref": 4}

    def test_nodes_only_reference_earlier_nodes(self):
        a = pybamm.Variable("a", domain="negative electrode")
        encoder = FlatEncoder()
        encoder.encode(pybamm.grad(a) * pybamm.x_average(a))
        for position, node in enumerate(encoder.tables()["nodes"]):
            for child in node.get("children", []):
                assert child["$ref"] < position

    def test_round_trip_matches_nested(self):
        a = pybamm.Variable("a", domain="negative electrode")
        expression = pybamm.div(pybamm.grad(a)) + pybamm.Scalar(3) * a
        encoder = FlatEncoder()
        document = as_json({"root": encoder.encode(expression), **encoder.tables()})
        assert to_nested(document) == {"root": as_json(encode(expression))}
        assert decode_root(document, document["root"]) == expression

    def test_non_symbol_fields_round_trip(self):
        vector = pybamm.Vector(np.array([1.0, 2.0]))
        infinity = pybamm.Scalar(float("inf"))
        encoder = FlatEncoder(default=Serialise._json_encoder)
        roots = [encoder.encode(vector), encoder.encode(infinity)]
        document = as_json({"roots": roots, **encoder.tables()})
        nodes = decode_nodes(document)
        decoded = [convert_symbol_from_json(root, nodes) for root in document["roots"]]
        np.testing.assert_array_equal(decoded[0].entries, vector.entries)
        assert decoded[1].value == float("inf")

    def test_unserialisable_object_raises(self):
        with pytest.raises(SerialisationError, match="no codec registered"):
            FlatEncoder().encode(object())

    def test_container_and_leaf_fields_round_trip(self):
        a = pybamm.Parameter("a")
        symbol = _Fields(a, (a, 1), {"key": a}, np.array([1.0, 2.0]))
        encoder = FlatEncoder(default=Serialise._json_encoder)
        document = as_json({"root": encoder.encode(symbol), **encoder.tables()})
        # symbols inside the tuple and dict fields are references, not copies
        assert len(document["nodes"]) == 2
        assert to_nested(document) == {"root": as_json(encode(symbol))}

        restored = decode_root(document, document["root"])
        assert restored.pair == (a, 1)
        assert restored.pair[0] is restored.children[0]
        assert restored.mapping["key"] is restored.children[0]
        np.testing.assert_array_equal(restored.weights, symbol.weights)


class TestDecodeNodes:
    def test_shared_term_decoded_once(self):
        a = pybamm.Variable("a")
        expression = (a * 2) + pybamm.exp(a * 2)
        encoder = FlatEncoder()
        document = as_json({"root": encoder.encode(expression), **encoder.tables()})
        decoded = decode_root(document, document["root"])
        assert decoded.children[0] is decoded.children[1].children[0]

    def test_reference_is_an_index_into_the_nodes(self):
        a = pybamm.Parameter("a")
        encoder = FlatEncoder()
        reference = encoder.encode(a + 1)
        nodes = decode_nodes(as_json(encoder.tables()))
        assert convert_symbol_from_json(reference, nodes) is nodes[reference["$ref"]]

    def test_document_without_nodes_has_no_nodes(self):
        assert decode_nodes({"model": {}}) == []

    def test_missing_table_raises(self):
        with pytest.raises(SerialisationError, match="missing the 'types' table"):
            decode_nodes({"nodes": [], "domains": []})

    def test_type_without_codec_raises(self):
        document = {"types": ["builtins.dict"], "domains": [], "nodes": [{"$type": 0}]}
        with pytest.raises(SerialisationError, match="No codec to decode"):
            decode_nodes(document)

    def test_invalid_reference_raises(self):
        with pytest.raises(SerialisationError, match="Invalid node reference"):
            convert_symbol_from_json({"$ref": 3}, [])


DEGRADATION_OPTIONS = {
    "SEI": "solvent-diffusion limited",
    "lithium plating": "partially reversible",
    "particle mechanics": "swelling and cracking",
    "thermal": "lumped",
}
MODELS = {
    "lithium_ion.BaseModel": pybamm.lithium_ion.BaseModel,
    "lithium_ion.BasicDFN": pybamm.lithium_ion.BasicDFN,
    "lithium_ion.BasicDFN2D": pybamm.lithium_ion.BasicDFN2D,
    "lithium_ion.BasicDFNComposite": pybamm.lithium_ion.BasicDFNComposite,
    "lithium_ion.BasicDFNHalfCell": pybamm.lithium_ion.BasicDFNHalfCell,
    "lithium_ion.BasicDFNUnstructured": pybamm.lithium_ion.BasicDFNUnstructured,
    "lithium_ion.BasicSPM": pybamm.lithium_ion.BasicSPM,
    "lithium_ion.DFN": pybamm.lithium_ion.DFN,
    "lithium_ion.DFN degradation": lambda: pybamm.lithium_ion.DFN(DEGRADATION_OPTIONS),
    "lithium_ion.ElectrodeSOHHalfCell": pybamm.lithium_ion.ElectrodeSOHHalfCell,
    "lithium_ion.MPM": pybamm.lithium_ion.MPM,
    "lithium_ion.NewmanTobias": pybamm.lithium_ion.NewmanTobias,
    "lithium_ion.SPM": pybamm.lithium_ion.SPM,
    "lithium_ion.SPMe": pybamm.lithium_ion.SPMe,
    "lithium_ion.SplitOCVR": pybamm.lithium_ion.SplitOCVR,
    "lithium_ion.Yang2017": pybamm.lithium_ion.Yang2017,
    "lead_acid.BaseModel": pybamm.lead_acid.BaseModel,
    "lead_acid.BasicFull": pybamm.lead_acid.BasicFull,
    "lead_acid.Full": pybamm.lead_acid.Full,
    "lead_acid.LOQS": pybamm.lead_acid.LOQS,
    "sodium_ion.BasicDFN": pybamm.sodium_ion.BasicDFN,
    "equivalent_circuit.Thevenin": pybamm.equivalent_circuit.Thevenin,
}


def as_bytes(document):
    return json.dumps(document, default=Serialise._json_encoder).encode()


def as_legacy(flat: bytes) -> dict:
    """The schema 1.1 (nested) document equivalent to a flat one."""
    document = to_nested(json.loads(flat))
    document["schema_version"] = "1.1"
    return document


class TestFlatModel:
    def test_model_document_layout(self):
        document = Serialise.serialise_custom_model(pybamm.lithium_ion.SPM())
        assert document["schema_version"] == "2.0"
        assert {"types", "domains", "nodes", "model"} <= document.keys()

    @pytest.mark.parametrize("build", MODELS.values(), ids=MODELS.keys())
    def test_loads_byte_identical(self, build):
        flat = as_bytes(Serialise.serialise_custom_model(build()))
        legacy = as_legacy(flat)
        assert len(flat) < len(as_bytes(legacy))

        # a reloaded model serialises to the same bytes
        loaded = Serialise.load_custom_model(json.loads(flat))
        assert as_bytes(Serialise.serialise_custom_model(loaded)) == flat

        # and so does one loaded from the equivalent legacy (nested) document
        from_legacy = Serialise.load_custom_model(legacy)
        assert from_legacy.schema_version == "1.1"
        assert as_bytes(Serialise.serialise_custom_model(from_legacy)) == flat

    @pytest.mark.parametrize("compress", [False, True])
    def test_file_round_trip_byte_identical(self, tmp_path, compress):
        model = pybamm.lithium_ion.DFN()
        first, second = tmp_path / "first.json", tmp_path / "second.json"
        Serialise.save_custom_model(model, first, compress=compress)
        loaded = Serialise.load_custom_model(str(first))
        assert loaded.schema_version == "2.0"
        Serialise.save_custom_model(loaded, second, compress=compress)
        assert second.read_bytes() == first.read_bytes()

    def test_legacy_file_upgrades_to_the_same_bytes(self, tmp_path):
        model = pybamm.lithium_ion.SPM()
        flat = as_bytes(model.to_json())
        old_file, new_file = tmp_path / "old.json", tmp_path / "new.json"
        old_file.write_bytes(as_bytes(as_legacy(flat)))
        pybamm.BaseModel.from_json(str(old_file)).to_json(str(new_file))
        assert json.loads(new_file.read_text()) == json.loads(flat)

    def test_config_round_trip_byte_identical(self):
        model = pybamm.lithium_ion.SPMe()
        loaded = pybamm.BaseModel.from_config(json.loads(as_bytes(model.to_config())))
        assert as_bytes(loaded.to_json()) == as_bytes(model.to_json())

    def test_legacy_and_flat_models_solve_identically(self):
        flat = as_bytes(Serialise.serialise_custom_model(pybamm.lithium_ion.SPM()))
        loaded = [
            Serialise.load_custom_model(document)
            for document in (as_legacy(flat), json.loads(flat))
        ]
        t_eval = np.linspace(0, 3600, 20)
        voltages = [
            pybamm.Simulation(m).solve([0, 3600])["Voltage [V]"](t_eval) for m in loaded
        ]
        np.testing.assert_array_equal(voltages[1], voltages[0])


class TestFlatProperties:
    @serialisation_settings
    @given(symbols(max_leaves=6))
    def test_symbol_round_trip_matches_nested(self, tree):
        nested = as_bytes(encode(tree))
        encoder = FlatEncoder(default=Serialise._json_encoder)
        document = json.loads(
            as_bytes({"root": encoder.encode(tree), **encoder.tables()})
        )
        assert as_bytes(to_nested(document)["root"]) == nested

        restored = decode_root(document, document["root"])
        assert restored.id == tree.id
        assert as_bytes(encode(restored)) == nested

    @serialisation_settings
    @given(symbols(max_leaves=6), symbols(max_leaves=6))
    def test_shared_table_across_roots(self, first, second):
        encoder = FlatEncoder(default=Serialise._json_encoder)
        roots = [encoder.encode(tree) for tree in (first, second, first)]
        assert roots[0] == roots[2]
        document = json.loads(as_bytes({"roots": roots, **encoder.tables()}))
        nodes = [as_bytes(node) for node in document["nodes"]]
        assert len(nodes) == len(set(nodes))
        nodes = decode_nodes(document)
        decoded = [convert_symbol_from_json(root, nodes) for root in document["roots"]]
        assert decoded[0] is decoded[2]
        assert [symbol.id for symbol in decoded[:2]] == [first.id, second.id]

    # each example builds and reloads a full model, so draw far fewer of them
    @settings(max_examples=25, deadline=None)
    @given(model_options_kwargs())
    def test_model_options_load_byte_identical(self, kwargs):
        model_class = kwargs.pop("_cls")
        try:
            model = model_class(options=kwargs)
        except (pybamm.OptionError, NotImplementedError, ValueError):
            assume(False)
        flat = as_bytes(Serialise.serialise_custom_model(model))
        for document in (json.loads(flat), as_legacy(flat)):
            loaded = Serialise.load_custom_model(document)
            assert as_bytes(Serialise.serialise_custom_model(loaded)) == flat
