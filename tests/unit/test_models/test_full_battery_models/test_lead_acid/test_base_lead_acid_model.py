#
# Tests for the base lead acid model class
#
from tests import TestCase
import pybamm
import unittest


class TestBaseLeadAcidModel(TestCase):
    def test_default_geometry(self):
        model = pybamm.lead_acid.BaseModel({"dimensionality": 0})
        self.assertEqual(
            model.default_geometry["current collector"]["z"]["position"], 1
        )
        model = pybamm.lead_acid.BaseModel({"dimensionality": 1})
        self.assertEqual(model.default_geometry["current collector"]["z"]["min"], 0)
        model = pybamm.lead_acid.BaseModel({"dimensionality": 2})
        self.assertEqual(model.default_geometry["current collector"]["y"]["min"], 0)

    def test_incompatible_options(self):
        with self.assertRaisesRegex(
            pybamm.OptionError,
            "Lead-acid models can only have thermal effects if dimensionality is 0.",
        ):
            pybamm.lead_acid.BaseModel({"dimensionality": 1, "thermal": "lumped"})
        with self.assertRaisesRegex(pybamm.OptionError, "SEI"):
            pybamm.lead_acid.BaseModel({"SEI": "constant"})
        with self.assertRaisesRegex(pybamm.OptionError, "lithium plating"):
            pybamm.lead_acid.BaseModel({"lithium plating": "reversible"})


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
