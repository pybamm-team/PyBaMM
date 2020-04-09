#
# Tests for the base lead acid model class
#
import pybamm
import unittest


class TestBaseLeadAcidModel(unittest.TestCase):
    def test_default_geometry(self):
        var = pybamm.standard_spatial_vars

        model = pybamm.lead_acid.BaseModel({"dimensionality": 0})
        self.assertEqual(
            model.default_geometry["current collector"]["primary"][var.z][
                "position"
            ].id,
            pybamm.Scalar(1).id,
        )
        model = pybamm.lead_acid.BaseModel({"dimensionality": 1})
        self.assertEqual(
            model.default_geometry["current collector"]["primary"][var.z]["min"].id,
            pybamm.Scalar(0).id,
        )
        model = pybamm.lead_acid.BaseModel({"dimensionality": 2})
        self.assertEqual(
            model.default_geometry["current collector"]["primary"][var.y]["min"].id,
            pybamm.Scalar(0).id,
        )

    def test_incompatible_options(self):
        with self.assertRaisesRegex(
            pybamm.OptionError,
            "Lead-acid models can only have thermal " "effects if dimensionality is 0.",
        ):
            pybamm.lead_acid.BaseModel({"dimensionality": 1, "thermal": "x-full"})


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
