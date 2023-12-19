#
# Test base submodel
#
from tests import TestCase

import pybamm
import unittest


class TestBaseSubModel(TestCase):
    def test_domain(self):
        # Accepted string
        submodel = pybamm.BaseSubModel(None, "negative", phase="primary")
        self.assertEqual(submodel.domain, "negative")

        # None
        submodel = pybamm.BaseSubModel(None, None)
        self.assertEqual(submodel.domain, None)

        # bad string
        with self.assertRaises(pybamm.DomainError):
            pybamm.BaseSubModel(None, "bad string")

    def test_phase(self):
        # Without domain
        submodel = pybamm.BaseSubModel(None, None)
        self.assertEqual(submodel.phase, None)

        with self.assertRaisesRegex(ValueError, "Phase must be None"):
            pybamm.BaseSubModel(None, None, phase="primary")

        # With domain
        submodel = pybamm.BaseSubModel(None, "negative", phase="primary")
        self.assertEqual(submodel.phase, "primary")
        self.assertEqual(submodel.phase_name, "")

        submodel = pybamm.BaseSubModel(
            None, "negative", options={"particle phases": "2"}, phase="secondary"
        )
        self.assertEqual(submodel.phase, "secondary")
        self.assertEqual(submodel.phase_name, "secondary ")

        with self.assertRaisesRegex(ValueError, "Phase must be 'primary'"):
            pybamm.BaseSubModel(None, "negative", phase="secondary")
        with self.assertRaisesRegex(ValueError, "Phase must be either 'primary'"):
            pybamm.BaseSubModel(
                None, "negative", options={"particle phases": "2"}, phase="tertiary"
            )
        with self.assertRaisesRegex(ValueError, "Phase must be 'primary'"):
            # 2 phases in the negative but only 1 in the positive
            pybamm.BaseSubModel(
                None,
                "positive",
                options={"particle phases": ("2", "1")},
                phase="secondary",
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
