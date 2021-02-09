"""
Tests for the update_parameter_sets_doc.py
"""
from pybamm.parameters.update_parameter_sets_doc import generate_ps_doc
import pybamm
import unittest


DOC = """
Parameter sets from papers. The 'citation' entry provides a reference to the appropriate
paper in the file "pybamm/CITATIONS.txt". To see which parameter sets have been used in
your simulation, add the line "pybamm.print_citations()" to your script.

Lead-acid parameter sets
------------------------
    * Sulzer2019 :
        Valentin Sulzer, S. Jon Chapman, Colin P. Please, David A. Howey, and Charles W.
        Monroe. Faster Lead-Acid Battery Simulations from Porous-Electrode Theory: Part
        I. Physical Model. Journal of The Electrochemical Society, 166(12):A2363–A2371,
        2019. doi:10.1149/2.0301910jes.

Lithium-ion parameter sets
--------------------------
    * Ecker2015 :
        Madeleine Ecker, Thi Kim Dung Tran, Philipp Dechent, Stefan Käbitz, Alexander
        Warnecke, and Dirk Uwe Sauer. Parameterization of a Physico-Chemical Model of a
        Lithium-Ion Battery: I. Determination of Parameters. Journal of the
        Electrochemical Society, 162(9):A1836–A1848, 2015.
        doi:10.1149/2.0551509jes.Giles Richardson, Ivan Korotkin, Rahifa Ranom, Michael
        Castle, and Jamie M. Foster. Generalised single particle models for high-rate
        operation of graded lithium-ion electrodes: systematic derivation and
        validation. Electrochimica Acta, 339:135862, 2020.
        doi:10.1016/j.electacta.2020.135862.
    * NCA_Kim2011 :
        Gi-Heon Kim, Kandler Smith, Kyu-Jin Lee, Shriram Santhanagopalan, and Ahmad
        Pesaran. Multi-domain modeling of lithium-ion batteries encompassing
        multi-physics in varied length scales. Journal of the Electrochemical Society,
        158(8):A955–A969, 2011. doi:10.1149/1.3597614.
"""

AUTHOR_YEAR_DICT = {
    "lithium-ion": [
        ("NCA_Kim2011", ["Kim2011"]),
        ("Ecker2015", ["Ecker2015i", "Richardson2020"]),
    ],
    "lead-acid": [("Sulzer2019", ["Sulzer2019physical"])],
}


class TestUpdateParameterSetsDoc(unittest.TestCase):
    def test_generate_ps_doc(self):
        output = generate_ps_doc(AUTHOR_YEAR_DICT)
        self.assertEqual(output, DOC)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
