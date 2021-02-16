"""
Tests for the update_parameter_sets_doc.py
"""
from pybamm.parameters.update_parameter_sets_doc import get_ps_dict, generate_ps_doc
import pybamm
import unittest


DOC = """
Parameter sets from papers. The 'citation' entry provides a reference to the appropriate
paper in the file "pybamm/CITATIONS.txt". To see which parameter sets have been used in
your simulation, add the line "pybamm.print_citations()" to your script.

Lead-acid parameter sets
------------------------
    * Sulzer2019 :
       - Valentin Sulzer, S. Jon Chapman, Colin P. Please, David A. Howey, and Charles
         W. Monroe. Faster Lead-Acid Battery Simulations from Porous-Electrode Theory:
         Part I. Physical Model. Journal of The Electrochemical Society,
         166(12):A2363–A2371, 2019. doi:10.1149/2.0301910jes.

Lithium-ion parameter sets
--------------------------
    * Prada2013 :
       - Chang-Hui Chen, Ferran Brosa Planella, Kieran O'Regan, Dominika Gastol, W.
         Dhammika Widanage, and Emma Kendrick. Development of Experimental Techniques
         for Parameterization of Multi-scale Lithium-ion Battery Models. Journal of The
         Electrochemical Society, 167(8):080534, 2020. doi:10.1149/1945-7111/ab9050.
       - Michael J. Lain, James Brandon, and Emma Kendrick. Design strategies for high
         power vs. high energy lithium ion cells. Batteries, 5(4):64, 2019.
         doi:10.3390/batteries5040064.
       - Eric Prada, D. Di Domenico, Y. Creff, J. Bernard, Valérie Sauvant-Moynot, and
         François Huet. A simplified electrochemical and thermal aging model of
         LiFePO4-graphite Li-ion batteries: power and capacity fade simulations. Journal
         of The Electrochemical Society, 160(4):A616, 2013. doi:10.1149/2.053304jes.
"""

AUTHOR_YEAR_DICT = {
    "lithium-ion": [
        ("Prada2013", ["Chen2020", "Lain2019", "Prada2013"]),
    ],
    "lead-acid": [("Sulzer2019", ["Sulzer2019physical"])],
}


class TestUpdateParameterSetsDoc(unittest.TestCase):
    def test_get_ps_dict(self):
        output = get_ps_dict()
        for key, values in AUTHOR_YEAR_DICT.items():
            output_values = output.get(key, [])
            for value in values:
                self.assertIn(value, output_values)

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
